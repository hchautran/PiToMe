import torch
import torch.nn as nn
from torch.distributions import RelaxedBernoulli

from .common import all_gather, get_rank, get_world_size


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, dim: int = -1) -> torch.Tensor:
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # Reparametrization trick
    y_soft = gumbels.softmax(dim)

    # Straight through, get differentiable hard_y
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(
        dim, index, 1.0
    )  # setting to 1 based on index's neighbors
    ret_hard = y_hard - y_soft.detach() + y_soft
    return y_soft, ret_hard


# def rerank(scores):
#     ratio = 0.5
#     scores = torch.where(scores.exp() > ratio, scores, torch.ones_like(scores, device=scores.device) * -torch.inf)
#     return scores
#     batch_size, num_tokens = scores.shape
#     h = w = int(math.sqrt(num_tokens))  # assume h=w,
#     new_scores = []
#     for idx, neighbor_indices in get_neighbor_indices(h, w):
#         # neighbor_indices.append(idx)
#         neighbor_scores = scores[:, neighbor_indices]
#         larger_neighbors = neighbor_scores.exp() > ratio
#         new_score = neighbor_scores.mean(1)
#         # new_score = scores[:, neighbor_indices].mean(1)
#         new_scores.append(new_score)
#     for idx, new_score in enumerate(new_scores):
#         scores[:, idx] = new_score
#     return scores


class TokenPruner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prune_layers = config.prune_layers
        embed_dim = getattr(config, "encoder_width", config.hidden_size)
        self.keep_ratios = {}
        if not self.prune_layers:
            # skip pruning
            return
        if self.config.keep_ratio >= 1 or self.config.keep_ratio <= 0:
            return

        self.keep_ratios = {layer: self.config.keep_ratio**i for i, layer in enumerate(self.prune_layers, 1)}
        # self.keep_ratios = {2: 0.5, 4:0.8, 6:0.75} # this is for simulating grounded pruning speedup, need to uncomment num_keep_tokens in the inference function to use this value

        if self.config.contrast_method == "states":
            self.text_projections = nn.ModuleDict(
                {str(layer): nn.Linear(config.hidden_size, embed_dim // 4) for layer in self.prune_layers}
            )
            self.image_projections = nn.ModuleDict(
                {str(layer): nn.Linear(embed_dim, embed_dim // 4) for layer in self.prune_layers}
            )

        if self.config.ib_kl:
            self.ib_temperatures = nn.ParameterDict(
                {str(layer): nn.Parameter(torch.ones([])) for layer in self.prune_layers}
            )

        if self.config.prune_method == "linear_states":
            self.token_predictors = nn.ModuleDict(
                {
                    str(layer): nn.Sequential(
                        nn.LayerNorm(embed_dim),
                        nn.Linear(embed_dim, 2),
                        nn.LogSoftmax(dim=-1),
                    )
                    for layer in self.prune_layers
                }
            )
        elif self.config.prune_method == "mlp_states":
            self.token_predictors = nn.ModuleDict(
                {
                    str(layer): nn.Sequential(
                        nn.LayerNorm(embed_dim),
                        nn.Linear(embed_dim, embed_dim),
                        nn.GELU(),
                        nn.Linear(embed_dim, embed_dim // 2),
                        nn.GELU(),
                        nn.Linear(embed_dim // 2, embed_dim // 4),
                        nn.GELU(),
                        nn.Linear(embed_dim // 4, 1 if self.config.ib_kl else 2),
                        nn.LogSigmoid() if self.config.ib_kl else nn.LogSoftmax(dim=-1),
                    )
                    for layer in self.prune_layers
                }
            )
        elif self.config.prune_method == "all_heads":
            self.token_predictors = nn.ModuleDict(
                {
                    str(layer): nn.Sequential(
                        nn.Linear(12, 2),
                        nn.LogSoftmax(dim=-1),
                    )
                    for layer in self.prune_layers
                }
            )
        elif self.config.prune_method in ["first_head", "mean_head"]:
            self.token_predictors = nn.ModuleDict(
                {
                    str(layer): nn.Sequential(
                        nn.Linear(1, 2),
                        nn.LogSoftmax(dim=-1),
                    )
                    for layer in self.prune_layers
                }
            )

    def forward(
        self, layer_idx, text_states, text_mask, image_states, image_mask, cross_attn, previous_keep_mask, **kwargs
    ):
        layer_keep_info = None
        if not self.prune_layers or layer_idx not in self.prune_layers or not self.keep_ratios:
            return image_states, image_mask, previous_keep_mask, layer_keep_info

        batch_size = text_states.shape[0]
        # text_len = text_states.shape[1]
        image_len = image_states.shape[1]  # include cls
        image_hidden_size = image_states.shape[-1]
        image_states_no_cls = image_states[:, 1:]
        cls_states = image_states[:, :1]
        cls_mask = image_mask[:, :1]

        # text_states = hidden_states[:, :text_len]
        # image_states = hidden_states[:, text_len + 1 :]
        # text_mask = co_masks[:, :text_len]

        # get true text len for each example
        t_len = text_mask.sum(1, keepdim=True).unsqueeze(-1)
        token_predictor = self.token_predictors[str(layer_idx)]
        if self.config.prune_method in ["linear_states", "mlp_states"]:
            token_scores = token_predictor(image_states_no_cls)  # [B, N, 2]
        elif self.config.prune_method == "all_heads":
            # txt2img_attention = attention[:, :, :text_len, text_len + 1 :]
            attn_scores = (cross_attn.sum(2) / t_len).transpose(1, 2)
            token_scores = token_predictor(attn_scores)
        elif self.config.prune_method == "first_head":
            # txt2img_attention = attention[:, :, :text_len, text_len + 1 :]
            # aggregate across text tokens for the first attention head
            attn_scores = (cross_attn[:, 0].sum(1) / t_len).unsqueeze(-1)
            token_scores = token_predictor(attn_scores)
        elif self.config.prune_method == "mean_head":
            # aggregate across text tokens for all attention heads
            # txt2img_attention = attention[:, :, :text_len, text_len + 1 :]
            attn_scores = (cross_attn.mean(1).sum(1) / t_len).unsqueeze(-1)
            token_scores = token_predictor(attn_scores)
        else:
            raise ValueError(f"prune_method {self.config.prune_method} not implemented!")

        if self.config.ib_kl:
            with torch.no_grad():
                self.ib_temperatures[str(layer_idx)].clamp_(0.001, 100)

        if self.training:
            if self.config.ib_kl:
                token_scores = token_scores.squeeze(-1)
                sample_distrib = RelaxedBernoulli(self.ib_temperatures[str(layer_idx)], logits=token_scores)
                keep_mask = sample_distrib.rsample() * previous_keep_mask
            else:
                # if True:
                # keep_mask = RelaxedBernoulli(logits=token_scores).rsample()
                # keep_mask = F.gumbel_softmax(token_scores, hard=True)[:, :, 0] * previous_keep_mask
                soft_mask, hard_mask = gumbel_softmax(token_scores)
                keep_mask = hard_mask[:, :, 0] * previous_keep_mask

            # use mask to remove prune tokens
            new_img_mask = torch.cat([cls_mask, keep_mask], dim=1)
            new_img_states = image_states

            previous_keep_mask = keep_mask

            if self.config.contrast_method == "states":
                pooled_txt_states = (text_states * text_mask.unsqueeze(-1)).sum(1) / (text_mask.sum(1, keepdim=True))
                text_feat = self.text_projections[str(layer_idx)](pooled_txt_states)

                pooled_img_states = (image_states_no_cls * keep_mask.unsqueeze(-1)).sum(1) / (
                    keep_mask.sum(1, keepdim=True)
                )
                image_feat = self.image_projections[str(layer_idx)](pooled_img_states)

                image_feat_all = all_gather(image_feat, get_rank(), get_world_size())
                text_feat_all = all_gather(text_feat, get_rank(), get_world_size())
                sim = text_feat_all @ image_feat_all.t()
            else:
                sim = None

            layer_keep_info = (
                (token_scores, sample_distrib) if self.config.ib_kl else token_scores[:, :, 0],
                sim,
            )

        else:
            # for inference
            scores = token_scores.squeeze(-1) if self.config.ib_kl else token_scores[:, :, 0]
            # scores = rerank(scores)
            # num_keep_tokens = int(image_len * self.keep_ratios[layer_idx])
            num_keep_tokens = int(image_len * self.config.keep_ratio)
            # NOTE: keep_idx is the token indices rather than keep_mask which has zeros and ones
            topk = torch.topk(scores, num_keep_tokens, dim=-1)
            keep_idx = topk.indices

            # ### only mask first or last prune layer
            # if layer_idx == 2:
            # # if layer_idx == 6:
            #     num_keep_tokens = int(image_len * (1 - self.config.keep_ratio))
            #     inverse_topk = torch.topk(scores, num_keep_tokens, largest=False, dim=-1)
            #     keep_idx = inverse_topk.indices

            # keep_idx = torch.randint(low=1, high=image_len-1, size=(batch_size, num_keep_tokens), device=scores.device) # for random idx experiments
            t_idx = keep_idx.unsqueeze(2).expand(batch_size, num_keep_tokens, image_hidden_size)

            img_states = image_states_no_cls.gather(1, t_idx)
            new_img_states = torch.cat([cls_states, img_states], dim=1)

            new_img_mask = torch.ones(
                (batch_size, num_keep_tokens + 1),
                dtype=torch.long,
                device=img_states.device,
            )
            # img_mask = image_mask_no_cls.gather(1, keep_idx)
            # new_img_mask = torch.cat([cls_mask, img_mask], dim=1)
            previous_keep_mask = keep_idx
            # print(f"{image_len=}, {num_keep_tokens=}, {layer_idx=}, {self.keep_ratios[layer_idx]=}")
            layer_keep_info = (scores, topk.values)
        return new_img_states, new_img_mask, previous_keep_mask, layer_keep_info