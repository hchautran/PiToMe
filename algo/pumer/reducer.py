import math
from typing import Callable, Tuple

import torch
import torch.nn as nn


def do_nothing(x, mode=None):
    return x


def bipartite_diff_matching(x, attn, r, mode="mean"):
    # reduce by a maximum of 50% tokens
    r = min(r, x.shape[1] // 2)
    if r <= 0:
        return x

    with torch.no_grad():
        a, b = attn[..., ::2], attn[..., 1::2]  # a: [bs, 36], b [bs, 36]
        scores = torch.abs(a.unsqueeze(-1) - b.unsqueeze(1))  # [bs, 36, 36]

        node_min, node_idx = scores.min(
            dim=-1
        )  # for each token in a, find most similar text-relevant (attn score difference is smallest) token in b, node_min is for a, node_idx is for b
        edge_idx = node_min.argsort(dim=-1)[..., None]  # [bs, 36, 1], get indices of top similar tokens

        unmerged_idx = edge_idx[..., r:, :]  # Unmerged Tokens, [bs, 36, 1]
        merged_idx = edge_idx[..., :r, :]  # Merged Tokens, [bs, 8, 1], indices for most similar tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=merged_idx)  # [bs, 8, 1], merged_idx in b

    src, dst = x[..., ::2, :], x[..., 1::2, :]
    n, t1, c = src.shape  # bs, 36, 768
    unm = src.gather(dim=-2, index=unmerged_idx.expand(n, t1 - r, c))
    src = src.gather(dim=-2, index=merged_idx.expand(n, r, c))
    dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

    return torch.cat([unm, dst], dim=1)


def bipartite_weighted_matching(x, metric, attn, r, mode="mean"):
    # reduce by a maximum of 50% tokens
    r = min(r, (x.shape[1] - 1) // 2)  # 1 for class token
    if r <= 0:
        return x

    with torch.no_grad():
        metric = metric * attn[..., None]
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]  # a: [bs, 36, 64], b [bs, 36, 64]
        scores = a @ b.transpose(-1, -2)  # [bs, 99, 98]

        scores[..., 0, :] = -math.inf  # cls token

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[
            ..., None
        ]  # [bs, 36, 1], get indices of top similar tokens

        unmerged_idx = edge_idx[..., r:, :]  # Unmerged Tokens, [bs, 36, 1]
        merged_idx = edge_idx[..., :r, :]  # Merged Tokens, [bs, 8, 1], indices for most similar tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=merged_idx)  # [bs, 8, 1], merged_idx in b
        unmerged_idx = unmerged_idx.sort(dim=1)[0]  # Sort to ensure the class token is at the start

    src, dst = x[..., ::2, :], x[..., 1::2, :]
    n, t1, c = src.shape  # bs, 36, 768
    unm = src.gather(dim=-2, index=unmerged_idx.expand(n, t1 - r, c))
    src = src.gather(dim=-2, index=merged_idx.expand(n, r, c))
    dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

    return torch.cat([unm, dst], dim=1)


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = True,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.

    When enabled, the class token won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]  # 197
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]  # a: [bs, 99, 64], b [bs, 98, 64]
        scores = a @ b.transpose(-1, -2)  # [bs, 99, 98]

        if class_token:
            scores[..., 0, :] = -math.inf

        node_max, node_idx = scores.max(
            dim=-1
        )  # for each token in a, find most similar token in b, node_max is for a, node_idx is for b
        edge_idx = node_max.argsort(dim=-1, descending=True)[
            ..., None
        ]  # [bs, 99, 1], get indices of top similar tokens

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, [bs, 91, 1]
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, [bs, 8, 1], indices for most similar tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # [bs, 8, 1], src_idx in b

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape  # 1, 99, 768
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    return merge


def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


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


class DyvitPruner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prune_layers = config.prune_layers
        if not self.prune_layers:
            # skip pruning
            return
        if self.config.keep_ratio >= 1 or self.config.keep_ratio <= 0:
            return

        embed_dim = getattr(config, "encoder_width", config.hidden_size)

        # TODO: for fair comparison use in_conv and out_conv as in https://github.com/raoyongming/DynamicViT/blob/master/models/dyvit.py
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
                    nn.Linear(embed_dim // 4, 2),
                    nn.LogSoftmax(dim=-1),
                )
                for layer in self.prune_layers
            }
        )

    def forward(
        self, layer_idx, text_states, text_mask, image_states, image_mask, cross_attn, previous_keep_mask, **kwargs
    ):
        layer_keep_info = None
        if not self.prune_layers or layer_idx not in self.prune_layers or not self.config.keep_ratio:
            return image_states, image_mask, previous_keep_mask, layer_keep_info

        batch_size = text_states.shape[0]
        # text_len = text_states.shape[1]
        image_len = image_states.shape[1]  # include cls
        image_hidden_size = image_states.shape[-1]
        image_states_no_cls = image_states[:, 1:]
        cls_states = image_states[:, :1]
        cls_mask = image_mask[:, :1]
        t_len = text_mask.sum(1, keepdim=True).unsqueeze(-1)
        token_predictor = self.token_predictors[str(layer_idx)]
        token_scores = token_predictor(image_states_no_cls)  # [B, N, 2]
        scores = token_scores[:, :, 0]
        if self.training:
            soft_mask, hard_mask = gumbel_softmax(token_scores)
            keep_mask = hard_mask[:, :, 0] * previous_keep_mask
            # use mask to remove prune tokens
            new_img_mask = torch.cat([cls_mask, keep_mask], dim=1)
            new_img_states = image_states

            previous_keep_mask = keep_mask
            layer_keep_info = (scores, None)
        else:
            # for inference
            num_keep_tokens = int(image_len * self.config.keep_ratio)
            # NOTE: keep_idx is the token indices rather than keep_mask which has zeros and ones
            topk = torch.topk(scores, num_keep_tokens, dim=-1)
            keep_idx = topk.indices

            t_idx = keep_idx.unsqueeze(2).expand(batch_size, num_keep_tokens, image_hidden_size)

            img_states = image_states_no_cls.gather(1, t_idx)
            new_img_states = torch.cat([cls_states, img_states], dim=1)

            new_img_mask = torch.ones(
                (batch_size, num_keep_tokens + 1),
                dtype=torch.long,
                device=img_states.device,
            )
            previous_keep_mask = keep_idx
            layer_keep_info = (scores, topk.values)
        return new_img_states, new_img_mask, previous_keep_mask, layer_keep_info


class TomeMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # merge no r, every layer, similarity metric,

    def forward(
        self, layer_idx, text_states, text_mask, image_states, image_mask, cross_attn, previous_keep_mask, **kwargs
    ):
        layer_keep_info = None
        if not self.prune_layers or layer_idx not in self.prune_layers or not self.config.keep_ratio:
            return image_states, image_mask, previous_keep_mask, layer_keep_info

        batch_size = text_states.shape[0]
        # text_len = text_states.shape[1]
        image_len = image_states.shape[1]  # include cls
        image_hidden_size = image_states.shape[-1]
        image_states_no_cls = image_states[:, 1:]
        cls_states = image_states[:, :1]
        cls_mask = image_mask[:, :1]
        t_len = text_mask.sum(1, keepdim=True).unsqueeze(-1)


class TokenReducer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # merge_n = 8
        # keep_ratio = 0.5

    def forward(
        self, layer_idx, text_states, text_mask, image_states, image_mask, cross_attn, previous_keep_mask, **kwargs
    ):
        layer_keep_info = None
        if not self.prune_layers or layer_idx not in self.prune_layers or not self.config.keep_ratio:
            return image_states, image_mask, previous_keep_mask, layer_keep_info
        """
        we use cross attention to remove and combine tokens
        """
        batch_size = text_states.shape[0]
        # text_len = text_states.shape[1]
        image_len = image_states.shape[1]  # include cls
        image_hidden_size = image_states.shape[-1]
        image_states_no_cls = image_states[:, 1:]
        cls_states = image_states[:, :1]
        cls_mask = image_mask[:, :1]
        t_len = text_mask.sum(1, keepdim=True).unsqueeze(-1)

        # elif self.config.prune_method == "all_heads":
        # txt2img_attention = attention[:, :, :text_len, text_len + 1 :]
        attn_scores = (cross_attn.sum(2) / t_len).transpose(1, 2)
        token_scores = attn_scores
        # elif self.config.prune_method == "first_head":
        #     # txt2img_attention = attention[:, :, :text_len, text_len + 1 :]
        #     # aggregate across text tokens for the first attention head
        #     attn_scores = (cross_attn[:, 0].sum(1) / t_len).unsqueeze(-1)
        #     token_scores = token_predictor(attn_scores)
        # elif self.config.prune_method == "mean_head":
        #     # aggregate across text tokens for all attention heads
        #     # txt2img_attention = attention[:, :, :text_len, text_len + 1 :]
        #     attn_scores = (cross_attn.mean(1).sum(1) / t_len).unsqueeze(-1)

        new_img_states, new_img_mask = None, None
        # TODO: use remover and combiner to do reduction

        return new_img_states, new_img_mask, previous_keep_mask, layer_keep_info