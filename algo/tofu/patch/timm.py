import math
from typing import Tuple

import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
from timm.models.registry import register_model
from timm.models.vision_transformer import Attention, LayerScale, VisionTransformer

from .threshold_masking import ThresholdMasker, softmax_with_mask
from .utils import create_vision_transformer


class LTPMBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LTPMAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Add learned threshold masking modules for merging and pruning
        self.merge_masker = ThresholdMasker()
        self.prune_masker = ThresholdMasker()
        nn.init.constant_(self.merge_masker.threshold, 1)
        nn.init.constant_(self.prune_masker.threshold, 0)

    def forward(self, x, size, mask, viz) -> torch.Tensor:
        b, t, c = x.shape
        x_attn, metric, importance_scores = self.attn(self.norm1(x), size, mask)

        prune_mask = self.prune_masker(importance_scores)
        mask.clone()
        mask[mask.bool()] = prune_mask[mask.bool()]
        mask.clone()

        x = x + self.drop_path1(self.ls1((x_attn)))

        x = x * size
        src_x, dst_x = x[..., ::2, :], x[..., 1::2, :]
        src_s, dst_s = size[..., ::2, :], size[..., 1::2, :]
        src_m, dst_m = mask[..., ::2], mask[..., 1::2]
        src_viz, dst_viz = viz[..., ::2, :], viz[..., 1::2, :]

        metric = metric / metric.norm(dim=-1, keepdim=True)
        metric *= mask[..., None].detach()
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        scores[..., 0, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        merge_mask = self.merge_masker(node_max)
        unm_mask = torch.ones_like(merge_mask) - merge_mask

        unm_mask[~src_m.bool()] = src_m[~src_m.bool()]
        merge_x = src_x * merge_mask[..., None].detach()
        merge_s = src_s * merge_mask[..., None].detach()
        merge_viz = src_viz * merge_mask[..., None].detach()
        dst_idx = node_idx
        dst_x = dst_x.scatter_reduce(1, dst_idx.unsqueeze(-1).expand(dst_idx.shape + (c,)), merge_x, reduce="sum")
        dst_s = dst_s.scatter_reduce(1, dst_idx.unsqueeze(-1).expand(dst_idx.shape + (1,)), merge_s, reduce="sum")
        dst_viz = dst_viz.scatter_reduce(
            1, dst_idx.unsqueeze(-1).expand(dst_idx.shape + (t,)), merge_viz, reduce="amax"
        )
        x = torch.cat([src_x, dst_x], dim=1)
        size = torch.cat([src_s, dst_s], dim=1)
        mask = torch.cat([unm_mask, dst_m], dim=1)
        viz = torch.cat([src_viz, dst_viz], dim=1)
        x = x / size
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, size, mask, viz


class LTPMAttention(Attention):
    def forward(self, x, size, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Apply proportional attention
        attn = attn + size[:, None, None, :, 0].log()

        # Apply softmax with mask
        attn = softmax_with_mask(attn, mask, self.training)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Calculate importance scores
        importance_scores = torch.mean(attn, dim=(1, 2))  # mean attention
        importance_scores[..., 0] = math.inf

        # Calculate similarity scores
        similarity_scores = k.mean(1)

        return x, similarity_scores, importance_scores


class LTPMVisionTransformer(VisionTransformer):
    def __init__(self, tau=0.1, **kwargs):
        super().__init__(**kwargs)
        self.masks = []
        self.vizs = []
        for block in self.blocks:
            block.merge_masker.tau = tau

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        b, t, _ = x.shape
        self.masks = []
        self.vizs = []
        s = torch.ones_like(x[..., 0, None])
        m = torch.ones_like(x[..., 0])
        v = torch.eye(t, device=x.device)[None, ...].expand(b, t, t)
        for block in self.blocks:
            x, s, m, v = block(x, s, m, v)
            self.masks.append(m.clone())
            self.vizs.append(v.clone())
        x = self.norm(x)
        return x


## Register all vision transformer variants to which you want to apply ltpm here.
# VIT
@register_model
def ltpm_vit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=LTPMBlock, **kwargs)
    model = create_vision_transformer(
        LTPMVisionTransformer, "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


# DEIT
@register_model
def ltpm_deit_small_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, block_fn=LTPMBlock, **kwargs)
    model = create_vision_transformer(
        LTPMVisionTransformer, "deit_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def ltpm_deit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=LTPMBlock, **kwargs)
    model = create_vision_transformer(
        LTPMVisionTransformer, "deit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def ltpm_deit_base_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=LTPMBlock, **kwargs)
    model = create_vision_transformer(
        LTPMVisionTransformer, "deit_base_patch16_384", pretrained=pretrained, **model_kwargs
    )
    return model