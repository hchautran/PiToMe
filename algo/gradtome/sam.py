"""GradToMe SAM patch: full attention with Hilbert-group merged K/V,
then merge → MLP → unmerge in each block."""

import sys
import os
import types
from typing import Optional, Tuple
import torch.nn.functional as F
import math
import torch

_here = os.path.dirname(__file__)
_sam_root = os.path.normpath(os.path.join(_here, '..', '..', '..', 'sam-hq'))
if _sam_root not in sys.path:
    sys.path.insert(0, _sam_root)

from segment_anything.modeling.image_encoder import (
    ImageEncoderViT,
    Block,
    Attention,
    add_decomposed_rel_pos,
    window_partition,
    window_unpartition,
)
from .merge import grad_bipartite_soft_matching
from .sam_hilbert import tile_stride_matching 
from .hilbert_utils import get_hilbert_inverse, get_hilbert_order

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """Extract / interpolate relative-position embeddings of length 2*max-1."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    merge,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """Decomposed rel-pos (mvitv2). If `merge` given, merges along the K axis
    to match merged keys before adding to `attn`."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).reshape(B, q_h*q_w, k_h)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).reshape(B, q_h*q_w, k_w)
    rel_pos = (rel_h[:, :, :, None] + rel_w[:, :, None, :]).reshape(B, q_h*q_w, k_h * k_w)

    if merge is not None:
        rel_pos = merge(rel_pos.transpose(-1, -2), mode=None)
        attn = attn + rel_pos.transpose(-1, -2)
    else:
        attn = attn + rel_pos
    return attn


def aggregate_over_head(x: torch.Tensor, num_heads: int, option: str = "mean") -> torch.Tensor:
    """Aggregate (B, N, H*C) → (B, N, C) by mean/max/sum over heads."""
    B, N, _ = x.shape
    metric = x.view(B, N, num_heads, -1)

    if option == "max":
        metric = metric.max(dim=2).values
    elif option == "mean":
        metric = metric.mean(dim=2)
    elif option == "sum":
        metric = metric.sum(dim=2)
    else:
        raise ValueError(f"Unknown aggregation option: {option}")

    return metric


class ToMeSAMAttention(Attention):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        C = _ // self.num_heads

        x = x.reshape(B, H*W, -1)
        _, N, _ = x.shape
        r = int(N * (1 - self._tome_info["ratio_scalar"]))

        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.num_heads, C).permute(2, 0, 3, 1, 4).reshape(3, B*self.num_heads, N, C)
        q, k, v = qkv.unbind(0)

        info = self._tome_info
        cache_key = info["cache_key"]
        x_merge = info[f"{cache_key}_merge"]
        x_unmerge = info[f"{cache_key}_unmerge"]

        if x_merge is None:
            x_merge, x_unmerge = tile_stride_matching(
                x=k, r=r, H=H, W=W
            )
            info[f"{cache_key}_merge"]   = x_merge
            info[f"{cache_key}_unmerge"] = x_unmerge

        k = x_merge(k, mode=None)
        v = x_merge(v, mode=None)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, x_merge,
                self.rel_pos_h, self.rel_pos_w,
                (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.view(B, self.num_heads, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
        x = self.proj(x)
        x = x.reshape(B, H, W, -1)

        return x


class ToMeSAMBlock(Block):
    """attn(full N) → merge → MLP(N') → unmerge. ratio==1: standard."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H_sp, W_sp, C = x.shape
        info  = self._tome_info
        ratio = info["ratio"].pop(0)
        info["ratio_scalar"] = ratio
        info["cache_key"]    = "local" if self.window_size > 0 else "global"

        shortcut = x
        x_n = self.norm1(x)

        if self.window_size > 0:
            ws = self.window_size
            H_w, W_w = x_n.shape[1], x_n.shape[2]
            x_n_win, pad_hw = window_partition(x_n, ws)
            x_attn_win = self.attn(x_n_win)
            x_attn = window_unpartition(x_attn_win, ws, pad_hw, (H_w, W_w))
        else:
            x_attn = self.attn(x_n)

        x = shortcut + x_attn
        x_seq = x.reshape(B, H_sp * W_sp, C)

        if ratio < 1.0 and self.window_size > 0:
            # Build a fresh merge here: the attn-cache was per-window+per-head.
            N = H_sp * W_sp
            r = int(N * (1 - ratio))
            x_merge, x_unmerge = tile_stride_matching(x=x_seq, r=r, H=H_sp, W=W_sp)
            x_merged = x_merge(x_seq, mode='mean')
            x_merged = x_merged + self.mlp(self.norm2(x_merged))
            x_seq    = x_unmerge(x_merged)
        else:
            x_seq = x_seq + self.mlp(self.norm2(x_seq))

        return x_seq.reshape(B, H_sp, W_sp, C)


def apply_patch(
    encoder: ImageEncoderViT,
    algo: str = "tome",
    ratio: float = 0.9,
    margin: float = 0.5,
    **_,
) -> ImageEncoderViT:
    """Patch SAM-1 image encoder for GradToMe (in-place).
    `**_` swallows extras forwarded by the registry (e.g. `mlp_merge`)."""
    assert algo in ("tome", "pitome"), f"algo must be 'tome' or 'pitome', got {algo!r}"
    assert 0 < ratio <= 1.0, "ratio must be in (0, 1]"

    tome_info = {
        "ratio":          ratio,
        "ratio_scalar":   ratio,
        "cache_key":      "local",
        "local_merge":    None,
        "local_unmerge":  None,
        "global_merge":   None,
        "global_unmerge": None,
    }
    encoder.tome_info = tome_info

    _orig_forward = encoder.__class__.forward

    def _patched_forward(self, x: torch.Tensor):
        n = len(self.blocks)
        r = self.tome_info["ratio"]
        self.tome_info["ratio"]          = [r] * n
        self.tome_info["local_merge"]    = None
        self.tome_info["local_unmerge"]  = None
        self.tome_info["global_merge"]   = None
        self.tome_info["global_unmerge"] = None

        result = _orig_forward(self, x)

        self.tome_info["ratio"] = r
        return result

    encoder.forward = types.MethodType(_patched_forward, encoder)

    for module in encoder.modules():
        if isinstance(module, Block) and not isinstance(module, ToMeSAMBlock):
            module.__class__  = ToMeSAMBlock
            module._tome_info = tome_info
        elif isinstance(module, Attention) and not isinstance(module, ToMeSAMAttention):
            module.__class__  = ToMeSAMAttention
            module._tome_info = tome_info

    n_blocks = len(encoder.blocks)
    n_global = sum(1 for blk in encoder.blocks if blk.window_size == 0)
    print(
        f"[ToMe-SAM] patched  algo={algo}  ratio={ratio}"
        + (f"  margin={margin}" if algo == "pitome" else "")
        + f"  blocks={n_blocks} (global={n_global} local={n_blocks-n_global})"
        + "  strategy=post-attn-merge / post-mlp-unmerge (all blocks)"
        + "  token-order=hilbert"
    )
    return encoder
