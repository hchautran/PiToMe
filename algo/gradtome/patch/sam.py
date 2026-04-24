"""
ToMe / PiToMe patch for SAM-1 ImageEncoderViT.

Merging strategy
----------------
Run attention on the FULL token set (preserving windowing and rel-pos), then
merge AFTER the attention residual using the current block's key metric, run the
MLP on the reduced N' tokens, and unmerge after the MLP residual.  Applies to
ALL blocks (both local/windowed and global).

    block[i] flow (ratio < 1):
        norm1 → attn (spatial/windowed) → residual          [B, N, C]
        → merge(metric_i)                                    [B, N', C]
        → norm2 → MLP → residual                            [B, N', C]
        → unmerge                                            [B, N, C]

    block[i] flow (ratio == 1): standard spatial forward, no merge.

Attention always runs on full N tokens so windowing and rel-pos are preserved.
Speedup comes from the MLP running on N' < N tokens.

Hilbert ordering
----------------
Before computing attention, tokens (the H*W sequence) are permuted to
Hilbert-curve order.  The attention logits are computed in raster order first
so that add_decomposed_rel_pos is still applied with the correct spatial
positions; then both axes of the attention matrix and the V tensor are permuted
to Hilbert order.  After attn @ V the output is inverse-permuted back to raster.

For dense attention this produces bit-identical results to raster order.  The
permutation prepares the token sequence so that spatially adjacent tokens are
also adjacent in the sequence index, making block-sparse patterns in Hilbert
order correspond directly to local spatial attention.

Speedup sources
  • MLP:  O(N)  →  O(N')

Supported algorithms
  'tome'   — bipartite soft matching on key-mean metric
  'pitome' — energy-score ranked bipartite matching (PiToMe)

Usage
-----
    from algo.tome.patch.sam import apply_patch as sam
    sam(encoder, algo='tome',   ratio=0.9)
    sam(encoder, algo='pitome', ratio=0.9, margin=0.5)
    encoder.tome_info['ratio'] = 0.8   # update at runtime
"""

import sys
import os
import types
from typing import Optional, Tuple
import torch.nn.functional as F
import math
import torch

# ── resolve SAM-1 imports ─────────────────────────────────────────────────────
_here = os.path.dirname(__file__)
_sam_root = os.path.normpath(os.path.join(_here, '..', '..', '..', '..', 'sam-hq'))
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
from ..merge import grad_bipartite_soft_matching
from .sam_hilbert import tile_stride_matching 
from ..hilbert_utils import get_hilbert_inverse, get_hilbert_order

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
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
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).reshape(B, q_h*q_w, k_h )
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).reshape(B, q_h*q_w, k_w)

    rel_pos =  (rel_h[:, : ,:, None] + rel_w[:, :, None, :]).reshape(B, q_h*q_w, k_h * k_w)
    

    if merge is not None:
        rel_pos  = merge(rel_pos.transpose(-1,-2), mode=None)
        attn = attn + rel_pos.transpose(-1,-2)
    else:
        attn = attn + rel_pos 
    return attn


def aggregate_over_head(x: torch.Tensor, num_heads: int, option: str = "mean") -> torch.Tensor:
    """
    Aggregates over multiple heads for computing similarity metric.

    Args:
        x: Input tokens with shape [B, N, C*num_heads]
        num_heads: Number of attention heads
        option: Aggregation method ('mean', 'max', 'sum')

    Returns:
        Aggregated tokens with shape [B, N, C]
    """
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

# ─────────────────────────────────────────────────────────────────────────────
# Patched Block — merge AFTER attention residual, unmerge AFTER MLP
# ─────────────────────────────────────────────────────────────────────────────

class ToMeSAMBlock(Block):
    """
    Block[i] forward (both local and global blocks):

    ratio < 1:
        1. norm1 → attn (spatial/windowed, full N)  → (x_attn [B,H,W,C], metric [B,N,hd])
        2. residual : x = x + x_attn                              [B, H, W, C]
        3. merge(metric) : x → [B, N', C]
        4. norm2 → MLP → residual                                 [B, N', C]
        5. unmerge → [B, H, W, C]

    ratio == 1:
        standard spatial forward, no merge.
    """

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

        if info["merge_mlp"] and ratio < 1.0:
            cache_key = info["cache_key"]
            x_merge   = info[f"{cache_key}_merge"]
            x_unmerge = info[f"{cache_key}_unmerge"]
            x_seq, _  = x_merge(x_seq, mode='mean')
            x_seq     = x_seq + self.mlp(self.norm2(x_seq))
            x_seq     = x_unmerge(x_seq)
        else:
            x_seq = x_seq + self.mlp(self.norm2(x_seq))

        return x_seq.reshape(B, H_sp, W_sp, C)


# ─────────────────────────────────────────────────────────────────────────────
# apply_patch
# ─────────────────────────────────────────────────────────────────────────────

def apply_patch(
    encoder: ImageEncoderViT,
    algo: str = "tome",
    ratio: float = 0.9,
    margin: float = 0.5,
    merge_mlp: bool = False,
    trace_source: bool = False,
) -> ImageEncoderViT:
    """
    Monkey-patch a SAM-1 ImageEncoderViT in-place to run ToMe / PiToMe.

    Parameters
    ----------
    encoder      : sam.image_encoder
    algo         : 'tome' or 'pitome'
    ratio        : fraction of tokens to keep per block  (0 < ratio ≤ 1).
                   Update at runtime via ``encoder.tome_info['ratio']``.
    margin       : PiToMe energy margin (ignored for ToMe).
    merge_mlp    : if True, also merge tokens before the MLP and unmerge after.
    trace_source : reserved for future source-tracking support.
    """
    assert algo in ("tome", "pitome"), f"algo must be 'tome' or 'pitome', got {algo!r}"
    assert 0 < ratio <= 1.0, "ratio must be in (0, 1]"

    tome_info = {
        "algo":           algo,
        "ratio":          ratio,   # scalar; rebuilt into a list each forward
        "margin":         margin,
        "merge_mlp":      merge_mlp,
        "ratio_scalar":   ratio,
        "cache_key":      "local",
        "local_merge":    None,
        "local_unmerge":  None,
        "global_merge":   None,
        "global_unmerge": None,
    }
    encoder.tome_info = tome_info

    # ── wrap encoder.forward to reset state before every pass ────────────────
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

        self.tome_info["ratio"] = r   # restore scalar for next call
        return result

    encoder.forward = types.MethodType(_patched_forward, encoder)

    # ── patch modules ─────────────────────────────────────────────────────────
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
