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
from ..merge import merge_wavg, bipartite_soft_matching
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


def add_decomposed_rel_pos_new(
        attn: torch.Tensor,
        q: torch.Tensor,
        absolute_indices: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Make some adaptions after applying token merging.
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B*nHeads, N_reduced, C).
        absolute_indices (Tensor): Tensor that records the indices of merged tokens (B, N_reduced).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q BEFORE merging with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k BEFORE merging with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    gather = mps_gather_workaround if attn.device.type == "mps" else torch.gather

    _, N_reduced, dim = q.shape
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h) # (q_h, k_h, dim)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w) # (q_w, k_w, dim)

    # Transform absolute indices to height indices and width indices for further decomposed RPE extraction
    h_indices = absolute_indices // q_w # (B, N_reduced)
    w_indices = absolute_indices % q_w # (B, N_reduced)

    nHeads = torch.tensor(q.shape[0] // absolute_indices.shape[0], device=q.device)

    # As merging indices are same for all heads
    h_indices = h_indices.repeat_interleave(nHeads, dim=0) # (B*nHeads, N_reduced)
    w_indices = w_indices.repeat_interleave(nHeads, dim=0) # (B*nHeads, N_reduced)

    Rh_gathered = Rh[h_indices, :, :] # (B*nHeads, N_reduced, k_h, dim)
    Rw_gathered = Rw[w_indices, :, :] # (B*nHeads, N_reduced, k_w, dim)

    rel_h = torch.einsum("bnc,bnkc->bnk", q, Rh_gathered) # (B*nHeads, N_reduced, k_h)
    rel_w = torch.einsum("bnc,bnkc->bnk", q, Rw_gathered) # (B*nHeads, N_reduced, k_w)

    rel_h = gather(rel_h, dim=-1, index=h_indices.unsqueeze(1).expand(-1, N_reduced, -1)) # (B*nHeads, N_reduced, N_reduced)
    rel_w = gather(rel_w, dim=-1, index=w_indices.unsqueeze(1).expand(-1, N_reduced, -1)) # (B*nHeads, N_reduced, N_reduced)

    attn = attn + rel_h + rel_w

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

    def forward(self, x: torch.Tensor, ratio ) -> torch.Tensor:  
        B, H, W, _ = x.shape
        C = _ // self.num_heads

        x = x.reshape(B, H*W, -1) # (B, N, C * nHeads)

        # mean aggregation over multiple heads to reduce dimensions for similarity comparison
        # breakpoint()
        metric =  aggregate_over_head(x, num_heads=self.num_heads, option="mean") 

        x_merge, x_unmerge = bipartite_soft_matching(
            metric=metric, ratio=ratio
        )

        x_reduced, merged_indices = x_merge(x) # (B, N', C*nHeads)

        _, N_reduced, _ = x_reduced.shape 
        qkv = self.qkv(x_reduced)
        
        qkv = qkv.view(B, N_reduced, 3, self.num_heads, C).permute(2, 0, 3, 1, 4).reshape(3, B*self.num_heads, N_reduced, C)

        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos_new(
                attn, q, merged_indices,
                self.rel_pos_h, self.rel_pos_w, 
                (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.view(B, self.num_heads, N_reduced, -1).permute(0, 2, 1, 3).reshape(B, N_reduced, -1)
        x = self.proj(x)

        x = x_unmerge(x)  # (B, N, C*nHeads)
        x = x.reshape(B, H, W, -1) # (B, H, W, C*nHeads)

        return x, x_merge, x_unmerge 

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

        shortcut = x
        x_n = self.norm1(x)
        if self.window_size > 0:
            ws = self.window_size
            H_w, W_w = x_n.shape[1], x_n.shape[2]
            x_n_win, pad_hw = window_partition(x_n, ws)
            x_attn, merge, unmerge = self.attn(x_n_win, ratio)
            x_attn = window_unpartition(x_attn, ws, pad_hw, (H_w, W_w))
        else:

            x_attn, merge, unmerge = self.attn(x_n, ratio)   

        x = shortcut + x_attn                         

        
   
        x_seq = x.reshape(B, H_sp * W_sp, C)          
        self._tome_info["x_attn"] = x_seq
        self._tome_info["metric"] = metric

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
    trace_source : reserved for future source-tracking support.
    """
    assert algo in ("tome", "pitome"), f"algo must be 'tome' or 'pitome', got {algo!r}"
    assert 0 < ratio <= 1.0, "ratio must be in (0, 1]"

    tome_info = {
        "algo":   algo,
        "ratio":  ratio,   # scalar; rebuilt into a list each forward
        "margin": margin,
        "x_attn": None,
        "metric": None 
    }
    encoder.tome_info = tome_info

    # ── wrap encoder.forward to reset state before every pass ────────────────
    _orig_forward = encoder.__class__.forward

    def _patched_forward(self, x: torch.Tensor):
        n = len(self.blocks)
        r = self.tome_info["ratio"]
        self.tome_info["ratio"] = [r] * n

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
