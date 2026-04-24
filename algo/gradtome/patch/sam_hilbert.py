"""
GradToMe-Hilbert patch for SAM-1 ImageEncoderViT.

Merging strategy
----------------
Run standard attention on the FULL token set, then merge AFTER the attention
residual using a Hilbert-group selection criterion, run the MLP on the reduced
N' tokens, and unmerge after the MLP residual.

    block[i] flow (ratio < 1):
        norm1 → attn (full N, spatial/windowed)  → residual   [B, N, C]
        → hilbert_group_matching(ratio)                        select groups
        → norm2 → MLP                                          [B, N', C]
        → unmerge                                              [B, N, C]

    block[i] flow (ratio == 1): standard forward, no merge.

Token selection
---------------
1. Compute per-token mean |activation| → reshape to (H, W) spatial map.
2. Apply Sobel edge filter → gradient magnitude map (H, W).
3. Reorder to Hilbert-curve order → (N,) Hilbert sequence.
4. Divide into consecutive groups of size `group_size` (default 64 = 8×8).
5. Compute per-group mean Sobel magnitude.
6. Select the groups with LOWEST mean magnitude (homogeneous / smooth regions)
   as merge candidates; keep high-gradient groups intact.
7. Each merge group is pooled to one representative token (mean of the group).
   High-gradient group tokens pass through unchanged.

Speedup sources
  • MLP: O(N) → O(N')   where N' ≈ ratio * N
"""

import sys
import os
import math
import types
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

# ── resolve SAM-1 imports ─────────────────────────────────────────────────────
_here    = os.path.dirname(__file__)
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
from ..hilbert_utils import get_hilbert_order


# ─────────────────────────────────────────────────────────────────────────────
# Core: Hilbert-group soft matching
# ─────────────────────────────────────────────────────────────────────────────

def tile_stride_matching(
    x:          torch.Tensor,   # (B, N, C) in raster order, N = H*W
    H:          int,
    W:          int,
    r:      float,          # fraction of tokens to KEEP
    group_size: int = 4,        # tokens per Hilbert group; 4 (2×2) is fine-grained, 64 (8×8) is coarse
) -> Tuple[Callable, Callable]:
    """
    Partition tokens into Hilbert groups, rank groups by spatial gradient
    magnitude, and return (merge, unmerge) callables.

    merge(x_in)  → (x_merged, None)
        x_in   : (B, N, C)
        x_merged: (B, N', C)  where N' = n_keep*group_size + n_merge

    unmerge(x_out) → (B, N, C)
        Restores full spatial resolution; merge-group members receive the
        group representative's value (mean of the original group).
    """
    N = H * W
    assert N % group_size == 0, (
        f"N={N} (H={H}, W={W}) must be divisible by group_size={group_size}"
    )

    n_groups = N // group_size
    gs       = group_size

    # How many groups to merge  (each merge removes gs-1 tokens)
    if r <= 0 or n_groups <= 1:
        def _identity(x_in, mode=None):
            return x_in, None
        return _identity, lambda x_out: x_out

    n_merge = min(n_groups - 1, math.ceil(r / max(gs - 1, 1)))
    n_keep  = n_groups - n_merge

    with torch.no_grad():
        B, _, C = x.shape
        device  = x.device

        # ── 1. Full-channel Sobel gradient (matches merge.py get_sobel_gradient) ──
        # Flatten batch×channel into a single batch dim so we need only one kernel.
        x_bchw = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x_flat = x_bchw.reshape(B * C, 1, H, W)                       # (B*C, 1, H, W)

        sobel_x_k = torch.tensor(
            [[-1., 0, 1], [-2., 0, 2], [-1., 0, 1]],
            device=device, dtype=torch.float16
        ).view(1, 1, 3, 3)
        sobel_y_k = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            device=device, dtype=torch.float16
        ).view(1, 1, 3, 3)

        gx = F.conv2d(x_flat, sobel_x_k, padding=1).reshape(B, C, H, W)
        gy = F.conv2d(x_flat, sobel_y_k, padding=1).reshape(B, C, H, W)
        # L2 across channels → (B, N) raster
        sobel_flat = torch.sqrt((gx**2 + gy**2).mean(dim=1)).reshape(B, N)

        # ── 2. Reorder to Hilbert curve ───────────────────────────────────────
        # perm[hilbert_pos] = raster_idx  →  x_hilbert = x_raster[:, perm]
        perm = get_hilbert_order(H, W, device=device)             # (N,)
        sobel_hilbert = sobel_flat[:, perm]                        # (B, N) Hilbert order

        # ── 3. Per-group mean Sobel magnitude ─────────────────────────────────
        grp_sobel = sobel_hilbert.view(B, n_groups, gs).mean(-1)  # (B, n_groups)

        # ── 4. Rank groups ascending → lowest gradient = merge candidates ─────
        grp_rank   = grp_sobel.argsort(dim=-1)                    # (B, n_groups)
        merge_grps = grp_rank[:, :n_merge]                         # (B, n_merge)
        keep_grps  = grp_rank[:, n_merge:]                         # (B, n_keep)


        # ── 5. Raster token indices per Hilbert group ─────────────────────────
        # Hilbert group g occupies Hilbert positions [g*gs, (g+1)*gs).
        # perm[g*gs : (g+1)*gs] gives the raster indices for group g.
        group_raster = perm.view(n_groups, gs)                     # (n_groups, gs)

        # Merge group raster indices: (B, n_merge, gs)
        merge_raster = group_raster[merge_grps.reshape(-1)].reshape(B, n_merge, gs)
        merge_flat   = merge_raster.reshape(B, n_merge * gs)       # (B, n_merge*gs)

        # Keep group raster indices: (B, n_keep*gs)
        keep_raster = group_raster[keep_grps.reshape(-1)].reshape(B, n_keep, gs)
        keep_flat   = keep_raster.reshape(B, n_keep * gs)          # (B, n_keep*gs)

    n_keep_flat = n_keep * gs

    def merge(x_in: torch.Tensor, mode: str = None):
        Bx, Nx, Cx = x_in.shape
        # Gather keep-group tokens unchanged
        k_idx       = keep_flat.unsqueeze(-1).expand(Bx, n_keep_flat, Cx)
        keep_tokens = x_in.gather(1, k_idx)                        # (B, n_keep*gs, C)
        m_idx        = merge_flat.unsqueeze(-1).expand(Bx, n_merge * gs, Cx)
        if mode is None:
            merge_tokens = x_in.gather(1, m_idx).reshape(Bx, n_merge, gs, Cx)
            merge_repr = merge_tokens[:, :, 0, :]                     # (B, n_merge, C)
        elif mode == 'permute':
            merge_tokens = x_in.gather(1, m_idx).reshape(Bx, n_merge, gs, Cx)
            merge_repr = merge_tokens.transpose(-2,-3).reshape(Bx, -1 ,Cx)
        else:
            merge_tokens = x_in.gather(1, m_idx).reshape(Bx, n_merge, gs, Cx)
            merge_repr = merge_tokens.mean(dim=2)                     # (B, n_merge, C)

        merged = torch.cat([keep_tokens, merge_repr], dim=1)        # (B, N', C)
        return merged

    def unmerge(x_out: torch.Tensor) -> torch.Tensor:
        Bx, _, Cx = x_out.shape

        keep_out  = x_out[:, :n_keep_flat, :]                      # (B, n_keep*gs, C)
        merge_out = x_out[:, n_keep_flat:, :]                      # (B, n_merge, C)

        out = torch.zeros(Bx, N, Cx, device=x_out.device, dtype=x_out.dtype)

        # Scatter keep tokens back to their original raster positions
        k_idx = keep_flat.unsqueeze(-1).expand(Bx, n_keep_flat, Cx)
        out.scatter_(1, k_idx, keep_out)

        # Broadcast each group representative to all gs group members
        m_exp = merge_out.unsqueeze(2).expand(Bx, n_merge, gs, Cx)
        m_idx = merge_flat.unsqueeze(-1).expand(Bx, n_merge * gs, Cx)
        out.scatter_(1, m_idx, m_exp.reshape(Bx, n_merge * gs, Cx))

        return out

    return merge, unmerge


# ─────────────────────────────────────────────────────────────────────────────
# Patched Attention — standard forward, no k/v merge
# ─────────────────────────────────────────────────────────────────────────────

class ToMeSAMAttention(Attention):

    def forward(self, x: torch.Tensor, ratio: float) -> torch.Tensor:
        B, H, W, _ = x.shape
        C = _ // self.num_heads
        N = H * W

        x = x.reshape(B, N, -1)                                    # (B, N, C_total)

        qkv = self.qkv(x)
        qkv = (qkv.view(B, N, 3, self.num_heads, C)
                   .permute(2, 0, 3, 1, 4)
                   .reshape(3, B * self.num_heads, N, C))
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        x    = attn @ v

        x = (x.view(B, self.num_heads, N, -1)
               .permute(0, 2, 1, 3)
               .reshape(B, N, -1))
        x = self.proj(x)
        x = x.reshape(B, H, W, -1)

        return x, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Patched Block — Hilbert-group merge for MLP
# ─────────────────────────────────────────────────────────────────────────────

class ToMeSAMBlock(Block):
    """
    Full attention on all N tokens, then:
        norm2 → hilbert_group_matching → MLP → unmerge → residual
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H_sp, W_sp, C = x.shape
        info       = self._tome_info
        ratio      = info["ratio"].pop(0)
        group_size = info.get("group_size", 4)

        # ── standard attention (full tokens) ─────────────────────────────────
        shortcut = x
        x_n = self.norm1(x)
        if self.window_size > 0:
            ws = self.window_size
            x_n_win, pad_hw = window_partition(x_n, ws)
            x_attn, _, _    = self.attn(x_n_win, ratio)
            x_attn          = window_unpartition(x_attn, ws, pad_hw, (x_n.shape[1], x_n.shape[2]))
        else:
            x_attn, _, _ = self.attn(x_n, ratio)

        x     = shortcut + x_attn
        x_seq = x.reshape(B, H_sp * W_sp, C)
        x_norm = self.norm2(x_seq)

        # ── Hilbert-group MLP merge ───────────────────────────────────────────
        # Compute group selection on normalized features (richer, more uniform signal)
        x_merge, x_unmerge = hilbert_group_matching(
            x_norm, H_sp, W_sp, ratio, group_size=group_size
        )
        x_norm_merged, _ = x_merge(x_norm)
        x_seq = x_seq + x_unmerge(self.mlp(x_norm_merged))

        return x_seq.reshape(B, H_sp, W_sp, C)


# ─────────────────────────────────────────────────────────────────────────────
# apply_patch
# ─────────────────────────────────────────────────────────────────────────────

def apply_patch(
    encoder:    ImageEncoderViT,
    algo:       str   = "tome",
    ratio:      float = 0.9,
    margin:     float = 0.5,
    group_size: int   = 4,
    trace_source: bool = False,
) -> ImageEncoderViT:
    """
    Monkey-patch a SAM-1 ImageEncoderViT in-place with Hilbert-group MLP merging.

    Parameters
    ----------
    encoder    : sam.image_encoder
    algo       : 'tome' or 'pitome' (retained for API compatibility; only 'tome' used)
    ratio      : fraction of tokens to keep per block (0 < ratio ≤ 1)
    margin     : unused (kept for API compatibility)
    group_size : tokens per Hilbert group; must divide H*W (default 4 = 2×2 on 64×64)
    """
    assert algo in ("tome", "pitome"), f"algo must be 'tome' or 'pitome', got {algo!r}"
    assert 0 < ratio <= 1.0,          "ratio must be in (0, 1]"

    tome_info = {
        "algo":       algo,
        "ratio":      ratio,
        "margin":     margin,
        "group_size": group_size,
    }
    encoder.tome_info = tome_info

    # ── wrap encoder.forward to rebuild ratio list before every pass ──────────
    _orig_forward = encoder.__class__.forward

    def _patched_forward(self, x: torch.Tensor):
        n = len(self.blocks)
        r = self.tome_info["ratio"]
        self.tome_info["ratio"] = [r] * n
        result = _orig_forward(self, x)
        self.tome_info["ratio"] = r
        return result

    encoder.forward = types.MethodType(_patched_forward, encoder)

    # ── patch block and attention classes in-place ────────────────────────────
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
        f"[GradToMe-Hilbert] patched  algo={algo}  ratio={ratio}"
        f"  group_size={group_size}"
        f"  blocks={n_blocks} (global={n_global} local={n_blocks - n_global})"
        f"  strategy=full-attn / hilbert-group-mlp-merge"
    )
    return encoder
