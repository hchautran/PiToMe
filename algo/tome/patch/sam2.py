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

import math
import sys
import os
import types
from typing import Tuple, Callable

import torch
import torch.nn.functional as F

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
from ..merge import bipartite_soft_matching, merge_wavg


# ─────────────────────────────────────────────────────────────────────────────
# Merge / unmerge helpers
# ─────────────────────────────────────────────────────────────────────────────

def _do_nothing(x, mode=None):
    return x

def _get_merge_unmerge(metric, ratio, algo, margin):
    return bipartite_soft_matching(metric, ratio, class_token=False)


# ─────────────────────────────────────────────────────────────────────────────
# Patched Attention — spatial forward (unchanged blocks) + seq forward (merged)
# ─────────────────────────────────────────────────────────────────────────────

class ToMeSAMAttention(Attention):
    """
    Spatial-only forward — returns (out, metric) instead of just out.
    Attention always runs on the full spatial grid (windowing and rel-pos intact).
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B,H,W,C] → (out [B,H,W,C], metric [B,N,hd])"""
        B, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
        attn = attn.softmax(dim=-1)

        out = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        out = self.proj(out)
        metric = q.reshape(B, self.num_heads, H * W, -1).mean(dim=1)  # [B, N, hd]
        return out, metric


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

        # ── norm1 + spatial attention (full N, windowing and rel-pos intact) ──
        shortcut = x
        x_n = self.norm1(x)

        if self.window_size > 0:
            H_w, W_w = x_n.shape[1], x_n.shape[2]
            x_n_win, pad_hw = window_partition(x_n, self.window_size)
            x_attn_win, metric_win = self.attn(x_n_win)  # [B*nw, ws², hd]
            x_attn = window_unpartition(x_attn_win, self.window_size, pad_hw, (H_w, W_w))
            _, _, hd = metric_win.shape
            ws = self.window_size
            m_sp = metric_win.reshape(-1, ws, ws, hd)
            m_sp = window_unpartition(m_sp, ws, pad_hw, (H_w, W_w))
            metric = m_sp.reshape(B, H_sp * W_sp, hd)    # [B, N, hd]
        else:
            x_attn, metric = self.attn(x_n)              # [B, H, W, C], [B, N, hd]

        x = shortcut + x_attn                            # attention residual [B, H, W, C]

        if ratio >= 1.0:
            x = x + self.mlp(self.norm2(x))
            return x

        # ── merge after attention residual ────────────────────────────────────
        x_seq = x.reshape(B, H_sp * W_sp, C)             # [B, N, C]
        breakpoint()
        merge, unmerge = _get_merge_unmerge(
            metric, ratio, info["algo"], info.get("margin", 0.5)
        )
        x_seq, _ = merge_wavg(merge, x_seq)              # [B, N', C]

        # ── norm2 + MLP on merged tokens ──────────────────────────────────────
        x_seq = x_seq + self.mlp(self.norm2(x_seq))      # [B, N', C]

        # ── unmerge ───────────────────────────────────────────────────────────
        x_seq = unmerge(x_seq)                           # [B, N, C]
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
            module.__class__     = ToMeSAMBlock
            module._tome_info    = tome_info
        elif isinstance(module, Attention) and not isinstance(module, ToMeSAMAttention):
            module.__class__ = ToMeSAMAttention

    n_blocks = len(encoder.blocks)
    n_global = sum(1 for blk in encoder.blocks if blk.window_size == 0)
    print(
        f"[ToMe-SAM] patched  algo={algo}  ratio={ratio}"
        + (f"  margin={margin}" if algo == "pitome" else "")
        + f"  blocks={n_blocks} (global={n_global} local={n_blocks-n_global})"
        + "  strategy=post-attn-merge / post-mlp-unmerge (all blocks)"
    )
    return encoder
