"""GradToMe-Hilbert SAM patch: full attention, then Hilbert-group merge
(low-Sobel-gradient groups pooled to one repr) → MLP → unmerge."""

import sys
import os
import math
import types
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

_here    = os.path.dirname(__file__)
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
from .hilbert_utils import get_hilbert_order


def tile_stride_matching(
    x: torch.Tensor, H: int, W: int,
    r: float,
    group_size: int = 4,
) -> Tuple[Callable, Callable]:
    """Hilbert-group merge/unmerge: rank groups by per-channel Sobel magnitude;
    lowest-gradient groups merge to one repr, others pass through."""
    N = H * W
    assert N % group_size == 0, (
        f"N={N} (H={H}, W={W}) must be divisible by group_size={group_size}"
    )

    n_groups = N // group_size
    gs       = group_size

    if r <= 0 or n_groups <= 1:
        def _identity(x_in, mode=None):
            return x_in, None
        return _identity, lambda x_out: x_out

    n_merge = min(n_groups - 1, math.ceil(r / max(gs - 1, 1)))
    n_keep  = n_groups - n_merge

    with torch.no_grad():
        B, _, C = x.shape
        device  = x.device

        x_bchw = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_flat = x_bchw.reshape(B * C, 1, H, W)

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
        sobel_flat = torch.sqrt((gx**2 + gy**2).mean(dim=1)).reshape(B, N)

        perm = get_hilbert_order(H, W, device=device)
        sobel_hilbert = sobel_flat[:, perm]

        grp_sobel = sobel_hilbert.view(B, n_groups, gs).mean(-1)
        grp_rank   = grp_sobel.argsort(dim=-1)
        merge_grps = grp_rank[:, :n_merge]
        keep_grps  = grp_rank[:, n_merge:]

        group_raster = perm.view(n_groups, gs)
        merge_raster = group_raster[merge_grps.reshape(-1)].reshape(B, n_merge, gs)
        merge_flat   = merge_raster.reshape(B, n_merge * gs)
        keep_raster = group_raster[keep_grps.reshape(-1)].reshape(B, n_keep, gs)
        keep_flat   = keep_raster.reshape(B, n_keep * gs)

    n_keep_flat = n_keep * gs

    def merge(x_in: torch.Tensor, mode: str = None):
        Bx, Nx, Cx = x_in.shape
        k_idx       = keep_flat.unsqueeze(-1).expand(Bx, n_keep_flat, Cx)
        keep_tokens = x_in.gather(1, k_idx)
        m_idx       = merge_flat.unsqueeze(-1).expand(Bx, n_merge * gs, Cx)
        if mode is None:
            merge_tokens = x_in.gather(1, m_idx).reshape(Bx, n_merge, gs, Cx)
            merge_repr = merge_tokens[:, :, 0, :]
        elif mode == 'permute':
            merge_tokens = x_in.gather(1, m_idx).reshape(Bx, n_merge, gs, Cx)
            merge_repr = merge_tokens.transpose(-2,-3).reshape(Bx, -1, Cx)
        else:
            merge_tokens = x_in.gather(1, m_idx).reshape(Bx, n_merge, gs, Cx)
            merge_repr = merge_tokens.mean(dim=2)

        return torch.cat([keep_tokens, merge_repr], dim=1)

    def unmerge(x_out: torch.Tensor) -> torch.Tensor:
        Bx, _, Cx = x_out.shape
        keep_out  = x_out[:, :n_keep_flat, :]
        merge_out = x_out[:, n_keep_flat:, :]

        out = torch.zeros(Bx, N, Cx, device=x_out.device, dtype=x_out.dtype)
        k_idx = keep_flat.unsqueeze(-1).expand(Bx, n_keep_flat, Cx)
        out.scatter_(1, k_idx, keep_out)

        m_exp = merge_out.unsqueeze(2).expand(Bx, n_merge, gs, Cx)
        m_idx = merge_flat.unsqueeze(-1).expand(Bx, n_merge * gs, Cx)
        out.scatter_(1, m_idx, m_exp.reshape(Bx, n_merge * gs, Cx))

        return out

    return merge, unmerge


class ToMeSAMAttention(Attention):

    def forward(self, x: torch.Tensor, ratio: float) -> torch.Tensor:
        B, H, W, _ = x.shape
        C = _ // self.num_heads
        N = H * W

        x = x.reshape(B, N, -1)

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


class ToMeSAMBlock(Block):
    """Full attention, then Hilbert-group merge → MLP → unmerge."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H_sp, W_sp, C = x.shape
        info       = self._tome_info
        ratio      = info["ratio"].pop(0)
        group_size = info.get("group_size", 4)

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

        x_merge, x_unmerge = tile_stride_matching(
            x_norm, H_sp, W_sp, ratio, group_size=group_size
        )
        x_norm_merged, _ = x_merge(x_norm)
        x_seq = x_seq + x_unmerge(self.mlp(x_norm_merged))

        return x_seq.reshape(B, H_sp, W_sp, C)


def apply_patch(
    encoder:    ImageEncoderViT,
    algo:       str   = "tome",
    ratio:      float = 0.9,
    margin:     float = 0.5,
    group_size: int   = 4,
    **_,
) -> ImageEncoderViT:
    """Patch SAM-1 image encoder with Hilbert-group MLP merging.
    `**_` swallows extras forwarded by the registry (e.g. `mlp_merge`)."""
    assert algo in ("tome", "pitome"), f"algo must be 'tome' or 'pitome', got {algo!r}"
    assert 0 < ratio <= 1.0,          "ratio must be in (0, 1]"

    tome_info = {
        "ratio":      ratio,
        "group_size": group_size,
    }
    encoder.tome_info = tome_info

    _orig_forward = encoder.__class__.forward

    def _patched_forward(self, x: torch.Tensor):
        n = len(self.blocks)
        r = self.tome_info["ratio"]
        self.tome_info["ratio"] = [r] * n
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
        f"[GradToMe-Hilbert] patched  algo={algo}  ratio={ratio}"
        f"  group_size={group_size}"
        f"  blocks={n_blocks} (global={n_global} local={n_blocks - n_global})"
        f"  strategy=full-attn / hilbert-group-mlp-merge"
    )
    return encoder
