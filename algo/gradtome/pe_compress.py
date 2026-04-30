"""GradToMe stage-boundary token compression for PE Core models.

Gradient-driven bipartite matching on the spatial grid (CLS passes through).
Falls back to plain bipartite when the current sequence isn't a perfect square.
Mirrors the SAM-side patch shape: subclass + `__class__` swap.
"""

from __future__ import annotations
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from .merge import grad_bipartite_soft_matching, do_nothing
from .._pe_stage import (
    apply_stage_compress, FlashRopePEAttention, StageCompressPEBlock,
)


def _make_grad_merge(metric: torch.Tensor, r_grid: int, sx: int, sy: int,
                     ratio: float, cls_off: int) -> Optional[Callable]:
    N = r_grid * r_grid
    metric_sp = metric[:, cls_off:, :].contiguous()
    r = math.floor(N - N * ratio)
    sp_merge, _ = grad_bipartite_soft_matching(
        metric=metric_sp, H=r_grid, W=r_grid, sx=sx, sy=sy, r=r,
    )
    if sp_merge is do_nothing:
        return None

    def merge_fn(t, mode="mean"):
        if cls_off:
            cls_tok = t[:, :cls_off, :]
            sp = t[:, cls_off:, :].contiguous()
            sp_m, abs_sp = sp_merge(sp, mode=mode)
            abs_full = torch.cat([
                torch.zeros(abs_sp.shape[0], cls_off,
                            dtype=abs_sp.dtype, device=abs_sp.device),
                abs_sp + cls_off,
            ], dim=1)
            return torch.cat([cls_tok, sp_m], dim=1), abs_full
        return sp_merge(t, mode=mode)

    return merge_fn


class GradTomePECompressBlock(StageCompressPEBlock):
    """Spatial-gradient-aware bipartite; bipartite fallback off-square."""

    def compress(self, x, active_idx, info):
        ratio: float = info["ratio"]
        sx: int = info.get("grad_sx", 2)
        sy: int = info.get("grad_sy", 2)
        has_cls: bool = info.get("use_cls_token", False)

        S = x.shape[1]
        cls_off = 1 if has_cls else 0
        N = S - cls_off
        r_grid = int(math.isqrt(N))
        square = (r_grid * r_grid == N)

        merge_fn = None
        if square:
            metric = x.mean(0, keepdim=True)
            merge_fn = _make_grad_merge(metric, r_grid, sx, sy, ratio, cls_off)

        if merge_fn is None:
            from ..tome.merge import bipartite_soft_matching as _bsm
            metric = x.mean(0, keepdim=True)
            _bm, _ = _bsm(metric=metric, ratio=ratio, class_token=has_cls)
            if _bm is do_nothing:
                return x, active_idx
            x_merged, idx_in_x = _bm(x, mode="mean")
        else:
            x_merged, idx_in_x = merge_fn(x, mode="mean")

        new_active = (idx_in_x[0] if active_idx is None
                      else active_idx.index_select(0, idx_in_x[0]))
        return x_merged, new_active


def apply_pe_gradtome_patch(model: nn.Module,
                            ratio: float = 0.7,
                            num_stages: int = 4,
                            grad_sx: int = 2,
                            grad_sy: int = 2,
                            use_flash_rope: bool = False,
                            compress_at_blocks: Optional[list] = None,
                            verbose: bool = True,
                            **_) -> int:
    """Gradient-driven bipartite at stage boundaries.
    `**_` swallows `group_size` from `_kw_compress` (unused here)."""
    assert 0 < ratio <= 1.0
    assert num_stages >= 1

    info = {
        "ratio": ratio,
        "grad_sx": grad_sx,
        "grad_sy": grad_sy,
        "use_flash_rope": bool(use_flash_rope),
    }
    return apply_stage_compress(
        model,
        compress_block_class=GradTomePECompressBlock,
        attn_class=FlashRopePEAttention,
        info=info,
        num_stages=num_stages,
        use_flash_rope=use_flash_rope,
        compress_at_blocks=compress_at_blocks,
        verbose_tag="pe-gradtome-stage" if verbose else "",
    )


def get_classes() -> Tuple[type, type]:
    return GradTomePECompressBlock, FlashRopePEAttention


def remove_pe_gradtome_patch(model: nn.Module) -> int:
    from ..registry import remove_all_pe
    return remove_all_pe(model)
