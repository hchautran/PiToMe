"""SparseSAM stage-boundary token compression for PE Core models.

Top-K Z-groups kept verbatim; remaining groups averaged into 1 representative
each. Mirrors SAM-side patch shape: subclass + `__class__` swap.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .z_utils import get_z_order
from .._pe_stage import (
    apply_stage_compress, FlashRopePEAttention, StageCompressPEBlock,
    apply_pe_flash_rope_patch, remove_pe_flash_rope_patch,
    flash_rope_attn,
)


_GROUP_RASTER_CACHE: dict = {}


@torch.no_grad()
def _get_group_raster(N: int, group_size: int, device) -> torch.Tensor:
    key = (N, group_size, str(device))
    cached = _GROUP_RASTER_CACHE.get(key)
    if cached is not None:
        return cached
    n_groups = N // group_size
    r_grid = int(math.isqrt(N))
    if r_grid * r_grid == N:
        z_perm = get_z_order(r_grid, r_grid, device=device)
        gr = z_perm.view(n_groups, group_size)
    else:
        gr = torch.arange(N, device=device).view(n_groups, group_size)
    _GROUP_RASTER_CACHE[key] = gr
    return gr


def _pick_group_size(N: int, gs_max: int) -> int:
    """Largest divisor of N in [2, gs_max], or 0 if none.

    After stage 1, the active token count is rarely a multiple of the
    user-configured `group_size` (e.g. PE-Core-L14-336: 576 -> 447 with
    ratio=0.7, group_size=4). Adapting per stage keeps every boundary
    actually compressing instead of silently passing through."""
    for gs in range(min(gs_max, N), 1, -1):
        if N % gs == 0:
            return gs
    return 0


@torch.no_grad()
def _compress_sparsesam_core(x: torch.Tensor,
                             active_idx: Optional[torch.Tensor],
                             info: dict) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Z-group merge core. Reused by both dense and sparse-attn variants."""
    ratio: float = info["ratio"]
    group_size_max: int = info.get("group_size", 4)
    has_cls: bool = info.get("use_cls_token", False)

    B, S, C = x.shape
    cls_off = 1 if has_cls else 0
    N = S - cls_off

    gs = _pick_group_size(N, group_size_max)
    if gs < 2:
        return x, active_idx

    n_groups = N // gs

    target_out = int(round(ratio * N))
    n_keep = int(round((target_out - n_groups) / (gs - 1)))
    n_keep = max(0, min(n_groups, n_keep))
    if n_keep >= n_groups:
        return x, active_idx
    n_merge = n_groups - n_keep

    group_raster = _get_group_raster(N, gs, x.device)

    x_sp = x[:, cls_off:, :]

    x_grp_one = x_sp[0].index_select(0, group_raster.reshape(-1)).view(n_groups, gs, C)
    score = x_grp_one.std(dim=1).mean(dim=-1)

    rank = score.argsort(descending=True)
    keep_grps = rank[:n_keep].sort().values
    merge_grps = rank[n_keep:]

    keep_raster = group_raster[keep_grps].reshape(-1)
    keep_tokens = x_sp.index_select(1, keep_raster)

    merge_raster = group_raster[merge_grps].reshape(-1)
    merge_repr = (x_sp.index_select(1, merge_raster)
                       .view(B, n_merge, gs, C)
                       .mean(dim=2))

    parts_x = []
    if cls_off:
        parts_x.append(x[:, :cls_off, :])
    parts_x.append(keep_tokens)
    parts_x.append(merge_repr)
    x_new = torch.cat(parts_x, dim=1)

    merge_first_member = group_raster[merge_grps, 0]
    keep_full = keep_raster + cls_off
    merge_full = merge_first_member + cls_off

    if active_idx is None:
        cls_orig = torch.arange(cls_off, device=x.device, dtype=keep_full.dtype)
        kept_orig = keep_full
        merge_orig = merge_full
    else:
        cls_orig = active_idx[:cls_off]
        kept_orig = active_idx.index_select(0, keep_full)
        merge_orig = active_idx.index_select(0, merge_full)

    parts_idx = []
    if cls_off:
        parts_idx.append(cls_orig)
    parts_idx.append(kept_orig)
    parts_idx.append(merge_orig)
    new_active = torch.cat(parts_idx, dim=0)

    return x_new, new_active


class SparsesamPECompressBlock(StageCompressPEBlock):
    """Z-group SparseSAM merge."""

    def compress(self, x, active_idx, info):
        return _compress_sparsesam_core(x, active_idx, info)


def apply_pe_sparsesam_patch(model: nn.Module,
                             ratio: float = 0.5,
                             num_stages: int = 4,
                             group_size: int = 4,
                             use_flash_rope: bool = False,
                             compress_at_blocks: Optional[list] = None,
                             verbose: bool = True) -> int:
    """SparseSAM Z-group merge at stage boundaries."""
    assert 0 < ratio <= 1.0
    assert num_stages >= 1
    assert group_size >= 1

    info = {
        "ratio": ratio,
        "group_size": group_size,
        "use_flash_rope": bool(use_flash_rope),
    }
    return apply_stage_compress(
        model,
        compress_block_class=SparsesamPECompressBlock,
        attn_class=FlashRopePEAttention,
        info=info,
        num_stages=num_stages,
        use_flash_rope=use_flash_rope,
        compress_at_blocks=compress_at_blocks,
        verbose_tag="pe-sparsesam-stage" if verbose else "",
    )


def get_classes() -> Tuple[type, type]:
    return SparsesamPECompressBlock, FlashRopePEAttention


def remove_pe_sparsesam_patch(model: nn.Module) -> int:
    from ..registry import remove_all_pe
    return remove_all_pe(model)


# Back-compat aliases.
apply_pe_sparse_patch = apply_pe_sparsesam_patch
remove_pe_sparse_patch = remove_pe_sparsesam_patch

__all__ = [
    "apply_pe_sparsesam_patch", "remove_pe_sparsesam_patch",
    "apply_pe_sparse_patch", "remove_pe_sparse_patch",
    "apply_pe_flash_rope_patch", "remove_pe_flash_rope_patch",
    "flash_rope_attn",
    "_compress_sparsesam_core",
    "get_classes",
]
