"""ToMe stage-boundary token compression for PE Core models.

Reduces `x` at each stage boundary via bipartite soft matching.
Mirrors the SAM-side patch shape: subclass + `__class__` swap.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .merge import bipartite_soft_matching, do_nothing
from .._pe_stage import (
    apply_stage_compress, FlashRopePEAttention, StageCompressPEBlock,
)


class TomePECompressBlock(StageCompressPEBlock):
    """Bipartite soft matching: matched pairs averaged together."""

    def compress(self, x, active_idx, info):
        ratio: float = info["ratio"]
        has_cls: bool = info.get("use_cls_token", False)

        metric = x.mean(0, keepdim=True)
        merge_fn, _ = bipartite_soft_matching(
            metric=metric, ratio=ratio, class_token=has_cls,
        )
        if merge_fn is do_nothing:
            return x, active_idx

        x_merged, idx_in_x = merge_fn(x, mode="mean")
        new_active = (idx_in_x[0] if active_idx is None
                      else active_idx.index_select(0, idx_in_x[0]))
        return x_merged, new_active


def apply_pe_tome_patch(model: nn.Module,
                        ratio: float = 0.7,
                        num_stages: int = 4,
                        use_flash_rope: bool = False,
                        compress_at_blocks: Optional[list] = None,
                        verbose: bool = True,
                        **_) -> int:
    """Bipartite soft matching at stage boundaries. ratio=1.0 disables.
    `use_flash_rope` routes attention through fused FA2+RoPE cute kernel.
    `compress_at_blocks` overrides `num_stages` with explicit block indices.
    `**_` silently swallows extra kwargs from `_kw_compress` (e.g. `group_size`)
    that other stage-compress patches consume but tome doesn't."""
    assert 0 < ratio <= 1.0
    assert num_stages >= 1

    info = {
        "ratio": ratio,
        "use_flash_rope": bool(use_flash_rope),
    }
    return apply_stage_compress(
        model,
        compress_block_class=TomePECompressBlock,
        attn_class=FlashRopePEAttention,
        info=info,
        num_stages=num_stages,
        use_flash_rope=use_flash_rope,
        compress_at_blocks=compress_at_blocks,
        verbose_tag="pe-tome-stage" if verbose else "",
    )


def get_classes() -> Tuple[type, type]:
    """Return (block_class, attn_class) for registry registration."""
    return TomePECompressBlock, FlashRopePEAttention


def remove_pe_tome_patch(model: nn.Module) -> int:
    from ..registry import remove_all_pe
    return remove_all_pe(model)
