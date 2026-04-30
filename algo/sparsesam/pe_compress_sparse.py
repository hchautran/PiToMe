"""SparseSAM stage compression with block-sparse FA2+RoPE attention.

Same Z-group merge as `pe_compress.py`, but every attention call routes
through the cute kernel with a banded-diagonal + keep-bar block-sparse mask.
Token permutation is applied ONCE per stage. Mirrors SAM-side pattern:
subclass + `__class__` swap.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch.nn as nn

from .pe_compress import _compress_sparsesam_core
from .._pe_stage_sparse import (
    apply_stage_compress_sparse,
    _ensure_attn_classes, _ensure_block_classes,
)
from .. import _pe_stage_sparse as _pss


def _make_SparsesamPECompressSparseBlock():
    _ensure_block_classes()

    class SparsesamPECompressSparseBlock(_pss.StageCompressSparsePEBlock):
        """Z-group merge + permute output once for downstream cute sparse-attn."""

        def compress(self, x, active_idx, info):
            return _compress_sparsesam_core(x, active_idx, info)

    return SparsesamPECompressSparseBlock


SparsesamPECompressSparseBlock: type = None  # type: ignore[assignment]


def _ensure_classes():
    global SparsesamPECompressSparseBlock
    _ensure_attn_classes()
    _ensure_block_classes()
    if SparsesamPECompressSparseBlock is None:
        SparsesamPECompressSparseBlock = _make_SparsesamPECompressSparseBlock()


def apply_pe_sparsesam_sparse_patch(model: nn.Module,
                                    ratio: float = 0.5,
                                    num_stages: int = 4,
                                    group_size: int = 4,
                                    sparse_ratio: Optional[float] = None,
                                    compress_at_blocks: Optional[list] = None,
                                    verbose: bool = True) -> int:
    """Z-group merge + cute block-sparse attention.

    `sparse_ratio` (defaults to `ratio`) controls keep-bar width inside the
    cute mask. Permutation is applied once at each stage's first block."""
    assert 0 < ratio <= 1.0
    assert num_stages >= 1
    assert group_size >= 1
    _ensure_classes()

    info = {
        "ratio": ratio,
        "group_size": group_size,
        "sparse_ratio": float(sparse_ratio) if sparse_ratio is not None else float(ratio),
    }
    return apply_stage_compress_sparse(
        model,
        compress_block_class=SparsesamPECompressSparseBlock,
        attn_class=_pss.SparseRopePEAttention,
        info=info,
        num_stages=num_stages,
        compress_at_blocks=compress_at_blocks,
        verbose_tag="pe-sparsesam-sparse" if verbose else "",
    )


def get_classes() -> Tuple[type, type]:
    _ensure_classes()
    return SparsesamPECompressSparseBlock, _pss.SparseRopePEAttention


def remove_pe_sparsesam_sparse_patch(model: nn.Module) -> int:
    from ..registry import remove_all_pe
    return remove_all_pe(model)


__all__ = [
    "apply_pe_sparsesam_sparse_patch",
    "remove_pe_sparsesam_sparse_patch",
    "get_classes",
]
