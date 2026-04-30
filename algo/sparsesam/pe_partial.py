"""SparseSAM partial: cute block-sparse FA2+RoPE attention with
uniform-stride permutation + optional ToMe-style merge/MLP/unmerge.
Token count preserved. """

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .. import _pe_stage as _ps
from .._pe_stage import (
    SelfAttention, ResidualAttentionBlock,
    _ensure_cute_deps, _get_kernel, _module_cached_cos_sin,
    _find_vision_transformer, _vit_uses_cls_token,
)
from .._pe_stage_sparse import (
    _make_A_mask, _get_uniform_stride_perm, flash_rope_sparse_attn,
    _ensure_block_mask,
)


@torch.no_grad()
def _build_partial_cache(self_attn: nn.Module, S: int, dtype: torch.dtype,
                         sparse_ratio: float, group_size: int, has_cls: bool,
                         device) -> dict:
    """Pre-build cos/sin (permuted) + perm/inv_perm + merge layout split."""
    kernel, _m_blk, n_blk = _get_kernel(dtype, self_attn.head_dim)
    if kernel is None:
        return {}

    cos_full, sin_full = _module_cached_cos_sin(self_attn, dtype)
    cos = cos_full[:S]
    sin = sin_full[:S]

    perm, inv_perm = _get_uniform_stride_perm(
        S, sparse_ratio, group_size, n_blk, has_cls, device,
    )
    cos = cos.index_select(0, perm).contiguous()
    sin = sin.index_select(0, perm).contiguous()

    cls_off = 1 if has_cls else 0
    N = S - cls_off
    if N > 0 and group_size > 0 and N % group_size == 0:
        n_groups       = N // group_size
        num_n_blocks   = math.ceil(S / n_blk)
        n_dense_blocks = int(sparse_ratio * num_n_blocks)
        n_keep_tokens  = min(n_dense_blocks * n_blk, n_groups * group_size)
        n_keep         = max(0, n_keep_tokens // group_size)
        n_merge        = n_groups - n_keep
        cls_part_size  = cls_off + n_keep * group_size
    else:
        n_merge       = 0
        cls_part_size = S

    return {
        "cos": cos, "sin": sin,
        "perm": perm, "inv_perm": inv_perm,
        "block_mask": None,
        "cls_part_size": cls_part_size,
        "n_merge":       n_merge,
        "gs":            group_size,
    }


def _kernel_dtype(self_attn) -> torch.dtype:
    """Pick the dtype for kernel lookup + cos/sin tables.

    `x.dtype` may be fp32 under autocast (PyTorch's LayerNorm policy upcasts
    fp16 -> fp32 mid-stack), so using it here would miss the kernel cache.
    Use the weight dtype instead — the cute kernel only supports fp16/bf16
    and `flash_rope_sparse_attn` casts x internally."""
    w_dtype = self_attn.in_proj_weight.dtype
    if w_dtype in (torch.float16, torch.bfloat16):
        return w_dtype
    return torch.float16


# ── Subclasses ───────────────────────────────────────────────────────────

class SparsesamPEPartialAttention(SelfAttention):
    """Block-sparse cute kernel attention. Expects permuted layout
    (the block forward did this once at encoder entry); falls back to
    stock SDPA if the kernel can't be built."""

    def forward(self, x, attn_mask=None):
        info = self._tome_info
        sr   = info.get("sparse_ratio", info.get("ratio", 1.0))

        kdtype = _kernel_dtype(self)
        kernel, _m_blk, _n_blk = _get_kernel(kdtype, self.head_dim)
        if kernel is None:
            return super().forward(x, attn_mask=attn_mask)

        cache = info.get("_perm_cache")
        cache_key = (x.shape[1], kdtype, sr)
        if cache is None or cache.get("_key") != cache_key:
            cache = _build_partial_cache(
                self, x.shape[1], kdtype, sr,
                info.get("group_size", 4),
                info.get("use_cls_token", False),
                x.device,
            )
            cache["_key"] = cache_key
            info["_perm_cache"] = cache

        _ensure_block_mask(cache, self, x, sr, dtype=kdtype)

        permuted = bool(info.get("x_is_permuted"))
        out = flash_rope_sparse_attn(
            self, x,
            cos=cache.get("cos"), sin=cache.get("sin"),
            block_mask=cache.get("block_mask"),
            perm=cache.get("perm"), inv_perm=cache.get("inv_perm"),
            assume_permuted=permuted,
        )
        if out is not None:
            return out

        if permuted and cache.get("inv_perm") is not None:
            x = x.index_select(1, cache["inv_perm"])
        out = super().forward(x, attn_mask=attn_mask)
        if permuted and cache.get("perm") is not None:
            out = out.index_select(1, cache["perm"])
        return out


class SparsesamPEPartialBlock(ResidualAttentionBlock):
    """Block forward: permute once on first call, cute sparse-attn,
    merge/MLP/unmerge."""

    def forward(self, x, attn_mask=None):
        info: dict = self._tome_info
        ratio = info.get("ratio", 1.0)
        sr    = info.get("sparse_ratio", ratio)

        # First block in the forward: permute x once.
        if not info.get("x_is_permuted"):
            kdtype = _kernel_dtype(self.attn)
            kernel, _, _ = _get_kernel(kdtype, self.attn.head_dim)
            if kernel is not None:
                cache_key = (x.shape[1], kdtype, sr)
                cache = info.get("_perm_cache")
                if cache is None or cache.get("_key") != cache_key:
                    cache = _build_partial_cache(
                        self.attn, x.shape[1], kdtype, sr,
                        info.get("group_size", 4),
                        info.get("use_cls_token", False),
                        x.device,
                    )
                    cache["_key"] = cache_key
                    info["_perm_cache"] = cache
                perm = cache.get("perm")
                if perm is not None:
                    x = x.index_select(1, perm)
                    info["x_is_permuted"] = True

        x = x + self.drop_path1(
            self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask))
        )

        cache = info.get("_perm_cache")
        n_merge = (cache.get("n_merge", 0) if cache else 0)
        mlp_merge = info.get("mlp_merge", True)
        if (mlp_merge and info.get("x_is_permuted")
                and n_merge > 0 and ratio < 1.0):
            B, S, C = x.shape
            cls_part_size = cache["cls_part_size"]
            gs            = cache["gs"]

            keep_part     = x[:, :cls_part_size, :]
            merge_section = x[:, cls_part_size:, :]
            merge_view    = merge_section.reshape(B, gs, n_merge, C)
            merge_repr    = merge_view[:, 0, :, :]

            reduced_x = torch.cat([keep_part, merge_repr], dim=1)
            reduced_x = reduced_x + self.drop_path2(
                self.ls_2(self.mlp(self.ln_2(reduced_x)))
            )

            keep_out          = reduced_x[:, :cls_part_size, :]
            merge_repr_out    = reduced_x[:, cls_part_size:, :]
            merge_section_out = (merge_repr_out
                                  .unsqueeze(1)
                                  .expand(B, gs, n_merge, C)
                                  .reshape(B, gs * n_merge, C))

            x = torch.cat([keep_out, merge_section_out], dim=1)
        else:
            x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))

        return x


# ── Per-forward state ────────────────────────────────────────────────────

def _reset_state_hook(info):
    def _hook(_module, _inputs):
        info["x_is_permuted"] = False
    return _hook


def _transformer_unpermute_post_hook(info):
    """Restore natural token order on the encoder output."""
    def _hook(_module, _inputs, output):
        if not info.get("x_is_permuted"):
            return output
        cache = info.get("_perm_cache")
        inv_perm = cache.get("inv_perm") if cache else None
        if inv_perm is not None:
            output = output.index_select(1, inv_perm)
        info["x_is_permuted"] = False
        return output
    return _hook


def apply_pe_sparsesam_partial_patch(model: nn.Module,
                                     ratio: float = 0.7,
                                     group_size: int = 4,
                                     sparse_ratio: Optional[float] = None,
                                     start_block: int = 0,
                                     mlp_merge: bool = True,
                                     verbose: bool = True) -> int:
    """Sparse-attention + (optional) ToMe-style merge/unmerge MLP."""
    transformer = _find_vision_transformer(model)
    if transformer is None:
        raise RuntimeError("Could not locate the PE vision Transformer in `model`.")

    _ensure_cute_deps()

    n_blocks = len(transformer.resblocks)
    sb = max(0, min(int(start_block), n_blocks))

    use_cls_token = _vit_uses_cls_token(model)

    info = {
        "ratio":         float(ratio),
        "sparse_ratio":  float(sparse_ratio if sparse_ratio is not None else ratio),
        "group_size":    int(group_size),
        "use_cls_token": use_cls_token,
        "mlp_merge":     bool(mlp_merge),
    }
    model._tome_info = info

    if not hasattr(transformer, "_pe_partial_pre_hook"):
        transformer._pe_partial_pre_hook = transformer.register_forward_pre_hook(
            _reset_state_hook(info)
        )
    if not hasattr(transformer, "_pe_partial_post_hook"):
        transformer._pe_partial_post_hook = transformer.register_forward_hook(
            _transformer_unpermute_post_hook(info)
        )

    n_attn = 0
    for idx in range(sb, n_blocks):
        blk = transformer.resblocks[idx]

        attn = getattr(blk, "attn", None)
        if isinstance(attn, SelfAttention) and attn.rope is not None:
            if not isinstance(attn, SparsesamPEPartialAttention):
                attn.__class__ = SparsesamPEPartialAttention
            attn._tome_info = info
            n_attn += 1

        if not isinstance(blk, SparsesamPEPartialBlock):
            blk.__class__ = SparsesamPEPartialBlock
        blk._tome_info = info

    if verbose:
        print(f"[pe-sparsesam-partial] L={n_blocks}  start_block={sb}  "
              f"patched_blocks={n_blocks - sb}  patched_attn={n_attn}  "
              f"ratio={info['ratio']}  sparse_ratio={info['sparse_ratio']}  "
              f"use_cls_token={use_cls_token}  mlp_merge={info['mlp_merge']}  "
              f"(blocks 0..{sb - 1}: stock SDPA + full MLP; "
              f"blocks {sb}..{n_blocks - 1}: sparse-attn + "
              + ("ToMe-style merge/MLP/unmerge"
                 if info['mlp_merge']
                 else "full MLP (permuted)") + ")")
    return n_blocks - sb


def get_classes() -> Tuple[type, type]:
    return SparsesamPEPartialBlock, SparsesamPEPartialAttention


def remove_pe_sparsesam_partial_patch(model: nn.Module) -> int:
    from ..registry import remove_all_pe
    return remove_all_pe(model)


__all__ = [
    "apply_pe_sparsesam_partial_patch",
    "remove_pe_sparsesam_partial_patch",
    "get_classes",
]
