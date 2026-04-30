"""Block-sparse FA2+RoPE stage compression (sibling of `_pe_stage.py`).

Adds banded-diagonal + keep-bar attention mask + uniform-stride perm.
Pre-compress blocks fall through to stock SDPA. Each compress block
permutes its output once; downstream blocks reuse that layout.

Provides:
  * `SparseRopePEAttention(SelfAttention)` — sparse cute-kernel attention.
  * `StageCompressSparsePEBlock(ResidualAttentionBlock)` — base for stage-end
    blocks. Subclasses override `compress(x, active_idx, info)
    -> (x, new_active_idx)`.
  * `apply_stage_compress_sparse(...)` — installs subclasses on the matching
    modules. Mirrors SAM-side `apply_patch` shape.
"""

from __future__ import annotations
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _pe_stage as _ps   # access lazy-initialized cute globals via module
from ._pe_stage import SelfAttention, ResidualAttentionBlock


_PERM_CACHE: dict = {}             # (S, ratio, gs, n_blk, has_cls, device) -> (perm, inv)
_SPARSE_MASK_CACHE: dict = {}      # (B, H, T, ratio, m_blk, n_blk, band, scale, device)

_DIAG_BAND_WIDTH = 1
_KEEP_BAR_SCALE = 1.0


@torch.no_grad()
def _make_A_mask(B: int, H: int, T: int, ratio: float,
                 m_block: int, n_block: int,
                 band_width: int = _DIAG_BAND_WIDTH,
                 keep_bar_scale: float = _KEEP_BAR_SCALE,
                 device="cuda") -> torch.Tensor:
    num_m = math.ceil(T / m_block)
    num_n = math.ceil(T / n_block)
    t = torch.zeros(B, H, num_m, num_n, dtype=torch.int32, device=device)

    half = band_width // 2
    for k in range(-half, band_width - half):
        if k == 0:
            i = torch.arange(min(num_m, num_n), device=device)
            t[:, :, i, i] = 1
        else:
            i = torch.arange(max(0, -k), min(num_m, num_n - k), device=device)
            t[:, :, i, i + k] = 1

    n_keep_cols = int(ratio * num_n * keep_bar_scale)
    if n_keep_cols > 0:
        t[:, :, :, :max(n_keep_cols - band_width + 1, 0)] = 1
    return t


@torch.no_grad()
def _get_uniform_stride_perm(S: int, ratio: float, group_size: int,
                             n_block: int, has_cls: bool,
                             device) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_key = (S, ratio, group_size, n_block, has_cls, str(device))
    cached = _PERM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cls_off = 1 if has_cls else 0
    N = S - cls_off
    if N <= 0 or N % group_size != 0:
        perm = torch.arange(S, device=device)
        _PERM_CACHE[cache_key] = (perm, perm.clone())
        return _PERM_CACHE[cache_key]

    n_groups = N // group_size
    gs = group_size
    num_n_blocks = math.ceil(S / n_block)
    n_dense_blocks = int(ratio * num_n_blocks)
    n_keep_tokens = min(n_dense_blocks * n_block, n_groups * gs)
    n_keep = max(0, n_keep_tokens // gs)
    n_merge = n_groups - n_keep

    group_raster = torch.arange(N, device=device).view(n_groups, gs)

    if 0 < n_keep < n_groups:
        keep_idx = torch.round(
            torch.arange(n_keep, device=device, dtype=torch.float32)
            * (n_groups / n_keep)
        ).long().clamp_(0, n_groups - 1)
        m = torch.ones(n_groups, dtype=torch.bool, device=device)
        m[keep_idx] = False
        merge_idx = torch.nonzero(m, as_tuple=False).squeeze(1)
    elif n_keep == 0:
        keep_idx = torch.empty(0, dtype=torch.long, device=device)
        merge_idx = torch.arange(n_groups, device=device)
    else:
        keep_idx = torch.arange(n_groups, device=device)
        merge_idx = torch.empty(0, dtype=torch.long, device=device)

    keep_raster = group_raster[keep_idx].reshape(n_keep * gs)
    if n_merge > 0:
        merge_raster = group_raster[merge_idx].permute(1, 0).reshape(n_merge * gs)
        perm_sp = torch.cat([keep_raster, merge_raster], dim=0)
    else:
        perm_sp = keep_raster

    if cls_off:
        cls_perm = torch.arange(cls_off, device=device, dtype=perm_sp.dtype)
        perm = torch.cat([cls_perm, perm_sp + cls_off], dim=0)
    else:
        perm = perm_sp
    inv = torch.argsort(perm, dim=0)
    _PERM_CACHE[cache_key] = (perm, inv)
    return _PERM_CACHE[cache_key]


def flash_rope_sparse_attn(self_attn: nn.Module, x: torch.Tensor,
                           cos: torch.Tensor, sin: torch.Tensor,
                           block_mask: torch.Tensor,
                           perm: Optional[torch.Tensor] = None,
                           inv_perm: Optional[torch.Tensor] = None,
                           assume_permuted: bool = False,
                           ) -> Optional[torch.Tensor]:
    """Run fused FA2+RoPE cute kernel with a block-sparse mask. cos/sin are
    expected to be already permuted to match the kernel layout.

    `assume_permuted=True`: caller has already permuted `x` upstream (stage-
    level optimization); skip the input/output `index_select`. Otherwise
    permute `x` with `perm` on entry and `inv_perm` on exit.

    Returns None if the cute kernel can't be built — caller should fall back."""
    if x.dtype not in (torch.float16, torch.bfloat16):
        x = x.to(torch.float16)

    head_dim = self_attn.head_dim
    H        = self_attn.num_heads
    E        = self_attn.embed_dim

    kernel, m_blk, n_blk = _ps._get_kernel(x.dtype, head_dim)
    if kernel is None or block_mask is None:
        # Kernel unavailable or mask wasn't built (caller missed `_ensure_block_mask`
        # or passed a mismatched dtype): fall back to stock SDPA.
        return None

    B, S, _ = x.shape

    if assume_permuted or perm is None:
        x_in = x
    else:
        x_in = x.index_select(1, perm)

    proj = F.linear(x_in, self_attn.in_proj_weight, self_attn.in_proj_bias)
    proj = (proj.unflatten(-1, (3, E))
                 .unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous())
    q, k, v = proj[0], proj[1], proj[2]

    q = q.view(B, S, H, head_dim)
    k = k.view(B, S, H, head_dim)
    v = v.view(B, S, H, head_dim)
    o = torch.empty_like(q)

    dtype_width = 16 if x.dtype in (torch.float16, torch.bfloat16) else 32
    def _cute_qkvo(t):
        return (_ps.from_dlpack(t, assumed_align=16)
                .mark_layout_dynamic(leading_dim=3)
                .mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(),
                                            divisibility=128 // dtype_width))
    q_c, k_c, v_c, o_c = _cute_qkvo(q), _cute_qkvo(k), _cute_qkvo(v), _cute_qkvo(o)
    cos_c = _ps.from_dlpack(cos, assumed_align=16)
    sin_c = _ps.from_dlpack(sin, assumed_align=16)
    mask_c = _ps.from_dlpack(block_mask, assumed_align=4)

    cu_stream = _ps.cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    scale = float(self_attn.scale)

    compiled = _ps._get_compiled(
        kernel, q_c, k_c, v_c, o_c, cos_c, sin_c, mask_c, scale, cu_stream,
        x.dtype, head_dim, B, S, H, m_blk, n_blk,
    )
    compiled(q_c, k_c, v_c, o_c, cos_c, sin_c, mask_c, scale, cu_stream)

    attn = o.view(B, S, H * head_dim)
    if not assume_permuted and inv_perm is not None:
        attn = attn.index_select(1, inv_perm)
    return F.linear(attn, self_attn.out_proj.weight, self_attn.out_proj.bias)


@torch.no_grad()
def _build_stage_cache(self_attn: nn.Module, S: int, dtype: torch.dtype,
                       active_idx: Optional[torch.Tensor],
                       sr: float, group_size: int, has_cls: bool,
                       device) -> dict:
    """Pre-build cos/sin (sliced by active_idx, permuted to kernel layout)
    plus the perm/inv_perm tensors. Block mask is built lazily per-batch."""
    kernel, _m_blk, n_blk = _ps._get_kernel(dtype, self_attn.head_dim)
    if kernel is None:
        return {}

    cos_full, sin_full = _ps._module_cached_cos_sin(self_attn, dtype)
    if active_idx is not None:
        cos = cos_full.index_select(0, active_idx)
        sin = sin_full.index_select(0, active_idx)
    else:
        cos = cos_full[:S]
        sin = sin_full[:S]

    perm, inv_perm = _get_uniform_stride_perm(
        S, sr, group_size, n_blk, has_cls, device,
    )
    cos = cos.index_select(0, perm).contiguous()
    sin = sin.index_select(0, perm).contiguous()

    return {
        "cos": cos, "sin": sin,
        "perm": perm, "inv_perm": inv_perm,
        "block_mask": None,
    }


def _ensure_block_mask(cache: dict, self_attn, x: torch.Tensor, sr: float,
                       dtype: Optional[torch.dtype] = None):
    """Build (or fetch) the cute block-sparse mask for this (B, S, ...).

    `dtype` selects which kernel tile sizes to key the mask on; pass the
    same dtype the caller used to look up the kernel and build the cache.
    Defaults to `x.dtype`, but under autocast `x.dtype` may be fp32 (LN
    upcast) while the kernel is fp16/bf16 — pass the weight dtype in
    those callers."""
    if cache.get("block_mask") is not None:
        return cache["block_mask"]
    kdtype = dtype if dtype is not None else x.dtype
    kernel, m_blk, n_blk = _ps._get_kernel(kdtype, self_attn.head_dim)
    if kernel is None:
        return None
    B, S, _ = x.shape
    H = self_attn.num_heads
    mask_key = (B, H, S, sr, m_blk, n_blk,
                _DIAG_BAND_WIDTH, _KEEP_BAR_SCALE, str(x.device))
    block_mask = _SPARSE_MASK_CACHE.get(mask_key)
    if block_mask is None:
        block_mask = _make_A_mask(
            B, H, S, sr, m_blk, n_blk,
            band_width=_DIAG_BAND_WIDTH, keep_bar_scale=_KEEP_BAR_SCALE,
            device=x.device,
        )
        _SPARSE_MASK_CACHE[mask_key] = block_mask
    cache["block_mask"] = block_mask
    return block_mask


_FALLBACK_WARNED: dict = {}     # one warning per (dtype, head_dim)


def _sdpa_with_active_rope(self_attn, x, attn_mask, active_idx):
    """SDPA fallback: slice rope.freq to active_idx, run stock SelfAttention.forward,
    restore rope.freq."""
    orig_freq = self_attn.rope.freq
    self_attn.rope.freq = orig_freq.index_select(1, active_idx)
    try:
        return SelfAttention.forward(self_attn, x, attn_mask=attn_mask)
    finally:
        self_attn.rope.freq = orig_freq


# ── Attention subclass ───────────────────────────────────────────────────

class SparseRopePEAttention(SelfAttention):
    """Sparse FA2+RoPE attention.

    Pre-compress (active_idx is None): falls through to stock SDPA.
    No permutation, no cute kernel.

    Post-compress: routes through the block-sparse cute kernel.
    `x` is expected to already be in permuted layout (the compress block
    did this once); `assume_permuted=True` skips per-call index_select.
    Falls back to SDPA if the cute kernel can't be built."""

    def forward(self, x, attn_mask=None):
        info       = self._tome_info
        active_idx = info.get("active_idx", None)

        if active_idx is None:
            return super().forward(x, attn_mask=attn_mask)

        kernel, _m_blk, _n_blk = _ps._get_kernel(x.dtype, self.head_dim)
        if kernel is None:
            key = (str(x.dtype), int(self.head_dim))
            if key not in _FALLBACK_WARNED:
                _FALLBACK_WARNED[key] = True
                print(f"[pe-stage-sparse] cute kernel unavailable for "
                      f"dtype={x.dtype} head_dim={self.head_dim} "
                      f"— falling back to stock SDPA. To force a working "
                      f"tile size, edit "
                      f"PiToMe/algo/_pe_stage.py::_BLOCK_CANDIDATES.")
            return _sdpa_with_active_rope(self, x, attn_mask, active_idx)

        sr         = info.get("sparse_ratio", info.get("ratio", 1.0))
        group_size = info.get("group_size", 4)
        has_cls    = info.get("use_cls_token", False)

        cache = info.get("_stage_cache")
        cache_key = (id(active_idx), x.shape[1], x.dtype, sr)
        if cache is None or cache.get("_key") != cache_key:
            cache = _build_stage_cache(
                self, x.shape[1], x.dtype, active_idx, sr,
                group_size, has_cls, x.device,
            )
            cache["_key"] = cache_key
            info["_stage_cache"] = cache

        _ensure_block_mask(cache, self, x, sr)

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
        out = _sdpa_with_active_rope(self, x, attn_mask, active_idx)
        if permuted and cache.get("perm") is not None:
            out = out.index_select(1, cache["perm"])
        return out


# ── Block base class ─────────────────────────────────────────────────────

class StageCompressSparsePEBlock(ResidualAttentionBlock):
    """Stage-end block for sparse-attn compression.

    Subclasses override `compress(x, active_idx, info)
    -> (x, new_active_idx)` with the merge rule.

    Forward flow:
      1. Run the original block (attention + MLP).
      2. Un-permute `x` if currently in permuted layout.
      3. Run `self.compress` on natural-order x.
      4. Build the next stage's cache (cos/sin sliced + permuted).
      5. Permute the compressed output once. `x_is_permuted` flips True.
         Downstream blocks see permuted x and run cute sparse with
         `assume_permuted=True` — no per-call permutation."""

    def compress(self, x, active_idx, info):
        raise NotImplementedError

    def forward(self, x, attn_mask=None):
        x = super().forward(x, attn_mask=attn_mask)
        info: dict = self._tome_info

        if info.get("x_is_permuted"):
            cache = info.get("_stage_cache")
            inv_perm = cache.get("inv_perm") if cache else None
            if inv_perm is not None:
                x = x.index_select(1, inv_perm)
            info["x_is_permuted"] = False

        if info.get("ratio", 1.0) >= 1.0:
            return x

        x, new_active = self.compress(x, info.get("active_idx", None), info)
        info["active_idx"] = new_active

        sr = info.get("sparse_ratio", info.get("ratio", 1.0))
        cache = _build_stage_cache(
            self.attn, x.shape[1], x.dtype, new_active, sr,
            info.get("group_size", 4),
            info.get("use_cls_token", False),
            x.device,
        )
        cache["_key"] = (id(new_active), x.shape[1], x.dtype, sr)
        info["_stage_cache"] = cache

        perm = cache.get("perm")
        if perm is not None:
            x = x.index_select(1, perm)
            info["x_is_permuted"] = True

        return x


# ── Per-forward state ────────────────────────────────────────────────────

def _reset_state_hook(info):
    def _hook(_module, _inputs):
        info["active_idx"] = None
        info["_stage_cache"] = None
        info["x_is_permuted"] = False
    return _hook


def _transformer_unpermute_post_hook(info):
    """If the last stage left x in permuted layout, restore natural order
    before the Transformer returns."""
    def _hook(_module, _inputs, output):
        if not info.get("x_is_permuted"):
            return output
        cache = info.get("_stage_cache")
        inv_perm = cache.get("inv_perm") if cache else None
        if inv_perm is not None:
            output = output.index_select(1, inv_perm)
        info["x_is_permuted"] = False
        return output
    return _hook


# ── apply_stage_compress_sparse ──────────────────────────────────────────

def apply_stage_compress_sparse(model: nn.Module,
                                compress_block_class: type,
                                attn_class: type,
                                info: dict,
                                num_stages: int,
                                compress_at_blocks: Optional[list] = None,
                                verbose_tag: str = "pe-stage-sparse") -> int:
    """Install `compress_block_class` at stage-end indices and `attn_class`
    on every SelfAttention with rope. State on `model._tome_info`.

    `info` should be pre-populated with at least: ratio, num_stages,
    group_size, plus algo-specific fields. Optional `sparse_ratio`
    (defaults to `ratio`) sets the keep-bar width inside the cute mask.

    Returns the number of compression points wired up.
    """
    transformer = _ps._find_vision_transformer(model)
    if transformer is None:
        raise RuntimeError("Could not locate the PE vision Transformer in `model`.")

    _ps._ensure_cute_deps()
    if _ps.FlashAttentionForwardAmpereRoPE is None:
        raise RuntimeError(
            f"[{verbose_tag}] cute kernel not importable: "
            f"{_ps._KERNEL_IMPORT_ERROR!r} — block-sparse patch requires "
            f"the cute kernel."
        )

    info.setdefault("use_cls_token", _ps._vit_uses_cls_token(model))
    info.setdefault("sparse_ratio", float(info.get("ratio", 1.0)))
    info["active_idx"] = None
    info["_stage_cache"] = None
    info["x_is_permuted"] = False
    model._tome_info = info

    if not hasattr(transformer, "_pe_compress_pre_hook"):
        transformer._pe_compress_pre_hook = transformer.register_forward_pre_hook(
            _reset_state_hook(info)
        )
    if not hasattr(transformer, "_pe_compress_post_hook"):
        transformer._pe_compress_post_hook = transformer.register_forward_hook(
            _transformer_unpermute_post_hook(info)
        )

    n_attn = 0
    for mod in model.modules():
        if isinstance(mod, SelfAttention) and mod.rope is not None:
            if not isinstance(mod, attn_class):
                mod.__class__ = attn_class
            mod._tome_info = info
            n_attn += 1

    n_blocks = len(transformer.resblocks)
    if compress_at_blocks is not None:
        stage_ends = sorted({int(i) for i in compress_at_blocks
                             if 0 <= int(i) < n_blocks})
    else:
        stage_ends = _ps._stage_end_indices(n_blocks, num_stages)

    for idx in stage_ends:
        blk = transformer.resblocks[idx]
        if not isinstance(blk, compress_block_class):
            blk.__class__ = compress_block_class
        blk._tome_info = info

    if verbose_tag:
        mode = "explicit" if compress_at_blocks is not None else f"num_stages={num_stages}"
        print(f"[{verbose_tag}] L={n_blocks}  {mode}  "
              f"compress_after_blocks={stage_ends}  "
              f"ratio={info.get('ratio')}  "
              f"sparse_ratio={info.get('sparse_ratio')}  "
              f"use_cls_token={info['use_cls_token']}  patched_attn={n_attn}  "
              f"(pre-compress: SDPA, post-compress: cute sparse)")
    return len(stage_ends)


def remove_stage_compress_sparse(model: nn.Module) -> int:
    from .registry import remove_all_pe
    return remove_all_pe(model)
