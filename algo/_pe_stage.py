"""Shared PE patching primitives.

Provides:
  * Cute-kernel helpers (`flash_rope_attn`, `_get_kernel`) for fused
    FA2 + RoPE attention.
  * `FlashRopePEAttention(SelfAttention)` — drop-in attention subclass that
    slices `rope.freq` to `info["active_idx"]` (so RoPE indices follow
    surviving tokens after compression) and optionally routes through the
    cute kernel.
  * `FlashRopeOnlyPEAttention(SelfAttention)` — pure cute-kernel swap, no
    compression awareness (used by the `flash_rope` algo).
  * `StageCompressPEBlock(ResidualAttentionBlock)` — base for
    stage-end blocks. Subclasses override `compress(x, active_idx, info)
    -> (x, new_active_idx)` to define the merge rule.
  * `apply_stage_compress(...)` — installs subclasses on the matching
    modules. Mirrors SAM-side `apply_patch` shape.

Patches on PE follow the SAM convention: subclass + reassign
`module.__class__`. `remove_all_pe` (in `registry.py`) walks the model and
reverts every registered subclass back to `SelfAttention` / `ResidualAttentionBlock`.
"""

from __future__ import annotations
import os
import sys
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Cute kernel: lazy-imported so this module is cheap to import on CUDA-less hosts.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CUTE_DIR = os.path.join(_REPO_ROOT, "cute")
if _CUTE_DIR not in sys.path:
    sys.path.insert(0, _CUTE_DIR)

_KERNEL_IMPORT_ERROR: Optional[Exception] = None
FlashAttentionForwardAmpereRoPE = None
from_dlpack = None
cuda_driver = None
cutlass = None


def _ensure_cute_deps():
    global FlashAttentionForwardAmpereRoPE, from_dlpack, cuda_driver, cutlass
    global _KERNEL_IMPORT_ERROR
    if FlashAttentionForwardAmpereRoPE is not None:
        return
    try:
        import cutlass as _cutlass
        from cutlass.cute.runtime import from_dlpack as _from_dlpack
        import cuda.bindings.driver as _cuda_driver
        from flash_attn_rope_fused import FlashAttentionForwardAmpereRoPE as _Kernel
    except Exception as e:
        _KERNEL_IMPORT_ERROR = e
        return
    cutlass = _cutlass
    from_dlpack = _from_dlpack
    cuda_driver = _cuda_driver
    FlashAttentionForwardAmpereRoPE = _Kernel


_KERNEL_CACHE: dict = {}      # (dtype, head_dim) -> (kernel, m_blk, n_blk)
_COMPILED_CACHE: dict = {}    # (dtype, head_dim, B, S, H, m_blk, n_blk) -> compiled
_MASK_CACHE: dict = {}        # (B, H, M_, N_, device) -> ones mask

# 64x64 first: PE has small grids (<=32x32 patches), so small tiles give the
# block-sparse mask more granularity.
_BLOCK_CANDIDATES: Tuple[Tuple[int, int, int], ...] = (
    ( 64,  64, 128),
    ( 64, 128, 128),
    (128,  64, 128),
    (128, 128, 128),
)


def _torch_dtype_to_cutlass(dtype: torch.dtype):
    return {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }.get(dtype, None)


def _get_kernel(dtype: torch.dtype, head_dim: int):
    key = (dtype, head_dim)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    cl_dtype = _torch_dtype_to_cutlass(dtype)
    if cl_dtype is None:
        print(f"[pe-stage] _get_kernel: no cutlass dtype for {dtype} — "
              f"kernel unavailable.")
        _KERNEL_CACHE[key] = (None, None, None)
        return _KERNEL_CACHE[key]

    for m_blk, n_blk, n_thr in _BLOCK_CANDIDATES:
        ok = FlashAttentionForwardAmpereRoPE.can_implement(
            cl_dtype, head_dim, m_blk, n_blk, n_thr,
        )
        if ok:
            kernel = FlashAttentionForwardAmpereRoPE(
                head_dim=head_dim, m_block_size=m_blk,
                n_block_size=n_blk, num_threads=n_thr,
            )
            _KERNEL_CACHE[key] = (kernel, m_blk, n_blk)
            return _KERNEL_CACHE[key]

    # No candidate worked — log SMEM/threading constraints for diagnosis.
    try:
        from cutlass.cute._mlir_helpers import _utils as _cute_utils  # type: ignore
        smem_cap = _cute_utils.get_smem_capacity_in_bytes("sm_80")
    except Exception:
        try:
            import cutlass.cute as _cute
            smem_cap = _cute.utils.get_smem_capacity_in_bytes("sm_80")
        except Exception:
            smem_cap = "?"
    head_dim_padded = (head_dim + 31) // 32 * 32
    print(f"[pe-stage] _get_kernel: NO candidate satisfies can_implement "
          f"for (dtype={dtype}, head_dim={head_dim}, head_dim_padded="
          f"{head_dim_padded}). SM80 SMEM budget = {smem_cap} bytes.")
    for m_blk, n_blk, n_thr in _BLOCK_CANDIDATES:
        smem = (m_blk * head_dim_padded * 2          # Q
                + n_blk * head_dim_padded * 2 * 2    # K + V
                + m_blk * head_dim_padded * 2 * 2    # cos+sin for Q
                + n_blk * head_dim_padded * 2 * 2)   # cos+sin for K
        thread_ok = (m_blk * 2) % n_thr == 0
        print(f"  candidate m={m_blk} n={n_blk} threads={n_thr}: "
              f"smem={smem}B  (m*2)%threads=={(m_blk*2)%n_thr} "
              f"(thread_ok={thread_ok})")
    _KERNEL_CACHE[key] = (None, None, None)
    return _KERNEL_CACHE[key]


def _get_compiled(kernel, q, k, v, o, cos, sin, mask, scale, stream,
                  dtype, head_dim, B, S, H, m_blk, n_blk):
    key = (dtype, head_dim, B, S, H, m_blk, n_blk)
    fn = _COMPILED_CACHE.get(key)
    if fn is None:
        import cutlass.cute as cute
        fn = cute.compile(kernel, q, k, v, o, cos, sin, mask, scale, stream)
        _COMPILED_CACHE[key] = fn
    return fn


@torch.no_grad()
def _build_cos_sin(rope, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    freq = rope.freq                                  # (1, S, D)
    if freq is None:
        raise RuntimeError(
            "Rope2D.freq is None — call rope.update_grid(...) first "
            "(usually done inside VisionTransformer.forward_features)."
        )
    cos = freq.cos().squeeze(0).contiguous().to(dtype)
    sin = freq.sin().squeeze(0).contiguous().to(dtype)
    return cos, sin


@torch.no_grad()
def _module_cached_cos_sin(self_attn: nn.Module,
                           dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-SelfAttention cache for full-grid cos/sin. Invalidates when
    `rope.freq.data_ptr()` or `dtype` changes."""
    freq = self_attn.rope.freq
    if freq is None:
        raise RuntimeError("Rope2D.freq is None — call rope.update_grid first.")
    key = (freq.data_ptr(), dtype, freq.shape[1])
    if getattr(self_attn, "_flash_cos_sin_key", None) == key:
        return self_attn._flash_cos, self_attn._flash_sin
    cos, sin = _build_cos_sin(self_attn.rope, dtype)
    self_attn._flash_cos = cos
    self_attn._flash_sin = sin
    self_attn._flash_cos_sin_key = key
    return cos, sin


def flash_rope_attn(self_attn: nn.Module, x: torch.Tensor,
                    cos: Optional[torch.Tensor] = None,
                    sin: Optional[torch.Tensor] = None,
                    block_mask: Optional[torch.Tensor] = None,
                    ) -> Optional[torch.Tensor]:
    """Run fused FA2 + RoPE cute kernel (dense). Returns None if the kernel
    can't be built for this (dtype, head_dim) — callers should fall back."""
    if x.dtype not in (torch.float16, torch.bfloat16):
        x = x.to(torch.float16)

    head_dim = self_attn.head_dim
    H        = self_attn.num_heads
    E        = self_attn.embed_dim

    kernel, m_blk, n_blk = _get_kernel(x.dtype, head_dim)
    if kernel is None:
        return None

    B, S, _ = x.shape

    proj = F.linear(x, self_attn.in_proj_weight, self_attn.in_proj_bias)
    proj = (proj.unflatten(-1, (3, E))
                 .unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous())
    q, k, v = proj[0], proj[1], proj[2]

    q = q.view(B, S, H, head_dim)
    k = k.view(B, S, H, head_dim)
    v = v.view(B, S, H, head_dim)
    o = torch.empty_like(q)

    if cos is None or sin is None:
        cos, sin = _module_cached_cos_sin(self_attn, x.dtype)
        cos = cos[:S].contiguous()
        sin = sin[:S].contiguous()

    if block_mask is None:
        M_ = (S + m_blk - 1) // m_blk
        N_ = (S + n_blk - 1) // n_blk
        mask_key = (B, H, M_, N_, str(x.device))
        block_mask = _MASK_CACHE.get(mask_key)
        if block_mask is None:
            block_mask = torch.ones((B, H, M_, N_), dtype=torch.int32, device=x.device)
            _MASK_CACHE[mask_key] = block_mask

    dtype_width = 16 if x.dtype in (torch.float16, torch.bfloat16) else 32
    def _cute_qkvo(t):
        return (from_dlpack(t, assumed_align=16)
                .mark_layout_dynamic(leading_dim=3)
                .mark_compact_shape_dynamic(mode=3, stride_order=t.dim_order(),
                                            divisibility=128 // dtype_width))
    q_c, k_c, v_c, o_c = _cute_qkvo(q), _cute_qkvo(k), _cute_qkvo(v), _cute_qkvo(o)
    cos_c = from_dlpack(cos, assumed_align=16)
    sin_c = from_dlpack(sin, assumed_align=16)
    mask_c = from_dlpack(block_mask, assumed_align=4)

    cu_stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    scale = float(self_attn.scale)

    compiled = _get_compiled(
        kernel, q_c, k_c, v_c, o_c, cos_c, sin_c, mask_c, scale, cu_stream,
        x.dtype, head_dim, B, S, H, m_blk, n_blk,
    )
    compiled(q_c, k_c, v_c, o_c, cos_c, sin_c, mask_c, scale, cu_stream)

    attn = o.view(B, S, H * head_dim)
    return F.linear(attn, self_attn.out_proj.weight, self_attn.out_proj.bias)


def _stage_end_indices(n_blocks: int, num_stages: int) -> List[int]:
    """Indices of blocks AFTER which compression happens. Length = num_stages-1."""
    if num_stages <= 1 or n_blocks < num_stages:
        return []
    chunk = n_blocks // num_stages
    return [chunk * (i + 1) - 1 for i in range(num_stages - 1)]


def _find_vision_transformer(model: nn.Module):
    from core.vision_encoder.pe import VisionTransformer, Transformer
    for m in model.modules():
        if isinstance(m, VisionTransformer):
            for sub in m.modules():
                if isinstance(sub, Transformer):
                    return sub
    return None


def _vit_uses_cls_token(model: nn.Module) -> bool:
    from core.vision_encoder.pe import VisionTransformer
    for m in model.modules():
        if isinstance(m, VisionTransformer):
            return bool(getattr(m, "use_cls_token", False))
    return False


# Lazy import: PE classes loaded the first time a base class is defined.
def _pe_classes():
    from core.vision_encoder.pe import SelfAttention, ResidualAttentionBlock
    return SelfAttention, ResidualAttentionBlock


# ── Attention base classes ───────────────────────────────────────────────
#
# Each algo's apply_patch reassigns `attn.__class__` to one of these (or a
# subclass). State is read from `self._tome_info`, which is set by the
# apply_patch.

def _make_FlashRopePEAttention():
    SelfAttention, _ = _pe_classes()

    class FlashRopePEAttention(SelfAttention):
        """SelfAttention that respects `info['active_idx']` (so RoPE follows
        surviving tokens after compression), and optionally routes through
        the fused FA2+RoPE cute kernel when `info['use_flash_rope']`."""

        def forward(self, x, attn_mask=None):
            info       = self._tome_info
            active_idx = info.get("active_idx", None)
            use_flash  = info.get("use_flash_rope", False)

            if use_flash:
                cache = info.get("_stage_cache")
                cache_key = (id(active_idx), x.shape[1], x.dtype)
                if cache is None or cache.get("_key") != cache_key:
                    cache = _build_stage_cache(self, x.shape[1], x.dtype, active_idx)
                    cache["_key"] = cache_key
                    info["_stage_cache"] = cache

                out = flash_rope_attn(
                    self, x,
                    cos=cache.get("cos"), sin=cache.get("sin"),
                    block_mask=cache.get("block_mask"),
                )
                if out is not None:
                    return out
                # Kernel couldn't be built — fall through to stock SDPA.

            if active_idx is None:
                return super().forward(x, attn_mask=attn_mask)
            # Stock SDPA with rope.freq sliced down to surviving tokens.
            orig_freq = self.rope.freq
            self.rope.freq = orig_freq.index_select(1, active_idx)
            try:
                return super().forward(x, attn_mask=attn_mask)
            finally:
                self.rope.freq = orig_freq

    return FlashRopePEAttention


def _make_FlashRopeOnlyPEAttention():
    SelfAttention, _ = _pe_classes()

    class FlashRopeOnlyPEAttention(SelfAttention):
        """Pure cute-kernel attention swap. No compression awareness."""

        def forward(self, x, attn_mask=None):
            del attn_mask
            out = flash_rope_attn(self, x)
            if out is None:
                return super().forward(x)
            return out

    return FlashRopeOnlyPEAttention


# Defer class creation: PE may not be importable at module load time.
FlashRopePEAttention: type = None      # type: ignore[assignment]
FlashRopeOnlyPEAttention: type = None  # type: ignore[assignment]


def _ensure_attn_classes():
    global FlashRopePEAttention, FlashRopeOnlyPEAttention
    if FlashRopePEAttention is None:
        FlashRopePEAttention = _make_FlashRopePEAttention()
    if FlashRopeOnlyPEAttention is None:
        FlashRopeOnlyPEAttention = _make_FlashRopeOnlyPEAttention()


# ── Block base class for stage-end compression ──────────────────────────

def _make_StageCompressPEBlock():
    _, ResidualAttentionBlock = _pe_classes()

    class StageCompressPEBlock(ResidualAttentionBlock):
        """Base for stage-end blocks. Runs the original block forward, then
        calls `self.compress(x, active_idx, info) -> (x, new_active_idx)`.

        Subclasses override `compress` with their merge rule."""

        def compress(self, x, active_idx, info):
            raise NotImplementedError

        def forward(self, x, attn_mask=None):
            x = super().forward(x, attn_mask=attn_mask)
            info: dict = self._tome_info
            if info.get("ratio", 1.0) >= 1.0:
                return x
            x, new_active = self.compress(x, info.get("active_idx", None), info)
            info["active_idx"] = new_active
            info["_stage_cache"] = None    # invalidate cos/sin cache
            return x

    return StageCompressPEBlock


StageCompressPEBlock: type = None  # type: ignore[assignment]


def _ensure_block_classes():
    global StageCompressPEBlock
    if StageCompressPEBlock is None:
        StageCompressPEBlock = _make_StageCompressPEBlock()


# ── Stage cache (cos/sin slice for surviving tokens) ─────────────────────

@torch.no_grad()
def _build_stage_cache(self_attn: nn.Module, S: int, dtype: torch.dtype,
                       active_idx: Optional[torch.Tensor]) -> dict:
    """Pre-build cos/sin (sliced down to the active subset) for the fused-
    kernel path so subsequent blocks in the same stage just reuse them."""
    kernel, _m_blk, _n_blk = _get_kernel(dtype, self_attn.head_dim)
    if kernel is None:
        return {}

    cos_full, sin_full = _module_cached_cos_sin(self_attn, dtype)
    if active_idx is not None:
        cos = cos_full.index_select(0, active_idx).contiguous()
        sin = sin_full.index_select(0, active_idx).contiguous()
    else:
        cos = cos_full[:S].contiguous()
        sin = sin_full[:S].contiguous()

    return {
        "cos": cos, "sin": sin,
        "block_mask": None,
    }


# ── Per-forward state reset ──────────────────────────────────────────────

def _reset_active_idx_hook(info):
    def _hook(_module, _inputs):
        info["active_idx"] = None
        info["_stage_cache"] = None
    return _hook


# ── apply_stage_compress: the convenience installer ──────────────────────

def apply_stage_compress(model: nn.Module,
                         compress_block_class: type,
                         attn_class: type,
                         info: dict,
                         num_stages: int,
                         use_flash_rope: bool,
                         compress_at_blocks: Optional[List[int]] = None,
                         verbose_tag: str = "pe-stage") -> int:
    """Install `compress_block_class` at stage-end indices and `attn_class`
    on every SelfAttention with rope. State lives on `model._tome_info`.

    `info` should be pre-populated with at least: ratio, num_stages,
    use_cls_token (auto-filled if absent), use_flash_rope, plus algo-specific
    fields. `apply_stage_compress` adds `active_idx` and `_stage_cache`.

    `compress_at_blocks` (optional) overrides `num_stages`: pass an explicit
    list of 0-indexed block indices after which compression should fire.

    Returns the number of compression points installed.
    """
    SelfAttention, ResidualAttentionBlock = _pe_classes()
    _ensure_attn_classes()
    _ensure_block_classes()

    transformer = _find_vision_transformer(model)
    if transformer is None:
        raise RuntimeError("Could not locate the PE vision Transformer in `model`.")

    if use_flash_rope:
        _ensure_cute_deps()
        if FlashAttentionForwardAmpereRoPE is None and verbose_tag:
            print(f"[{verbose_tag}] use_flash_rope=True but cute kernel not "
                  f"importable ({_KERNEL_IMPORT_ERROR!r}); will fall back at "
                  f"runtime to stock SDPA.")

    info.setdefault("use_cls_token", _vit_uses_cls_token(model))
    info["active_idx"] = None
    info["_stage_cache"] = None
    model._tome_info = info

    # Pre-hook on the Transformer resets per-forward state.
    if not hasattr(transformer, "_pe_compress_pre_hook"):
        transformer._pe_compress_pre_hook = transformer.register_forward_pre_hook(
            _reset_active_idx_hook(info)
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
        stage_ends = _stage_end_indices(n_blocks, num_stages)

    for idx in stage_ends:
        blk = transformer.resblocks[idx]
        if not isinstance(blk, compress_block_class):
            blk.__class__ = compress_block_class
        blk._tome_info = info

    if verbose_tag:
        mode = "explicit" if compress_at_blocks is not None else f"num_stages={num_stages}"
        print(f"[{verbose_tag}] L={n_blocks}  {mode}  "
              f"compress_after_blocks={stage_ends}  ratio={info.get('ratio')}  "
              f"use_cls_token={info['use_cls_token']}  patched_attn={n_attn}  "
              f"use_flash_rope={info.get('use_flash_rope', False)}")
    return len(stage_ends)


# ── Flash-rope-only patch (no compression) ───────────────────────────────

def apply_pe_flash_rope_patch(model: nn.Module, verbose: bool = True) -> int:
    """Replace every PE SelfAttention with `FlashRopeOnlyPEAttention`. No
    token compression. Falls back at runtime if the cute kernel can't be
    built for this (dtype, head_dim)."""
    SelfAttention, _ = _pe_classes()

    _ensure_cute_deps()
    if FlashAttentionForwardAmpereRoPE is None:
        msg = (f"[pe-flash-rope] cute kernel not importable: "
               f"{_KERNEL_IMPORT_ERROR!r} — patch not applied")
        if verbose:
            print(msg)
        raise RuntimeError(msg)

    _ensure_attn_classes()

    n = 0
    for mod in model.modules():
        if isinstance(mod, SelfAttention) and mod.rope is not None:
            if not isinstance(mod, FlashRopeOnlyPEAttention):
                mod.__class__ = FlashRopeOnlyPEAttention
            n += 1

    if verbose:
        print(f"[pe-flash-rope] patched {n} SelfAttention module(s)")
    return n


# Per-patch remove functions are no longer needed: `remove_all_pe` in
# `registry.py` walks the model and reverts every registered subclass.
# These wrappers exist for back-compat with code that imports them.

def remove_stage_compress(model: nn.Module) -> int:
    from .registry import remove_all_pe
    return remove_all_pe(model)


def remove_pe_flash_rope_patch(model: nn.Module) -> int:
    from .registry import remove_all_pe
    return remove_all_pe(model)
