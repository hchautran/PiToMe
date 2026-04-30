"""Central PE + SAM algorithm registry.

Eval / profile scripts iterate this registry. Adding a new algo = write
the patch + one `register_pe(...)` / `register_sam(...)` call here.
See `docs/ADDING_ALGORITHMS.md`."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class PEAlgoSpec:
    name: str
    apply: Callable[..., Any]
    remove: Optional[Callable[..., Any]] = None  # optional; remove_all_pe walks subclasses
    kwargs_from_args: Optional[Callable[[Any, Optional[float]], dict]] = None
    accepts_ratio: bool = True
    description: str = ""
    category: str = "compress"     # "compress" | "partial" | "attention"


PE_REGISTRY: Dict[str, PEAlgoSpec] = {}


def register_pe(spec: PEAlgoSpec) -> PEAlgoSpec:
    if spec.name in PE_REGISTRY:
        raise ValueError(f"PE algorithm {spec.name!r} already registered")
    PE_REGISTRY[spec.name] = spec
    return spec


def algo_choices() -> List[str]:
    """All valid `--algorithm` strings, including the no-op baseline."""
    return ["none"] + sorted(PE_REGISTRY.keys())


def is_partial(name: str) -> bool:
    """True iff the named algorithm is in the *_partial family (no token
    compression — sparse/merged K-V + optional merge/MLP/unmerge)."""
    spec = PE_REGISTRY.get(name)
    return spec is not None and spec.category == "partial"


def apply_pe(model, name: str, args=None, ratio: Optional[float] = None):
    """Resolve and apply a registered PE algorithm. `name='none'` is a no-op."""
    if name == "none":
        return None
    if name not in PE_REGISTRY:
        raise KeyError(
            f"Unknown PE algorithm {name!r}. "
            f"Valid choices: {algo_choices()}"
        )
    spec = PE_REGISTRY[name]
    kwargs = spec.kwargs_from_args(args, ratio) if spec.kwargs_from_args else {}
    return spec.apply(model, **kwargs)


def remove_all_pe(model) -> int:
    """Idempotently undo every registered PE patch. Walks the model and
    reverts any subclass of `ResidualAttentionBlock`/`SelfAttention` back
    to its stock class (mirroring `remove_all_sam` on the SAM side), then
    removes registered transformer hooks and clears `model._tome_info`.

    Detecting "any subclass" rather than registry-listed classes lets us
    avoid PE imports at registry load time (so SAM-only eval scripts can
    import the registry without `perception_models/` on `sys.path`).

    Safe to call before each sweep config — no-op if no patch is installed.
    """
    try:
        from core.vision_encoder.pe import SelfAttention, ResidualAttentionBlock
    except Exception:
        # PE not importable — nothing to revert. Still try the per-spec
        # `remove` callables for back-compat.
        n = 0
        for spec in PE_REGISTRY.values():
            if spec.remove is not None:
                try:
                    r = spec.remove(model)
                    if isinstance(r, int):
                        n += r
                except Exception:
                    pass
        return n

    n = 0
    for module in model.modules():
        cls = type(module)
        if cls is not ResidualAttentionBlock and issubclass(cls, ResidualAttentionBlock):
            module.__class__ = ResidualAttentionBlock
            n += 1
        elif cls is not SelfAttention and issubclass(cls, SelfAttention):
            module.__class__ = SelfAttention
            n += 1

        # Remove any per-forward state hooks registered on the Transformer.
        for hook_attr in (
            "_pe_compress_pre_hook", "_pe_compress_post_hook",
            "_pe_partial_pre_hook", "_pe_partial_post_hook",
        ):
            handle = getattr(module, hook_attr, None)
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass
                delattr(module, hook_attr)

        # Clear per-module caches & patch-state pointer.
        for attr in ("_flash_cos", "_flash_sin", "_flash_cos_sin_key"):
            if hasattr(module, attr):
                delattr(module, attr)
        if hasattr(module, "_tome_info"):
            try:
                del module._tome_info
            except AttributeError:
                pass

    if hasattr(model, "_tome_info"):
        try:
            del model._tome_info
        except AttributeError:
            pass

    return n


def _kw_compress(args, ratio):
    """For stage-compression patches (tome/gradtome/sparsesam stage)."""
    cab = getattr(args, "compress_at_blocks", None)
    return dict(
        ratio=float(ratio if ratio is not None else args.ratio[0]),
        num_stages=int(args.num_stages),
        group_size=int(args.group_size),
        use_flash_rope=bool(getattr(args, "use_flash_rope", False)),
        compress_at_blocks=list(cab) if cab else None,
    )


def _kw_partial_basic(args, ratio):
    """tome_partial / gradtome_partial: full Q + merged K/V + merge MLP."""
    return dict(
        ratio=float(ratio if ratio is not None else args.ratio[0]),
        start_block=int(getattr(args, "partial_start_block", 0)),
        mlp_merge=bool(getattr(args, "mlp_merge", True)),
    )


def _kw_partial_sparsesam(args, ratio):
    """sparsesam_partial: cute block-sparse attn + uniform-stride perm."""
    return dict(
        ratio=float(ratio if ratio is not None else args.ratio[0]),
        group_size=int(getattr(args, "group_size", 4)),
        sparse_ratio=getattr(args, "sparse_ratio", None),
        start_block=int(getattr(args, "partial_start_block", 0)),
        mlp_merge=bool(getattr(args, "mlp_merge", True)),
    )


def _kw_flash_rope(args, ratio):
    """Standalone fused FA2+RoPE (no compression)."""
    return {}


def _register_builtins():
    # ── stage-compression family ────────────────────────────────────────────
    from .tome.pe_compress import (
        apply_pe_tome_patch, remove_pe_tome_patch,
    )
    register_pe(PEAlgoSpec(
        name="tome",
        apply=apply_pe_tome_patch,
        remove=remove_pe_tome_patch,
        kwargs_from_args=_kw_compress,
        category="compress",
        description="Bipartite-soft-matching token merge at every stage "
                    "boundary. Reduces token count by `ratio` per stage.",
    ))

    from .gradtome.pe_compress import (
        apply_pe_gradtome_patch, remove_pe_gradtome_patch,
    )
    register_pe(PEAlgoSpec(
        name="gradtome",
        apply=apply_pe_gradtome_patch,
        remove=remove_pe_gradtome_patch,
        kwargs_from_args=_kw_compress,
        category="compress",
        description="Spatial-gradient-aware bipartite matching on the (H,W) "
                    "grid. Same stage-compression flow as tome.",
    ))

    from .sparsesam.pe_compress import (
        apply_pe_sparsesam_patch, remove_pe_sparsesam_patch,
        apply_pe_flash_rope_patch, remove_pe_flash_rope_patch,
    )
    register_pe(PEAlgoSpec(
        name="sparsesam",
        apply=apply_pe_sparsesam_patch,
        remove=remove_pe_sparsesam_patch,
        kwargs_from_args=_kw_compress,
        category="compress",
        description="SparseSAM Z-group merge: keep top-K groups verbatim, "
                    "average the rest into one representative each.",
    ))

    register_pe(PEAlgoSpec(
        name="flash_rope",
        apply=apply_pe_flash_rope_patch,
        remove=remove_pe_flash_rope_patch,
        kwargs_from_args=_kw_flash_rope,
        accepts_ratio=False,
        category="attention",
        description="Standalone fused FA2 + 2D-axial RoPE cute kernel. "
                    "Pure attention speedup — no token compression.",
    ))

    # ── _partial: full token count, K/V or MLP compression ─────────────────
    from .tome.pe_partial import (
        apply_pe_tome_partial_patch, remove_pe_tome_partial_patch,
    )
    register_pe(PEAlgoSpec(
        name="tome_partial",
        apply=apply_pe_tome_partial_patch,
        remove=remove_pe_tome_partial_patch,
        kwargs_from_args=_kw_partial_basic,
        category="partial",
        description="Full-Q + ToMe-merged-K/V SDPA + (optional) ToMe-style "
                    "merge → MLP → unmerge. No token compression.",
    ))

    from .gradtome.pe_partial import (
        apply_pe_gradtome_partial_patch, remove_pe_gradtome_partial_patch,
    )
    register_pe(PEAlgoSpec(
        name="gradtome_partial",
        apply=apply_pe_gradtome_partial_patch,
        remove=remove_pe_gradtome_partial_patch,
        kwargs_from_args=_kw_partial_basic,
        category="partial",
        description="Same as tome_partial but with grad-bipartite matching "
                    "on the spatial grid.",
    ))

    from .sparsesam.pe_partial import (
        apply_pe_sparsesam_partial_patch, remove_pe_sparsesam_partial_patch,
    )
    register_pe(PEAlgoSpec(
        name="sparsesam_partial",
        apply=apply_pe_sparsesam_partial_patch,
        remove=remove_pe_sparsesam_partial_patch,
        kwargs_from_args=_kw_partial_sparsesam,
        category="partial",
        description="Block-sparse cute-kernel attention (uniform-stride "
                    "perm + keep-bar mask) + (optional) merge/MLP/unmerge.",
    ))


_register_builtins()


# SAM side — same registry pattern, separate dict (different patch surface:
# `segment_anything.modeling.image_encoder.{Block, Attention}` vs PE's
# `core.vision_encoder.pe.SelfAttention`).

@dataclass
class SAMAlgoSpec:
    name: str
    apply: Callable[..., Any]                  # (encoder, **kwargs) -> Any
    block_class: Optional[type] = None         # subclass of Block, or None
    attn_class: Optional[type] = None          # subclass of Attention, or None
    kwargs_from_args: Optional[Callable] = None
    accepts_ratio: bool = True
    description: str = ""


SAM_REGISTRY: Dict[str, SAMAlgoSpec] = {}


def register_sam(spec: SAMAlgoSpec) -> SAMAlgoSpec:
    if spec.name in SAM_REGISTRY:
        raise ValueError(f"SAM algorithm {spec.name!r} already registered")
    SAM_REGISTRY[spec.name] = spec
    return spec


def sam_algo_choices() -> List[str]:
    return ["none"] + sorted(SAM_REGISTRY.keys())


def apply_sam(encoder, name: str, args=None, ratio: Optional[float] = None,
              **extra):
    """Resolve and apply a registered SAM algorithm. `name='none'` and
    `ratio>=1.0` are no-ops."""
    if name == "none" or (ratio is not None and ratio >= 1.0):
        return None
    if name not in SAM_REGISTRY:
        raise KeyError(
            f"Unknown SAM algorithm {name!r}. "
            f"Valid choices: {sam_algo_choices()}"
        )
    spec = SAM_REGISTRY[name]
    kwargs = (spec.kwargs_from_args(args, ratio)
              if spec.kwargs_from_args else dict(ratio=ratio))
    kwargs.update(extra)
    return spec.apply(encoder, **kwargs)


def remove_all_sam(encoder, mask_decoder=None) -> int:
    """Idempotently undo every registered SAM patch by reverting subclassed
    Block/Attention modules to their originals and clearing patch state.
    Pass `mask_decoder` to also undo any HQ-decoder permutations.
    """
    from segment_anything.modeling.image_encoder import Block, Attention

    block_classes = tuple(s.block_class for s in SAM_REGISTRY.values()
                          if s.block_class is not None)
    attn_classes  = tuple(s.attn_class  for s in SAM_REGISTRY.values()
                          if s.attn_class  is not None)

    n = 0
    for module in encoder.modules():
        if block_classes and type(module) in block_classes:
            module.__class__ = Block
            n += 1
        elif attn_classes and type(module) in attn_classes:
            module.__class__ = Attention
            n += 1

    # Clear patch info / instance-level forward overrides.
    if hasattr(encoder, 'tome_info'):
        del encoder.tome_info
    encoder.__dict__.pop('forward', None)
    for blk in getattr(encoder, 'blocks', []):
        blk.__dict__.pop('forward', None)
    return n


def update_sam_ratio(encoder, ratio: float):
    """Update merge ratio without re-patching (works if already patched)."""
    if hasattr(encoder, 'tome_info'):
        encoder.tome_info['ratio'] = ratio


def _sam_kw_basic(args, ratio):
    """Stage-compression patches share the same (algo, ratio, margin) shape.
    The `algo` kwarg is the *internal* string the patch's apply_patch fn
    branches on (e.g. 'tome' vs 'pitome' inside the same patch module).
    `mlp_merge` is consumed by patches that support a partial-MLP path
    (sparsesam family); other patches ignore it."""
    return dict(
        ratio=float(ratio if ratio is not None else args.ratios[0]),
        margin=float(getattr(args, "margin", 0.5)),
        mlp_merge=bool(getattr(args, "mlp_merge", True)),
    )


import inspect as _inspect


def _wrap_apply_with_internal_algo(apply_fn, internal_algo):
    """Wrap a patch's `apply_patch(encoder, algo=, ratio=, margin=, …)`
    with the algo string baked in. Optional kwargs (e.g. `mlp_merge`) are
    forwarded only to patches whose signature actually accepts them, so
    legacy patches without the option keep working unchanged."""
    accepted = set(_inspect.signature(apply_fn).parameters)

    def _apply(encoder, ratio, margin=0.5, **extra):
        kw = dict(algo=internal_algo, ratio=ratio, margin=margin)
        for k, v in extra.items():
            if k in accepted:
                kw[k] = v
        return apply_fn(encoder, **kw)
    return _apply


def _register_sam_builtins():
    from .tome.sam              import (apply_patch as _patch_tome,
                                              ToMeSAMBlock as _B_t,
                                              ToMeSAMAttention as _A_t)
    from .sparsesam.sam         import (apply_patch as _patch_sparsesam,
                                              ToMeSAMBlock as _B_s,
                                              ToMeSAMAttention as _A_s)
    from .sparsesam.sam_random  import (apply_patch as _patch_sparsesam_random,
                                              ToMeSAMBlockRandom as _B_sr,
                                              ToMeSAMAttentionRandom as _A_sr)
    from .gradtome.sam          import (apply_patch as _patch_gradtome,
                                              ToMeSAMBlock as _B_g,
                                              ToMeSAMAttention as _A_g)
    from .gradtome.sam_hilbert  import (apply_patch as _patch_gradtome_hilbert,
                                              ToMeSAMBlock as _B_gh,
                                              ToMeSAMAttention as _A_gh)

    # Plain ToMe + PiToMe (same patch module, different `algo` string).
    register_sam(SAMAlgoSpec(
        name="tome",
        apply=_wrap_apply_with_internal_algo(_patch_tome, "tome"),
        block_class=_B_t, attn_class=_A_t,
        kwargs_from_args=_sam_kw_basic,
        description="Bipartite token merge per block on the SAM-HQ encoder.",
    ))
    register_sam(SAMAlgoSpec(
        name="pitome",
        apply=_wrap_apply_with_internal_algo(_patch_tome, "pitome"),
        block_class=_B_t, attn_class=_A_t,
        kwargs_from_args=_sam_kw_basic,
        description="PiToMe (energy-margin) variant of ToMe.",
    ))

    register_sam(SAMAlgoSpec(
        name="sparsesam",
        apply=_wrap_apply_with_internal_algo(_patch_sparsesam, "tome"),
        block_class=_B_s, attn_class=_A_s,
        kwargs_from_args=_sam_kw_basic,
        description="SparseSAM Z-group merge on the SAM-HQ encoder.",
    ))
    register_sam(SAMAlgoSpec(
        name="sparsesam_pitome",
        apply=_wrap_apply_with_internal_algo(_patch_sparsesam, "pitome"),
        block_class=_B_s, attn_class=_A_s,
        kwargs_from_args=_sam_kw_basic,
        description="SparseSAM with PiToMe energy-margin matching.",
    ))
    register_sam(SAMAlgoSpec(
        name="sparsesam_random",
        apply=_wrap_apply_with_internal_algo(_patch_sparsesam_random,
                                             "sparsesam_random"),
        block_class=_B_sr, attn_class=_A_sr,
        kwargs_from_args=_sam_kw_basic,
        description="SparseSAM with random keep-set selection (sanity baseline).",
    ))

    register_sam(SAMAlgoSpec(
        name="gradtome",
        apply=_wrap_apply_with_internal_algo(_patch_gradtome, "tome"),
        block_class=_B_g, attn_class=_A_g,
        kwargs_from_args=_sam_kw_basic,
        description="Gradient-aware bipartite matching on the spatial grid.",
    ))
    register_sam(SAMAlgoSpec(
        name="gradtome_pitome",
        apply=_wrap_apply_with_internal_algo(_patch_gradtome, "pitome"),
        block_class=_B_g, attn_class=_A_g,
        kwargs_from_args=_sam_kw_basic,
        description="GradToMe with PiToMe energy-margin matching.",
    ))
    register_sam(SAMAlgoSpec(
        name="gradtome_hilbert",
        apply=_wrap_apply_with_internal_algo(_patch_gradtome_hilbert, "tome"),
        block_class=_B_gh, attn_class=_A_gh,
        kwargs_from_args=_sam_kw_basic,
        description="GradToMe with Hilbert-curve token ordering.",
    ))


_register_sam_builtins()


__all__ = [
    "PEAlgoSpec", "PE_REGISTRY", "register_pe",
    "algo_choices", "is_partial",
    "apply_pe", "remove_all_pe",
    "SAMAlgoSpec", "SAM_REGISTRY", "register_sam",
    "sam_algo_choices",
    "apply_sam", "remove_all_sam", "update_sam_ratio",
]
