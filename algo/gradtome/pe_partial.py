"""GradToMe partial: spatial-gradient-aware merged-K/V attention +
merge/MLP/unmerge. Falls back to plain (chained) bipartite when the spatial
grid isn't a perfect square. 
"""

from __future__ import annotations
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .._pe_stage import (
    SelfAttention, ResidualAttentionBlock,
    _find_vision_transformer, _vit_uses_cls_token,
)
from .merge import grad_bipartite_soft_matching, do_nothing
from ..tome.merge import bipartite_soft_matching


def _grad_match_single(metric_sp: torch.Tensor, ratio: float,
                       sx: int, sy: int):
    N = metric_sp.shape[1]
    r_grid = int(math.isqrt(N))
    if r_grid * r_grid != N:
        return None

    r = math.floor(N - N * ratio)
    return grad_bipartite_soft_matching(
        metric=metric_sp, H=r_grid, W=r_grid, sx=sx, sy=sy, r=r,
    )


def _build_grad_match(metric: torch.Tensor, ratio: float, has_cls: bool,
                      sx: int, sy: int) -> Tuple[Callable, Callable]:
    """Build (merge, unmerge) over the spatial grid. Falls back to chained
    plain bipartite when off-square or when ratio < 0.5 needs chaining."""
    cls_off = 1 if has_cls else 0
    metric_sp = metric[:, cls_off:, :].contiguous()
    N = metric_sp.shape[1]
    r_grid = int(math.isqrt(N))

    if r_grid * r_grid != N or ratio < 0.5 - 1e-6:
        return _chained_bipartite_match(metric, ratio, class_token=has_cls)

    sp = _grad_match_single(metric_sp, ratio, sx, sy)
    if sp is None:
        return _chained_bipartite_match(metric, ratio, class_token=has_cls)
    sp_merge, sp_unmerge = sp
    if sp_merge is do_nothing:
        return do_nothing, do_nothing

    if not has_cls:
        return sp_merge, sp_unmerge

    def merge(x, mode="mean"):
        cls_tok = x[:, :cls_off, :]
        sp = x[:, cls_off:, :].contiguous()
        sp_m, _ = sp_merge(sp, mode=mode)
        return torch.cat([cls_tok, sp_m], dim=1), None

    def unmerge(x):
        cls_tok = x[:, :cls_off, :]
        sp = x[:, cls_off:, :]
        sp_full = sp_unmerge(sp)
        return torch.cat([cls_tok, sp_full], dim=1)

    return merge, unmerge


def _chained_bipartite_match(metric: torch.Tensor, ratio: float,
                             class_token: bool):
    """Same chained-bipartite helper as `tome/pe_partial.py` —
    duplicated here so this module doesn't pull from the sibling tome package."""
    if ratio >= 0.5 - 1e-6:
        return bipartite_soft_matching(metric, ratio=ratio, class_token=class_token)

    n_passes = max(1, math.ceil(math.log(max(ratio, 1e-6), 0.5)))
    per_pass_ratio = ratio ** (1.0 / n_passes)
    per_pass_ratio = max(per_pass_ratio, 0.5 + 1e-6)

    merges, unmerges = [], []
    current_metric = metric
    achieved = 1.0
    for i in range(n_passes):
        if i == n_passes - 1:
            need = ratio / achieved
            r_pass = max(min(need, 1.0), 0.5 + 1e-6)
        else:
            r_pass = per_pass_ratio

        m, u = bipartite_soft_matching(current_metric, ratio=r_pass,
                                       class_token=class_token)
        if m is do_nothing:
            break
        merges.append(m)
        unmerges.append(u)
        achieved *= r_pass
        current_metric, _ = m(current_metric, mode="mean")

    if not merges:
        return do_nothing, do_nothing

    def merge_chained(x, mode="mean"):
        for m in merges:
            x, _ = m(x, mode=mode)
        return x, None

    def unmerge_chained(x):
        for u in reversed(unmerges):
            x = u(x)
        return x

    return merge_chained, unmerge_chained


# ── Subclasses ───────────────────────────────────────────────────────────

class GradTomePEPartialAttention(SelfAttention):
    """SelfAttention with full Q and GradToMe-merged K/V."""

    def forward(self, x, attn_mask=None):
        info    = self._tome_info
        ratio   = info.get("ratio", 1.0)
        has_cls = info.get("use_cls_token", False)
        sx      = info.get("grad_sx", 2)
        sy      = info.get("grad_sy", 2)

        if ratio >= 1.0:
            info["_block_merge_fn"]   = do_nothing
            info["_block_unmerge_fn"] = do_nothing
            return super().forward(x, attn_mask=attn_mask)

        B, S, E = x.shape
        H = self.num_heads
        D = self.head_dim

        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        proj = (proj.unflatten(-1, (3, E))
                     .unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous())
        q, k, v = proj[0], proj[1], proj[2]

        q = rearrange(q, "b s (h d) -> b h s d", h=H)
        k = rearrange(k, "b s (h d) -> b h s d", h=H)
        v = rearrange(v, "b s (h d) -> b h s d", h=H)

        if self.rope:
            q, k = self.rope(q, k)

        k_flat = rearrange(k, "b h s d -> b s (h d)")
        v_flat = rearrange(v, "b h s d -> b s (h d)")

        metric = x.mean(0, keepdim=True)
        merge_fn, unmerge_fn = _build_grad_match(metric, ratio, has_cls, sx, sy)
        info["_block_merge_fn"]   = merge_fn
        info["_block_unmerge_fn"] = unmerge_fn

        if merge_fn is do_nothing:
            return super().forward(x, attn_mask=attn_mask)

        k_merged_flat, _ = merge_fn(k_flat, mode="mean")
        v_merged_flat, _ = merge_fn(v_flat, mode="mean")

        k_merged = rearrange(k_merged_flat, "b s (h d) -> b h s d", h=H)
        v_merged = rearrange(v_merged_flat, "b s (h d) -> b h s d", h=H)

        attn = F.scaled_dot_product_attention(
            q, k_merged, v_merged,
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale,
        )
        attn = rearrange(attn, "b h s d -> b s (h d)")
        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)


class GradTomePEPartialBlock(ResidualAttentionBlock):
    """attn (computes per-block grad matching) + optional merge/MLP/unmerge."""

    def forward(self, x, attn_mask=None):
        info: dict = self._tome_info

        info["_block_merge_fn"]   = None
        info["_block_unmerge_fn"] = None

        x = x + self.drop_path1(
            self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask))
        )

        merge_fn   = info.get("_block_merge_fn")
        unmerge_fn = info.get("_block_unmerge_fn")
        mlp_merge  = info.get("mlp_merge", True)

        if mlp_merge and merge_fn is not None and merge_fn is not do_nothing:
            x_merged, _ = merge_fn(x, mode="mean")
            x_merged = x_merged + self.drop_path2(
                self.ls_2(self.mlp(self.ln_2(x_merged)))
            )
            x = unmerge_fn(x_merged)
        else:
            x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))

        info["_block_merge_fn"]   = None
        info["_block_unmerge_fn"] = None
        return x


def apply_pe_gradtome_partial_patch(model: nn.Module,
                                    ratio: float = 0.7,
                                    grad_sx: int = 2,
                                    grad_sy: int = 2,
                                    start_block: int = 0,
                                    mlp_merge: bool = True,
                                    verbose: bool = True) -> int:
    """GradToMe merged-K/V attention + optional merge/MLP/unmerge.

    `mlp_merge`: True (default) runs MLP on merged tokens then unmerges back;
    False runs MLP on full S — only K/V attention compression remains."""
    transformer = _find_vision_transformer(model)
    if transformer is None:
        raise RuntimeError("Could not locate the PE vision Transformer.")

    use_cls_token = _vit_uses_cls_token(model)

    n_blocks = len(transformer.resblocks)
    sb = max(0, min(int(start_block), n_blocks))

    info = {
        "ratio":         float(ratio),
        "grad_sx":       int(grad_sx),
        "grad_sy":       int(grad_sy),
        "use_cls_token": use_cls_token,
        "mlp_merge":     bool(mlp_merge),
    }
    model._tome_info = info

    n_attn = 0
    for idx in range(sb, n_blocks):
        blk = transformer.resblocks[idx]
        attn = getattr(blk, "attn", None)
        if isinstance(attn, SelfAttention) and attn.rope is not None:
            if not isinstance(attn, GradTomePEPartialAttention):
                attn.__class__ = GradTomePEPartialAttention
            attn._tome_info = info
            n_attn += 1
        if not isinstance(blk, GradTomePEPartialBlock):
            blk.__class__ = GradTomePEPartialBlock
        blk._tome_info = info

    if verbose:
        print(f"[pe-gradtome-partial] L={n_blocks}  start_block={sb}  "
              f"patched_blocks={n_blocks - sb}  patched_attn={n_attn}  "
              f"ratio={info['ratio']}  grad_sx={info['grad_sx']}  grad_sy={info['grad_sy']}  "
              f"use_cls_token={use_cls_token}  mlp_merge={info['mlp_merge']}  "
              f"(blocks 0..{sb - 1}: stock; "
              f"blocks {sb}..{n_blocks - 1}: full-Q + GradToMe-merged-K/V SDPA + "
              f"{'merge/MLP/unmerge' if info['mlp_merge'] else 'full MLP'})")
    return n_blocks - sb


def get_classes() -> Tuple[type, type]:
    return GradTomePEPartialBlock, GradTomePEPartialAttention


def remove_pe_gradtome_partial_patch(model: nn.Module) -> int:
    from ..registry import remove_all_pe
    return remove_all_pe(model)


__all__ = [
    "apply_pe_gradtome_partial_patch",
    "remove_pe_gradtome_partial_patch",
    "get_classes",
]
