"""ToMe partial: full-Q + merged-K/V attention + merge/MLP/unmerge.

Per block: bipartite-match `x`, run SDPA(Q_full, K_merged, V_merged),
then reuse the match for merge -> MLP -> unmerge. Token count preserved.
"""

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .._pe_stage import (
    SelfAttention, ResidualAttentionBlock,
    _find_vision_transformer, _vit_uses_cls_token,
)
from .merge import bipartite_soft_matching, do_nothing


def _chained_bipartite_match(metric: torch.Tensor, ratio: float,
                             class_token: bool):
    """Bipartite that supports `ratio < 0.5` via chained passes."""
    if ratio >= 0.5 - 1e-6:
        return bipartite_soft_matching(metric, ratio=ratio, class_token=class_token)

    n_passes = max(1, math.ceil(math.log(max(ratio, 1e-6), 0.5)))
    per_pass_ratio = max(ratio ** (1.0 / n_passes), 0.5 + 1e-6)

    merges, unmerges = [], []
    current_metric = metric
    achieved = 1.0
    for i in range(n_passes):
        if i == n_passes - 1:
            r_pass = max(min(ratio / achieved, 1.0), 0.5 + 1e-6)
        else:
            r_pass = per_pass_ratio
        m, u = bipartite_soft_matching(current_metric, ratio=r_pass,
                                       class_token=class_token)
        if m is do_nothing:
            break
        merges.append(m); unmerges.append(u)
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

class TomePEPartialAttention(SelfAttention):
    """Full-Q + ToMe-merged-K/V SDPA. Match is built from `x` so its `T`
    matches `k_flat.shape[1]`; cached on `info` for the block to reuse."""

    def forward(self, x, attn_mask=None):
        info    = self._tome_info
        ratio   = info.get("ratio", 1.0)
        has_cls = info.get("use_cls_token", False)

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

        # RoPE before merge -> merged K averages positionally-encoded values.
        if self.rope:
            q, k = self.rope(q, k)

        k_flat = rearrange(k, "b h s d -> b s (h d)")
        v_flat = rearrange(v, "b h s d -> b s (h d)")

        metric = x.mean(0, keepdim=True)
        merge_fn, unmerge_fn = _chained_bipartite_match(
            metric, ratio=ratio, class_token=has_cls,
        )
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


class TomePEPartialBlock(ResidualAttentionBlock):
    """Block forward: attn (which stashes the match) + optional merge/MLP/unmerge."""

    def forward(self, x, attn_mask=None):
        info: dict = self._tome_info
        info["_block_merge_fn"] = info["_block_unmerge_fn"] = None

        x = x + self.drop_path1(
            self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask))
        )

        merge_fn   = info.get("_block_merge_fn")
        unmerge_fn = info.get("_block_unmerge_fn")
        if (info.get("mlp_merge", True)
                and merge_fn is not None and merge_fn is not do_nothing):
            x_merged, _ = merge_fn(x, mode="mean")
            x_merged = x_merged + self.drop_path2(
                self.ls_2(self.mlp(self.ln_2(x_merged)))
            )
            x = unmerge_fn(x_merged)
        else:
            x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))

        info["_block_merge_fn"] = info["_block_unmerge_fn"] = None
        return x


def apply_pe_tome_partial_patch(model: nn.Module,
                                ratio: float = 0.7,
                                start_block: int = 0,
                                mlp_merge: bool = True,
                                verbose: bool = True) -> int:
    """ToMe merged-K/V attention + optional merge/MLP/unmerge.
    `start_block`: blocks below this stay stock SDPA + full MLP."""
    transformer = _find_vision_transformer(model)
    if transformer is None:
        raise RuntimeError("Could not locate the PE vision Transformer.")

    use_cls_token = _vit_uses_cls_token(model)

    n_blocks = len(transformer.resblocks)
    sb = max(0, min(int(start_block), n_blocks))

    info = {
        "ratio":         float(ratio),
        "use_cls_token": use_cls_token,
        "mlp_merge":     bool(mlp_merge),
    }
    model._tome_info = info

    n_attn = 0
    for idx in range(sb, n_blocks):
        blk = transformer.resblocks[idx]
        attn = getattr(blk, "attn", None)
        if isinstance(attn, SelfAttention) and attn.rope is not None:
            if not isinstance(attn, TomePEPartialAttention):
                attn.__class__ = TomePEPartialAttention
            attn._tome_info = info
            n_attn += 1
        if not isinstance(blk, TomePEPartialBlock):
            blk.__class__ = TomePEPartialBlock
        blk._tome_info = info

    if verbose:
        print(f"[pe-tome-partial] L={n_blocks}  start_block={sb}  "
              f"patched_blocks={n_blocks - sb}  patched_attn={n_attn}  "
              f"ratio={info['ratio']}  use_cls_token={use_cls_token}  "
              f"mlp_merge={info['mlp_merge']}  "
              f"(blocks 0..{sb - 1}: stock; "
              f"blocks {sb}..{n_blocks - 1}: full-Q + ToMe-merged-K/V SDPA + "
              f"{'merge/MLP/unmerge' if info['mlp_merge'] else 'full MLP'})")
    return n_blocks - sb


def get_classes() -> Tuple[type, type]:
    return TomePEPartialBlock, TomePEPartialAttention


def remove_pe_tome_partial_patch(model: nn.Module) -> int:
    from ..registry import remove_all_pe
    return remove_all_pe(model)


__all__ = [
    "apply_pe_tome_partial_patch",
    "remove_pe_tome_partial_patch",
    "get_classes",
]
