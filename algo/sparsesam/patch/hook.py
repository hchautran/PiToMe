"""
Hook patch for SAM ImageEncoderViT.

Registers a forward pre-hook on every Block (and optionally on the Attention
and MLP sub-modules) so that the raw input tensor arriving at each layer is
captured and stored for later inspection (e.g. for calibrating quantisation
ranges, computing statistics, or debugging).

Captured data
-------------
After ``encoder.forward(x)`` completes, the captured tensors live in:

    encoder.hook_info["block_inputs"]    : list[Tensor]  – one per Block,
                                           shape (B, H, W, C) *before* norm1.
    encoder.hook_info["attn_inputs"]     : list[Tensor]  – one per Attention,
                                           shape (B, H, W, C) or (B*nW, ws, ws, C)
                                           (the tensor that arrives at attn.forward).
    encoder.hook_info["mlp_inputs"]      : list[Tensor]  – one per MLPBlock,
                                           shape (B, N, C) (the tensor that arrives
                                           at mlp.forward, *after* norm2).

    All lists are ordered by block index (layer 0 → layer depth-1).

Usage
-----
    from PiToMe.algo.sparsesam.patch.hook import apply_patch as hook
    encoder = hook(encoder, capture=("block", "attn", "mlp"))
    encoder(image)

    inputs_layer0 = encoder.hook_info["block_inputs"][0]   # (B, H, W, C)

Remove hooks
------------
    encoder.hook_info["remove"]()   # de-register all hooks cleanly
"""

import sys
import os
import types
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# ── resolve SAM-1 on sys.path ─────────────────────────────────────────────────
_here = os.path.dirname(__file__)
_sam_root = os.path.normpath(os.path.join(_here, "..", "..", "..", "..", "sam-hq"))
if _sam_root not in sys.path:
    sys.path.insert(0, _sam_root)

from segment_anything.modeling.image_encoder import ImageEncoderViT, Block, Attention
from segment_anything.modeling.common import MLPBlock


# ─────────────────────────────────────────────────────────────────────────────
# apply_patch
# ─────────────────────────────────────────────────────────────────────────────

def apply_patch(
    encoder: ImageEncoderViT,
    capture: Sequence[str] = ("block",),
) -> ImageEncoderViT:
    """
    Register forward pre-hooks on every requested layer type so that the input
    tensor *before* each layer is captured and stored.

    Parameters
    ----------
    encoder : ImageEncoderViT
        The SAM image encoder to instrument (modified in-place).
    capture : sequence of str
        Which layers to hook.  Any combination of:
          - ``"block"``  – the transformer Block (input before norm1+attn+mlp)
          - ``"attn"``   – the Attention sub-module inside each Block
          - ``"mlp"``    – the MLPBlock sub-module inside each Block

    Returns
    -------
    encoder : ImageEncoderViT
        The same encoder, now instrumented with hooks.
    """
    valid = {"block", "attn", "mlp"}
    unknown = set(capture) - valid
    if unknown:
        raise ValueError(f"Unknown capture targets: {unknown}.  Choose from {valid}.")

    hook_info: Dict[str, object] = {
        "block_inputs": [],   # list[Tensor] – filled on each forward pass
        "attn_inputs":  [],
        "mlp_inputs":   [],
        "remove":       None, # callable that de-registers all hooks
    }
    encoder.hook_info = hook_info

    handles: List[torch.utils.hooks.RemovableHook] = []

    # ── helper: pre-hook factory ──────────────────────────────────────────────
    def _make_pre_hook(store_list: List[torch.Tensor]):
        """
        Returns a forward pre-hook that appends a *detached cpu copy* of the
        first positional input tensor to ``store_list``.
        """
        def _hook(module: nn.Module, args: Tuple) -> None:
            if args:
                x = args[0]
                if isinstance(x, torch.Tensor):
                    store_list.append(x.detach().cpu())
        return _hook

    # ── helper: reset lists at the start of each encoder forward pass ─────────
    _orig_forward = encoder.__class__.forward

    def _patched_forward(self: ImageEncoderViT, x: torch.Tensor):
        # Clear previous captures before every new image
        self.hook_info["block_inputs"].clear()
        self.hook_info["attn_inputs"].clear()
        self.hook_info["mlp_inputs"].clear()
        return _orig_forward(self, x)

    encoder.forward = types.MethodType(_patched_forward, encoder)

    # ── register hooks on every Block (and its children) ─────────────────────
    for blk in encoder.blocks:
        if not isinstance(blk, Block):
            continue

        if "block" in capture:
            h = blk.register_forward_pre_hook(
                _make_pre_hook(hook_info["block_inputs"])
            )
            handles.append(h)

        if "attn" in capture and isinstance(blk.attn, Attention):
            h = blk.attn.register_forward_pre_hook(
                _make_pre_hook(hook_info["attn_inputs"])
            )
            handles.append(h)

        if "mlp" in capture and isinstance(blk.mlp, MLPBlock):
            h = blk.mlp.register_forward_pre_hook(
                _make_pre_hook(hook_info["mlp_inputs"])
            )
            handles.append(h)

    # ── cleanup callable ──────────────────────────────────────────────────────
    def _remove_all():
        for h in handles:
            h.remove()
        handles.clear()
        # Restore the original forward method
        if hasattr(encoder, "forward") and isinstance(encoder.forward, types.MethodType):
            del encoder.forward   # falls back to class-level forward

    hook_info["remove"] = _remove_all

    # ── summary ──────────────────────────────────────────────────────────────
    n_blocks = len(encoder.blocks)
    n_global = sum(1 for blk in encoder.blocks if blk.window_size == 0)
    print(
        f"[Hook-SAM] registered pre-hooks"
        f"  capture={list(capture)}"
        f"  blocks={n_blocks} (global={n_global}  local={n_blocks - n_global})"
        f"  handles={len(handles)}"
    )

    return encoder
