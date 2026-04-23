"""
Channel-sort patch for SAM ImageEncoderViT.

Reorders the residual-stream channels of the encoder so that channel 0 has
the highest per-channel activation std and channel C-1 has the lowest.  The
permutation is *globally consistent* across all blocks: because SAM uses
residual connections (out = x + f(x)), every block shares the same channel
ordering, so a single permutation P covers the whole encoder.

Layers modified (in-place on a copy, or in-place on the original):
    patch_embed.proj       – output channels
    pos_embed              – channel dim
    block[i].norm1/norm2   – weight & bias
    block[i].attn.qkv      – input channels
    block[i].attn.proj     – output channels
    block[i].mlp.lin1      – input channels
    block[i].mlp.lin2      – output channels
    neck[0]                – input channels (Conv2d)

After sorting the channel-sort info is stored on the encoder:
    encoder.channel_sort_info["sort_idx"]   (C,) long tensor – sorted channel order
    encoder.channel_sort_info["inv_idx"]    (C,) long tensor – inverse permutation
    encoder.channel_sort_info["ch_std"]     (C,) float tensor – std in original order

Usage
-----
    from algo.sparsesam.patch.channel_sort_sam import apply_patch as channel_sort_sam

    # sort only (default: uses one dummy forward pass for calibration)
    encoder = channel_sort_sam(encoder)

    # sort + then apply ToMe/PiToMe on top
    encoder = channel_sort_sam(encoder, apply_tome=True, algo="tome", ratio=0.9)

    # provide your own calibration images (B, 3, 1024, 1024)
    encoder = channel_sort_sam(encoder, calib_images=my_batch)
"""

import sys
import os
import copy
import types
from typing import Optional

import torch
import torch.nn as nn

_here     = os.path.dirname(__file__)
_sam_root = os.path.normpath(os.path.join(_here, "..", "..", "..", "..", "sam-hq"))
if _sam_root not in sys.path:
    sys.path.insert(0, _sam_root)

from segment_anything.modeling.image_encoder import ImageEncoderViT
from .hook import apply_patch as _hook


# ─────────────────────────────────────────────────────────────────────────────
# Internal: calibration
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_channel_metric(
    encoder: ImageEncoderViT,
    calib_images: Optional[torch.Tensor],
    sort_by: str,
) -> torch.Tensor:
    """
    Run a forward pass through *encoder* with block-input hooks and return a
    per-channel importance score (C,) according to *sort_by*:
      - "std"       : std across all blocks × spatial positions × batch
      - "magnitude" : mean absolute value across the same axes
    """
    device = next(encoder.parameters()).device
    dtype  = next(encoder.parameters()).dtype

    if calib_images is None:
        calib_images = torch.randn(
            1, 3, encoder.img_size, encoder.img_size,
            dtype=dtype, device=device,
        )
    else:
        calib_images = calib_images.to(device=device, dtype=dtype)

    encoder_hook = _hook(encoder, capture=("block",))

    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        encoder_hook(calib_images)
    if was_training:
        encoder.train()

    block_inputs = encoder_hook.hook_info["block_inputs"]   # list of (B, H, W, C)
    encoder_hook.hook_info["remove"]()

    all_acts = torch.stack([t.float() for t in block_inputs], dim=0)  # (n_blocks, B, H, W, C)

    if sort_by == "std":
        return all_acts.std(dim=(0, 1, 2, 3))           # (C,)
    elif sort_by == "magnitude":
        return all_acts.abs().mean(dim=(0, 1, 2, 3))    # (C,)
    else:
        raise ValueError(f"sort_by must be 'std' or 'magnitude', got {sort_by!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Internal: weight permutation
# ─────────────────────────────────────────────────────────────────────────────

def _apply_channel_permutation(
    encoder: ImageEncoderViT,
    sort_idx: torch.Tensor,
) -> None:
    """Apply channel permutation *in-place* to all weights in the residual stream."""

    with torch.no_grad():
        # ── patch_embed ───────────────────────────────────────────────────────
        pe = encoder.patch_embed.proj
        pe.weight.data = pe.weight.data[sort_idx]
        if pe.bias is not None:
            pe.bias.data = pe.bias.data[sort_idx]

        # ── pos_embed ─────────────────────────────────────────────────────────
        if encoder.pos_embed is not None:
            encoder.pos_embed.data = encoder.pos_embed.data[..., sort_idx]

        # ── transformer blocks ────────────────────────────────────────────────
        for blk in encoder.blocks:
            # norm1 – reads from sorted residual stream
            blk.norm1.weight.data = blk.norm1.weight.data[sort_idx]
            blk.norm1.bias.data   = blk.norm1.bias.data[sort_idx]

            # attn.qkv – input channels come from sorted stream
            blk.attn.qkv.weight.data = blk.attn.qkv.weight.data[:, sort_idx]
            # bias is (3C,) added to the qkv output – independent of input order

            # attn.proj – output must write back to sorted residual stream
            blk.attn.proj.weight.data = blk.attn.proj.weight.data[sort_idx, :]
            if blk.attn.proj.bias is not None:
                blk.attn.proj.bias.data = blk.attn.proj.bias.data[sort_idx]

            # norm2 – reads from sorted residual stream (after attn residual add)
            blk.norm2.weight.data = blk.norm2.weight.data[sort_idx]
            blk.norm2.bias.data   = blk.norm2.bias.data[sort_idx]

            # mlp.lin1 – input channels from sorted stream
            blk.mlp.lin1.weight.data = blk.mlp.lin1.weight.data[:, sort_idx]
            # bias (mlp_dim,) – added to lin1 output, independent of input order

            # mlp.lin2 – output must write back to sorted residual stream
            blk.mlp.lin2.weight.data = blk.mlp.lin2.weight.data[sort_idx, :]
            if blk.mlp.lin2.bias is not None:
                blk.mlp.lin2.bias.data = blk.mlp.lin2.bias.data[sort_idx]

        # ── neck – first conv reads from sorted residual stream ───────────────
        neck_conv = encoder.neck[0]
        neck_conv.weight.data = neck_conv.weight.data[:, sort_idx, :, :]


def _calibrate_mlp_hidden_metric(
    encoder: ImageEncoderViT,
    calib_images: Optional[torch.Tensor],
    sort_by: str,
) -> list:
    """Run a forward pass and return a per-channel importance score (mlp_dim,)
    for each block's MLP hidden dim, captured as the input to lin2 (= GELU output)."""
    device = next(encoder.parameters()).device
    dtype  = next(encoder.parameters()).dtype

    if calib_images is None:
        calib_images = torch.randn(
            1, 3, encoder.img_size, encoder.img_size,
            dtype=dtype, device=device,
        )
    else:
        calib_images = calib_images.to(device=device, dtype=dtype)

    captured = [[] for _ in encoder.blocks]
    handles  = []
    for i, blk in enumerate(encoder.blocks):
        def _make_hook(idx):
            def _h(_module, args):
                if args:
                    captured[idx].append(args[0].detach().cpu().float())
            return _h
        handles.append(blk.mlp.lin2.register_forward_pre_hook(_make_hook(i)))

    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        encoder(calib_images)
    if was_training:
        encoder.train()
    for h in handles:
        h.remove()

    metrics = []
    for acts in captured:
        x = acts[0].reshape(-1, acts[0].shape[-1])   # (N, mlp_dim)
        if sort_by == "std":
            metrics.append(x.std(dim=0))
        else:
            metrics.append(x.abs().mean(dim=0))
    return metrics   # list of (mlp_dim,) tensors, one per block


def _apply_mlp_hidden_permutation(
    encoder: ImageEncoderViT,
    mlp_sort_indices: list,
) -> None:
    """Permute lin1 output rows and lin2 input columns per block in-place."""
    with torch.no_grad():
        for blk, idx in zip(encoder.blocks, mlp_sort_indices):
            idx = idx.to(blk.mlp.lin1.weight.device)
            # lin1 output → reorder which hidden neuron goes where
            blk.mlp.lin1.weight.data = blk.mlp.lin1.weight.data[idx]
            if blk.mlp.lin1.bias is not None:
                blk.mlp.lin1.bias.data = blk.mlp.lin1.bias.data[idx]
            # lin2 input columns must match the new hidden order
            blk.mlp.lin2.weight.data = blk.mlp.lin2.weight.data[:, idx]


def _apply_hq_decoder_permutation(
    mask_decoder: nn.Module,
    sort_idx: torch.Tensor,
) -> None:
    """Permute compress_vit_feat[0] input channels in the SAM-HQ mask decoder.

    SAM-HQ feeds interm_embeddings[0] (a global-block output in sorted channel
    space) into compress_vit_feat, a ConvTranspose2d whose weights expect the
    original channel order.  This fixes that mismatch in-place.
    """
    with torch.no_grad():
        cvf = mask_decoder.compress_vit_feat[0]   # ConvTranspose2d(vit_dim, ...)
        # weight shape: (in_channels, out_channels, kH, kW)
        cvf.weight.data = cvf.weight.data[sort_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def apply_patch(
    encoder: ImageEncoderViT,
    *,
    calib_images: Optional[torch.Tensor] = None,
    sort_by: str = "std",
    inplace: bool = True,
    apply_tome: bool = False,
    algo: str = "tome",
    ratio: float = 0.9,
    margin: float = 0.5,
    trace_source: bool = False,
    mask_decoder: Optional[nn.Module] = None,
    sort_mlp_hidden: bool = True,
) -> ImageEncoderViT:
    """
    Sort SAM encoder channels by a per-channel activation metric (highest → lowest),
    then optionally apply the ToMe/PiToMe patch on top.

    Parameters
    ----------
    encoder : ImageEncoderViT
        SAM image encoder.  Must already be on the target device.
    calib_images : Tensor | None
        Calibration batch (B, 3, H, W) used to estimate the metric.
        If None, a single random image is used (fast but less accurate).
    sort_by : str
        Metric used to rank channels:
          - "std"       : per-channel activation standard deviation (default)
          - "magnitude" : per-channel mean absolute activation value
    inplace : bool
        If True (default) modify the encoder in-place.
        If False, work on a deep copy and leave the original untouched.
    apply_tome : bool
        If True, chain-call the ToMe/PiToMe patch after channel sorting.
    algo : str
        Passed to sam.apply_patch when apply_tome=True.  "tome" or "pitome".
    ratio : float
        Token-reduction ratio for ToMe/PiToMe.
    margin : float
        PiToMe margin (ignored when algo="tome").
    trace_source : bool
        Passed to sam.apply_patch when apply_tome=True.

    Returns
    -------
    encoder : ImageEncoderViT
        The patched encoder.  ``encoder.channel_sort_info`` holds the
        permutation tensors and the per-channel metric values in original order.
    """
    if not inplace:
        encoder = copy.deepcopy(encoder)

    # ── Step 1: calibrate ─────────────────────────────────────────────────────
    ch_metric = _calibrate_channel_metric(encoder, calib_images, sort_by)
    sort_idx  = torch.argsort(ch_metric, descending=True)
    inv_idx   = torch.argsort(sort_idx)

    C = ch_metric.numel()
    print(
        f"[ChannelSort-SAM] sort_by={sort_by!r}  C={C}"
        f"  metric range [{ch_metric.min():.4f}, {ch_metric.max():.4f}]"
        f"  top-3 orig channels: {sort_idx[:3].tolist()}"
    )

    # ── Step 2: permute weights ───────────────────────────────────────────────
    _apply_channel_permutation(encoder, sort_idx)

    # ── Step 3: store permutation info on the encoder ─────────────────────────
    encoder.channel_sort_info = {
        "sort_idx": sort_idx.cpu(),
        "inv_idx":  inv_idx.cpu(),
        "ch_metric": ch_metric.cpu(),
        "sort_by":   sort_by,
    }

    print(f"[ChannelSort-SAM] channel permutation applied.")

    # ── Step 3b (optional): sort MLP hidden dim per block ────────────────────
    if sort_mlp_hidden:
        mlp_metrics      = _calibrate_mlp_hidden_metric(encoder, calib_images, sort_by)
        mlp_sort_indices = [torch.argsort(m, descending=True) for m in mlp_metrics]
        _apply_mlp_hidden_permutation(encoder, mlp_sort_indices)
        encoder.channel_sort_info["mlp_sort_indices"] = [idx.cpu() for idx in mlp_sort_indices]
        print(f"[ChannelSort-SAM] MLP hidden permutation applied ({len(mlp_sort_indices)} blocks).")

    # ── Step 4 (optional): fix HQ decoder compress_vit_feat ──────────────────
    if mask_decoder is not None:
        _apply_hq_decoder_permutation(mask_decoder, sort_idx)
        print(f"[ChannelSort-SAM] mask decoder compress_vit_feat permuted.")

    # ── Step 5 (optional): chain ToMe/PiToMe ─────────────────────────────────
    # if apply_tome:
    #     from .sam import apply_patch as _sam_patch
    #     encoder = _sam_patch(
    #         encoder,
    #         algo=algo,
    #         ratio=ratio,
    #         margin=margin,
    #         trace_source=trace_source,
    #     )

    return encoder
