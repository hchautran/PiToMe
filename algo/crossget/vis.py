# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.


def generate_colormap(N: int, attention_score: torch.Tensor, seed: int = 0) -> List[Tuple[float, float, float]]:
  """
  Generates a colormap with N elements, with a bolder blue base and lightness adjusted based on attention scores.

  Args:
      N: Number of colors to generate.
      attention_score: A torch.Tensor representing the attention scores.
          This will be used to modulate the lightness of the blue color.
      seed: An optional integer seed for reproducibility.

  Returns:
      A list of tuples representing RGB color values (0.0 to 1.0).
  """


  def adjust_lightness(attention_value):
    normalized_attention = (attention_value - attention_score.min()) / (attention_score.max() - attention_score.min())
    lightness_adjustment = 0.3 * normalized_attention  # Adjust factor for lightness range
    base = (0.2, 0.4, 0.8)
    adjusted_color = [base[0] + lightness_adjustment, base[1], base[2] + lightness_adjustment]
    return tuple(max(0.0, min(1.0, val)) for val in adjusted_color)

  colormap = [adjust_lightness(attention_value) for attention_value in attention_score.flatten().tolist()]

  return colormap


def make_visualization(
    img: Image, source: torch.Tensor, attention_score:torch.Tensor, patch_size: int = 16, class_token: bool = True
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     -

    Returns:
     - A PIL image the same size as the input.
    """

    img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    if class_token:
        source = source[:, :, 1:]
    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1
    print('num_group',num_groups)

    cmap = generate_colormap(num_groups, attention_score)
    vis_img = 0

    for i in range(num_groups):
        print('vis', vis.shape)
        mask = (vis == i).float().view(1, 1, ph, pw)
        print('mask', mask.shape)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        print('color', color.shape)
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded
        print('mask edge', mask_edge.shape)

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img
