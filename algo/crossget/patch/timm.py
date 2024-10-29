# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Block
from ..merge import merge_source, crossget, merge_wavg, merge_mean



class CrossGetBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_size = self._info["size"] if self._info["prop_attn"] else None
        x_attn, metric, attn = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        ratio = self._info["ratio"].pop(0)
        if ratio < 1.0:
            merge = crossget(
                ratio=ratio,
                metric=metric,
                class_token=self._info["class_token"]
            )

            if self._info["trace_source"]:
                self._info["source"] = merge_source(
                    merge, x, self._info["source"]
                )

            weight = self._info["size"] 
            x, self._info["size"] = merge_wavg(merge, x, weight)

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x 


class CrossGetAttention(Attention):
    """
    Modifications:
    - Apply proportional attention
    - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, isolation_score: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if isolation_score is not None:
            attn = attn +  isolation_score.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(attn.shape)

        return x, k.mean(1), attn

