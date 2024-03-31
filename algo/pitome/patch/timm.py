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
from ..merge import merge_source, pitome_vision, merge_wavg, merge_mean



class PiToMeBlockUsingRatio(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def init_margin(self, margin=0.5):
        # self.margin = nn.Parameter(torch.tensor(margin)) 
        self.margin = margin

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric, attn = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        ratio = self._tome_info["ratio"].pop(0)
        if ratio < 1.0:
            merge, isolated_score = pitome_vision(
                ratio=ratio,
                size=self._tome_info["size"],
                attn=attn,
                metric=metric,
                margin=self.margin,
                class_token=self._tome_info["class_token"]
            )

            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )

            if isolated_score is not None and self._tome_info["size"] is not None:
                # weight = self._tome_info["size"] + isolated_score
                x, self._tome_info["size"] = merge_wavg(merge, x, None)
                # x = merge_mean(merge, x)
            else:
                weight = self._tome_info["size"] 
                x, self._tome_info["size"] = merge_wavg(merge, x, weight)

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        # print(x.shape)
        return x 


class PiToMeBlock(Block):
    """
    Modifications:
     - Apply PiToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def init_margin(self, margin=0.5):
        # self.margin = nn.Parameter(torch.tensor(margin))
        self.margin = margin
        self.merge_dropout = nn.Dropout1d(p=0.2)

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self._tome_info["r"].pop(0)
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        if r > 0:
            merge, isolated_score = pitome_vision(
                r=r,
                metric=metric,
                margin=self.margin,
                class_token=self._tome_info["class_token"],
                dropout=self.merge_dropout
            )

            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )

            
            if isolated_score is not None and self._tome_info["size"] is not None:
                weight = isolated_score
                x, self._tome_info["size"] = merge_wavg(merge, x, weight)
                # x = merge_mean(merge, x)
            else:
                weight = self._tome_info["size"] 
                x, self._tome_info["size"] = merge_wavg(merge, x, weight)

            

            return x

class PiToMeAttention(Attention):
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

