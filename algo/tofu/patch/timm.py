# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


import torch
from typing import Tuple
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from ..merge import bipartite_soft_matching, merge_source


class ToFuBlock(Block):
    """
    Modifications:
     - Apply ToFu between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def init_strategy(self, strategy='mean'):
        # self.margin = nn.Parameter(torch.tensor(margin)) 
        self.strategy = strategy 


    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tofu_info["size"] if self._tofu_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tofu_info["r"].pop(0)
        if r > 0:
            # Apply ToFu here
            merge, _ = bipartite_soft_matching(
                metric=metric,
                r=r,
                class_token=self._tofu_info["class_token"],
                # distill_token=self._tofu_info["distill_token"],
            )
            if self._tofu_info["trace_source"]:
                self._tofu_info["source"] = merge_source(
                    merge, x, self._tofu_info["source"]
                )
            x, self._tofu_info["size"] = merge(x, mode=self.strategy)

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToFuBlockUsingRatio(Block):
    """
    Modifications:
     - Apply ToFu between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def init_strategy(self, strategy='mean'):
        # self.margin = nn.Parameter(torch.tensor(margin)) 
        self.strategy = strategy 


    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tofu_info["size"] if self._tofu_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)
        ratio = self._tofu_info["ratio"].pop(0)
        if ratio < 1.0:
            # Apply ToFu here
            merge, _ = bipartite_soft_matching(
                metric=metric,
                ratio=ratio,
                class_token=self._tofu_info["class_token"],
                distill_token=self._tofu_info["distill_token"],
            )

            if self._tofu_info["trace_source"]:
                self._tofu_info["source"] = merge_source(
                    merge, x, self._tofu_info["source"]
                )

            x = merge(x, mode=self.strategy)


        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToFuAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
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
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)

