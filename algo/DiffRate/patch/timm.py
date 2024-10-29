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
from timm.models.vision_transformer import Attention, Block, VisionTransformer
import torch.nn as nn
# import DiffRate.ddp as ddp
from ..ddp import DiffRate
from ..merge import get_merge_func


class DiffRateBlock(Block):
    """
    Modifications:
     - Apply DiffRate between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def introduce_diffrate(self,patch_number, prune_granularity, merge_granularity):
        self.prune_ddp = DiffRate(patch_number,prune_granularity)
        self.merge_ddp = DiffRate(patch_number,merge_granularity)
        
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        size = self._info["size"]
        mask = self._info["mask"]
        x_attn, attn = self.attn(self.norm1(x), size, mask=self._info["mask"])
        x = x + self._drop_path1(x_attn)

        # importance metric
        cls_attn = attn[:, :, 0, 1:]
        cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
        _, idx = torch.sort(cls_attn, descending=True)
        cls_index = torch.zeros((B,1), device=idx.device).long()
        idx = torch.cat((cls_index, idx+1), dim=1)
        
        # sorting
        x = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        self._info["size"] = torch.gather(self._info["size"], dim=1, index=idx.unsqueeze(-1))
        mask = torch.gather( mask, dim=1, index=idx)
        if self._info["trace_source"]:
            self._info["source"] = torch.gather(self._info["source"], dim=1, index=idx.unsqueeze(-1).expand(-1, -1, self._info["source"].shape[-1]))

        
        if self.training:
            # pruning, pruning only needs to generate masks during training
            last_token_number = mask[0].sum().int()
            prune_kept_num = self.prune_ddp.update_kept_token_number()      # expected prune compression rate, has gradiet
            self._info["prune_kept_num"].append(prune_kept_num)
            if prune_kept_num < last_token_number:        # make sure the kept token number is a decreasing sequence
                prune_mask = self.prune_ddp.get_token_mask(last_token_number)
                mask = mask * prune_mask.expand(B, -1)

            mid_token_number = min(last_token_number, int(prune_kept_num)) # token number after pruning
            # merging
            merge_kept_num = self.merge_ddp.update_kept_token_number()
            self._info["merge_kept_num"].append(merge_kept_num)

            if merge_kept_num < mid_token_number:
                merge_mask = self.merge_ddp.get_token_mask(mid_token_number)
                x_compressed, size_compressed = x[:, mid_token_number:], self._info["size"][:,mid_token_number:]
                merge_func, node_max = get_merge_func(metric=x[:, :mid_token_number].detach(), kept_number=int(merge_kept_num))
                x = merge_func(x[:,:mid_token_number],  mode="mean", training=True)
                # optimize proportional attention in ToMe by considering similarity
                size = torch.cat((self._info["size"][:, :int(merge_kept_num)],self._info["size"][:, int(merge_kept_num):mid_token_number]*node_max[..., None]),dim=1)
                size = size.clamp(1)
                size = merge_func(size,  mode="sum", training=True)
                x = torch.cat([x, x_compressed], dim=1)
                self._info["size"] = torch.cat([size, size_compressed], dim=1)
                mask = mask * merge_mask

            self._info["mask"] = mask
            x = x + self._drop_path2(self.mlp(self.norm2(x)))
            
        else:
            # pruning
            prune_kept_num = self.prune_ddp.kept_token_number
            x = x[:, :prune_kept_num]
            self._info["size"] = self._info["size"][:, :prune_kept_num]
            if self._info["trace_source"]:
                self._info["source"] = self._info["source"][:, :prune_kept_num]
                
            
            # merging
            merge_kept_num = self.merge_ddp.kept_token_number
            if merge_kept_num < prune_kept_num:
                merge,node_max = get_merge_func(x.detach(), kept_number=merge_kept_num)
                x = merge(x,mode='mean')
                # optimize proportional attention in ToMe by considering similarity, this is benefit to the accuracy of off-the-shelf model.
                self._info["size"] = torch.cat((self._info["size"][:, :merge_kept_num],self._info["size"][:, merge_kept_num:]*node_max[..., None] ),dim=1)
                self._info["size"] = merge(self._info["size"], mode='sum')
                if self._info["trace_source"]:
                    self._info["source"] = merge(self._info["source"], mode="amax")

            x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x
                

class DiffRateAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None, mask: torch.Tensor = None
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
        
        if self.training:
            attn = self.softmax_with_policy(attn, mask)
        else:
            attn = attn.softmax(dim=-1)
            
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # Return attention map as well here
        return x, attn


