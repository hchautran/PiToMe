# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def do_nothing(x, mode=None):
    return x



def crossget(
    metric: torch.Tensor, 
    r:int=0,
    ratio:float=1.0,
    class_token: bool = False,
):
        with torch.no_grad():
            if class_token:
                metric=metric[:,1:,:]
            B,T,C = metric.shape
            if r > 0:
                r = min(r, T // 2)
            elif ratio < 1.0:
                r = math.floor(T- T*ratio)
            else:
                return do_nothing, do_nothing
            metric = F.normalize(metric, p=2, dim=-1) 


            D = metric@metric.transpose(-1,-2) - torch.eye(T, device=metric.device)[None, ...] 
            A_s = torch.argsort(torch.max(D, dim=-1, keepdim=True)[0], dim=-2, descending=True)
            A_d = torch.argsort(torch.max(D, dim=-2, keepdim=True)[0], dim=-1, descending=True)

            D = D.gather(dim=-2, index=A_s.expand(B, T, T)) 
            D = D.gather(dim=-1, index=A_d.expand(B, T, T))

            D = D -  1e9*torch.tril((torch.ones_like(D)))

            A = torch.argsort(torch.max(D, dim=-1)[0], dim=-1, descending=True)[..., None]
            unm_idx = A[..., r:, :]  # Unmerged Tokens
            src_idx = A[..., :r, :]  # Merged Tokens
            scores = D.gather(dim=-1, index=src_idx.expand(B, r, D.shape[-1])) 
            dst_idx = scores.argmax(dim=-1)[..., None]
        
        
        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            if class_token:
                x_cls=x[:,0,:].unsqueeze(1)
                x=x[:,1:,:]
            else:
                x_cls = None
            B, T, C = x.shape

            dst = x.gather(dim=-2, index=unm_idx.expand(B, T - r, C))
            src = x.gather(dim=-2, index=src_idx.expand(B, r, C))
            dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

   
            if x_cls is not None:
                return torch.cat([x_cls, dst], dim=1)
            else:
                return torch.cat([dst], dim=1)

        if class_token:
            return merge, None 
        return merge, None 




def merge_mean(
    merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    x = merge(x, mode="mean")
    return x

def prune(
    merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    x = merge(x, mode="prune")
    return x


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x*size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    return x, None 


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source

def merge_attention_mask(
    merge, attention_mask: torch.Tensor
): 
    attention_mask = merge(attention_mask, mode="amax")
    return attention_mask 


