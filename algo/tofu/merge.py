# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple
import torch
import torch.nn.functional as F


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int=0,
    ratio:float=1.0,    
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    
    protected = 0
    if class_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    T = metric.shape[1]
    
    if r > 0:
        # print(r)
        r = min(r, (T-protected) // 2)
    elif ratio < 1.0:
        r = math.floor(T- T*ratio)
    else:
        return do_nothing, do_nothing


    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, method) -> torch.Tensor:
        n, t1, c = src.shape
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))

        if method == 'prune':
            dst = x[..., 1::2, :]
        elif method == 'average':
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='mean')
        else: 
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            src_norm, dst_norm = torch.norm(src,dim=-1,keepdim=True), torch.norm(dst, dim=-1, keepdim=True) 
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce='mean')
            norms = dst_norm.scatter_reduce(-2, dst_idx, src_norm, reduce='amax')
            dst = dst/dst_norm * norms

        return torch.cat([unm, dst], dim=1)
    return merge, None


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    return x, None 



def merge_mean(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    size = torch.ones_like(x[..., 0, None])

    x = merge(x, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    return x, size 



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
