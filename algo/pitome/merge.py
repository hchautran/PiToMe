# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple
import torch
import torch.nn.functional as F
from geoopt import Lorentz


def do_nothing(x, mode=None):
    return x

def get_bsm_merge(
    node_max:torch.Tensor,
    node_idx:torch.Tensor,
    r:int,
    class_token=False,
    a_idx:torch.Tensor=None,
    b_idx:torch.Tensor=None,
) -> Tuple[Callable, Callable]:

    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
    src_idx = edge_idx[..., :r, :]  # Merged Tokens
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls=x[:,0,:].unsqueeze(1)
            x=x[:,1:,:]
        else:
            x_cls = None
        if a_idx is not None and b_idx is not None:
            B, _, _ = x.shape
            batch_idx = torch.arange(B).unsqueeze_(1).to(x.device)
            src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]
        else:
            src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if x_cls is not None:
            return torch.cat([x_cls, unm, dst], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    return merge, None

def pitome_vision(
    metric: torch.Tensor, 

    r:int=0,
    ratio:float=1.0,
    margin:torch.Tensor=0.5,
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

    sim = metric@metric.transpose(-1,-2) - torch.eye(T)[None,...].to(metric.device)

    with torch.no_grad():
        if margin >= 0.45:
            isolation_score = sim.mean(dim=-1) 
            indices =  torch.argsort(isolation_score, descending=True)
            a_idx, b_idx = indices[..., ::2], indices[..., 1::2] 
            scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, b_idx.shape[-1])) 
            scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, a_idx.shape[-1], b_idx.shape[-1]))
            node_max, node_idx = scores.max(dim=-1)
            return get_bsm_merge(node_max=node_max, node_idx=node_idx, r=r, class_token=class_token, a_idx=a_idx, b_idx=b_idx)
        else:
            sim = F.elu((sim - margin)/0.1) 
            isolation_score = sim.mean(dim=-1) 
            indices =  torch.argsort(isolation_score, descending=True)
            merge_idx = indices[..., :2*r]
            protected_idx = indices[..., 2*r:]
            a_idx, b_idx = merge_idx[..., :r], merge_idx[..., r:] 
            scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
            scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r ))
            _, dst_idx = scores.max(dim=-1) 
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls=x[:,0,:].unsqueeze(1)
            x=x[:,1:,:]
        else:
            x_cls = None
        B, T, C = x.shape
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        protected = x[batch_idx, protected_idx, :]
        src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
        dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)
        if x_cls is not None:
            return torch.cat([x_cls, protected, dst], dim=1)
        else:
            return torch.cat([protected, dst], dim=1)

    return merge, None 

def pitome_text(
    metric: torch.Tensor, 
    ratio:float=1.0,
    margin:torch.Tensor=0.5,
    class_token: bool = False,
):
    with torch.no_grad():
        if class_token:
            metric=metric[:,1:,:]
        if len(metric.shape) == 2:
            metric = metric[None,...]
        # B,H,T,C = metric.shape
        B,T,C = metric.shape
        r = math.floor(T- T*ratio)
        metric = F.normalize(metric, p=2, dim=-1) 
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)

    sim = F.elu((metric@metric.transpose(-1,-2) - margin)/0.01)
    isolation_score = sim.mean(dim=-1) + sim.sum(-1)
    # print(isolation_score.shape)

    with torch.no_grad():
        indices =  torch.argsort(isolation_score, descending=True)
        merge_idx = indices[..., :2*r]
        protected_idx = indices[..., 2*r:]
        a_idx, b_idx = merge_idx[..., :r], merge_idx[..., r:]
        scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
        scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r ))
        _, dst_idx = scores.max(dim=-1) 
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:

        if class_token:
            x_cls=x[:,0,:].unsqueeze(1)
            x=x[:,1:,:]
        else:
            x_cls = None

        B, T, C = x.shape
        protected = x[batch_idx, protected_idx, :]
        src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
        dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)

        if x_cls is not None:
            return torch.cat([x_cls, protected, dst], dim=1)
        else:
            return torch.cat([protected, dst], dim=1)

    isolation_score = 1 - F.softmax(isolation_score, dim=-1) 

    if class_token:
        return merge, torch.cat([torch.ones(B, 1).to(metric.device), isolation_score], dim=-1)[..., None]
    return merge, isolation_score[..., None] 


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
