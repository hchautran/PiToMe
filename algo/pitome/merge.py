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

def pitome(
    metric=None,
    class_token: bool = False,
    indices:torch.Tensor=None, 
    scores:torch.Tensor=None,
    r:int=None
) -> Tuple[Callable, Callable]:
    B, T, T = scores.shape
    merge_idx = indices[..., :2*r]
    protected_idx = indices[..., 2*r:]
    a_idx, b_idx = merge_idx[..., ::2], merge_idx[..., 1::2] 

    # get similarity scores between mergeable tokens
    scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
    scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r ))
    _, dst_idx = scores.max(dim=-1) 
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls=x[:,0,:].unsqueeze(1)
            x=x[:,1:,:]

        B, T, C = x.shape
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        protected = x[batch_idx, protected_idx, :]
        src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]

        dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)

        if class_token:
            return torch.cat([x_cls, protected, dst], dim=1)
        return torch.cat([protected, dst], dim=1)

    return merge


def pitome_bsm(
    metric=None,
    class_token: bool = False,
    indices:torch.Tensor=None,
    scores:torch.Tensor=None,
    r:int=None
) -> Tuple[Callable, Callable]:

    with torch.no_grad():
        B, T, T = scores.shape
        a_idx, b_idx = indices[..., ::2], indices[..., 1::2] 
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        scores = scores.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, b_idx.shape[-1])) 
        scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, a_idx.shape[-1], b_idx.shape[-1]))
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls=x[:,0,:].unsqueeze(1)
            x=x[:,1:,:]

        src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if class_token:
            return torch.cat([x_cls, unm, dst], dim=1)
        return torch.cat([unm, dst], dim=1)
    return merge

def pitome_vision(
    metric: torch.Tensor, 
    ratio:float=1.0,
    margin:torch.Tensor=0.5,
    class_token: bool = False,
    alpha=1.0,
    use_bsm_pitome=False
):
    with torch.no_grad():
        if class_token:
            metric=metric[:,1:,:]
        B,T,C = metric.shape
        if ratio < 1.0:
            r = math.floor(T- T*ratio)
        else:
            return do_nothing, do_nothing

        # calculate energy score  
        metric = F.normalize(metric, p=2, dim=-1) 
        sim = metric@metric.transpose(-1,-2)
        energy_score = F.elu((sim - margin), alpha=alpha).mean(dim=-1)
        indices =  torch.argsort(energy_score, descending=True)
        # seperate protected token and mergeable tokens  
        if use_bsm_pitome:
            return pitome_bsm(metric=metric, class_token=class_token, indices=indices, scores=sim, r=r)
        else:
            return pitome(metric=metric, class_token=class_token, indices=indices, scores=sim, r=r)



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
        B,T,C = metric.shape
        r = math.floor(T- T*ratio)
        metric = F.normalize(metric, p=2, dim=-1) 
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        # calculate energy score for in this implementation we use gaussian kernel which show better performance than the equation (4) in the paper 
        sim = metric@metric.transpose(-1,-2)
        # sim = F.elu((metric@metric.transpose(-1,-2) - margin)/0.01, alpha=alpha)
        sigma = 1 - margin 
        energy_score = (torch.exp(-(((1 - sim)/sigma)**2 * 0.5))).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi))) 
        indices =  torch.argsort(energy_score , descending=True)
        merge_idx = indices[..., :2*r]
        protected_idx = indices[..., 2*r:]
        # Also instead of using odd and even indices since we choose to split based on higher and lower energy set which show significant better performance 
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

        if class_token:
            return torch.cat([x_cls, protected, dst], dim=1)
        return torch.cat([protected, dst], dim=1)

    return merge


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


