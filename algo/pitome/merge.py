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

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int=0,
    ratio:float=1.0,    
    class_token: bool = False,
    distill_token: bool = False,
    a_idx=None, 
    b_idx=None,
    scores=None,
) -> Tuple[Callable, Callable]:
    

    
    if a_idx is not None and b_idx is not None and scores is not None:

        B,T,_ = metric.shape
        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        a, b = metric[batch_idx, a_idx, :], metric[batch_idx, b_idx, :]
        node_max, node_idx = scores.max(dim=-1)

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
    else:
        protected = 0
        if class_token:
            protected += 1
        if len(metric.shape) == 2:
            metric = metric[None,...]

        # We can only reduce by a maximum of 50% tokens
        B,T,_ = metric.shape
        
        if r > 0:
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

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if a_idx is not None and b_idx is not None and scores is not None:
            src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]
        else:
            src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)


    return merge, None


def get_merge_func(metric: torch.Tensor, attn_idx:torch.Tensor, ratio: float=1.0, r:int=0,  class_token: bool = True):
    B,T,C = metric.shape
    if r > 0:
        r = min(r, T // 2)
    elif ratio < 1.0:
        kept_number = math.ceil(T*ratio)
    else:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = torch.gather(metric, dim=1, index=attn_idx.unsqueeze(-1).expand(-1, -1, metric.shape[-1]))
        metric = metric/metric.norm(dim=-1, keepdim=True)
        unimportant_tokens_metric = metric[:, kept_number:]
        compress_number = unimportant_tokens_metric.shape[1]
        important_tokens_metric = metric[:,:kept_number]
        similarity = unimportant_tokens_metric@important_tokens_metric.transpose(-1,-2)
        if class_token:
            similarity[..., :, 0] = -math.inf
        _, node_idx = similarity.max(dim=-1)
        dst_idx = node_idx[..., None]
    def merge(x: torch.Tensor, mode="mean", training=False) -> torch.Tensor:
        x= torch.gather(x, dim=1, index=attn_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        src = x[:,kept_number:]
        dst = x[:,:kept_number]
        n, t1, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, compress_number, c), src, reduce=mode) 
        if training:
            return torch.cat([dst, src], dim=1)
        else:
            return dst
    return merge


def pitome_vision(
    metric: torch.Tensor, 
    attn:torch.Tensor=None,
    r:int=0,
    ratio:float=1.0,
    margin:torch.Tensor=0.5,
    class_token: bool = False,
    prune:bool=False
):
    if attn is not None and class_token:
        B,T,C = metric.shape
        if len(attn.shape)==3:
            cls_attn = attn[:,  0, 1:]
        else:
            cls_attn = attn[:, :, 0, 1:]
            cls_attn = cls_attn.mean(dim=1)
        _, idx = torch.sort(cls_attn, descending=True)
        cls_index = torch.zeros((B,1), device=idx.device).long()
        idx = torch.cat((cls_index, idx+1), dim=1)
        merge = get_merge_func(metric, ratio=ratio, class_token=class_token, attn_idx=idx)
        return merge, None
    elif margin >=0.45 and not prune:
        # with torch.no_grad():
        #     if class_token:
        #         metric=metric[:,1:,:]
        #     B,T,C = metric.shape
        #     metric = F.normalize(metric, p=2, dim=-1) 
        #     sigma =  1 - margin 
        #     sim = metric@metric.transpose(-1,-2) 
        #     isolation_score = (2*(torch.exp(-(((1 - sim)/sigma)**2))) - 1).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))
        #     indices =  torch.argsort(isolation_score , descending=True)
        #     a_idx, b_idx = indices[..., ::2], indices[..., 1::2]
        #     scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, b_idx.shape[-1])) 
        #     scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, a_idx.shape[-1], b_idx.shape[-1] ))

        return bipartite_soft_matching(metric, ratio=ratio, class_token=class_token, a_idx=None, b_idx=None, scores=None)
    else:
        # print(metric.shape)
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
            # sim = F.elu((metric@metric.transpose(-1,-2) - margin)/0.01)
            # isolation_score = sim.mean(dim=-1) + sim.sum(-1)
            # indices =  torch.argsort(isolation_score, descending=True)
            sigma = margin 
            sim = metric@metric.transpose(-1,-2) 
            isolation_score = (2*(torch.exp(-(((1 - sim)/sigma)**2))) - 1).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))
            # isolation_score = (2*torch.exp(-(((1 - sim)/sigma)**2))-1).mean(-1) 

            # print(isolation_score.shape)
            indices =  torch.argsort(isolation_score , descending=True)
            merge_idx = indices[..., :2*r]
            protected_idx = indices[..., 2*r:]
        
        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            if class_token:
                x_cls=x[:,0,:].unsqueeze(1)
                x=x[:,1:,:]
            else:
                x_cls = None
            B, T, C = x.shape
            batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
            protected = x[batch_idx, protected_idx, :]

            if not prune:
                a_idx, b_idx = merge_idx[..., :r], merge_idx[..., r:] 
                scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
                scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r ))
                _, dst_idx = scores.max(dim=-1) 
                src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
                dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)
            else:
                dst_idx = merge_idx[...,  r:]
                dst = x[batch_idx,  dst_idx, :]

            if x_cls is not None:
                return torch.cat([x_cls, protected, dst], dim=1)
            else:
                return torch.cat([protected, dst], dim=1)

        if class_token:
            return merge, None 
        return merge, 1- F.softmax(isolation_score, dim=-1) 


def unprotected_pitome_vision(
    metric: torch.Tensor, 
    r:int=0,
    ratio:float=1.0,
    margin:torch.Tensor=0.5,
    class_token: bool = False,
    prune:bool=False
):
    # if margin >=0.45 and not prune:

        # return bipartite_soft_matching(metric, ratio=ratio, class_token=class_token, a_idx=None, b_idx=None, scores=None)
    # else:
        # print(metric.shape)
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
            # sim = F.elu((metric@metric.transpose(-1,-2) - margin)/0.01)
            # isolation_score = sim.mean(dim=-1) + sim.sum(-1)
            # indices =  torch.argsort(isolation_score, descending=True)
            sigma =  1 - margin 
            sim = metric@metric.transpose(-1,-2) 
            isolation_score = (2*(torch.exp(-(((1 - sim)/sigma)**2))) - 1).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))
            # isolation_score = (2*torch.exp(-(((1 - sim)/sigma)**2))-1).mean(-1) 

            # print(isolation_score.shape)
            indices =  torch.argsort(isolation_score , descending=True)
        
        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            if class_token:
                x_cls=x[:,0,:].unsqueeze(1)
                x=x[:,1:,:]
            else:
                x_cls = None
            B, T, C = x.shape
            batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)

            if not prune:
                a_idx, b_idx = indices[..., :r], indices[..., r:] 
                scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, b_idx.shape[-1])) 
                scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, a_idx.shape[-1], b_idx.shape[-1]))
                _, dst_idx = scores.max(dim=-1) 
                src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
                dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)
   
            if x_cls is not None:
                return torch.cat([x_cls, dst], dim=1)
            else:
                return torch.cat([dst], dim=1)

        if class_token:
            return merge, None 
        return merge, 1- F.softmax(isolation_score, dim=-1) 


def pitome_vision_using_attn(
    metric: torch.Tensor, 
    attn:torch.Tensor=None,
    use_cls_attn:bool=False,
    r:int=0,
    ratio:float=1.0,
    margin:torch.Tensor=0.5,
    class_token: bool = False,
    prune:bool=False
):
 
    if margin >=0.45 and not prune:
        return bipartite_soft_matching(metric, ratio=ratio, class_token=class_token, a_idx=None, b_idx=None, scores=None)
    else:
        # print(metric.shape)
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
            # sim = F.elu((metric@metric.transpose(-1,-2) - margin)/0.01)
            # isolation_score = sim.mean(dim=-1) + sim.sum(-1)
            # indices =  torch.argsort(isolation_score, descending=True)
            sigma =  1 - margin 
            sim = metric@metric.transpose(-1,-2) 
            # isolation_score = (2*(torch.exp(-(((1 - sim)/sigma)**2))) - 1).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))
            # isolation_score = (2*torch.exp(-(((1 - sim)/sigma)**2))-1).mean(-1) 

            score = attn[:, :, 1:, 1:].mean(1).mean(-1)
            indices =  torch.argsort(score, descending=True)
            merge_idx = indices[..., :2*r]
            protected_idx = indices[..., 2*r:]
        
        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            if class_token:
                x_cls=x[:,0,:].unsqueeze(1)
                x=x[:,1:,:]
            else:
                x_cls = None
            B, T, C = x.shape
            batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
            protected = x[batch_idx, protected_idx, :]

            if not prune:
                a_idx, b_idx = merge_idx[..., :r], merge_idx[..., r:] 
                scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
                scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r ))
                _, dst_idx = scores.max(dim=-1) 
                src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
                dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)
            else:
                dst_idx = merge_idx[...,  r:]
                dst = x[batch_idx,  dst_idx, :]

            if x_cls is not None:
                return torch.cat([x_cls, protected, dst], dim=1)
            else:
                return torch.cat([protected, dst], dim=1)

        if class_token:
            return merge, None 
        return merge, None 


def pitome_text(
    metric: torch.Tensor, 
    ratio:float=1.0,
    attn:torch.Tensor = None,
    margin:torch.Tensor=0.5,
    class_token: bool = False,
    training:bool=False
):
    if attn is not None and class_token:
        B,T,C = metric.shape
        if len(attn.shape)==3:
            cls_attn = attn[:,  0, 1:]
        else:
            cls_attn = attn[:, :, 0, 1:]
            cls_attn = cls_attn.mean(dim=1)
        _, idx = torch.sort(cls_attn, descending=True)
        cls_index = torch.zeros((B,1), device=idx.device).long()
        idx = torch.cat((cls_index, idx+1), dim=1)
        return get_merge_func(metric, ratio=ratio, class_token=class_token, attn_idx=idx), None

    with torch.no_grad():

        if class_token:
            metric=metric[:,1:,:]

        if len(metric.shape) == 2:
            metric = metric[None,...]
        B,T,C = metric.shape
        r = math.floor(T- T*ratio)
        metric = F.normalize(metric, p=2, dim=-1) 

        batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)
        sim = metric@metric.transpose(-1,-2)
        sigma = 1 - margin 
        isolation_score = (torch.exp(-(((1 - sim)/sigma)**2 * 0.5))).mean(-1) *  1/(sigma*torch.sqrt(torch.tensor(2*torch.pi))) 
        indices =  torch.argsort(isolation_score, descending=True)

    with torch.no_grad():
        merge_idx = indices[..., :2*r]
        protected_idx = indices[..., 2*r:]
        # a_idx, b_idx = merge_idx[..., :r], merge_idx[..., r:]
        # scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
        # scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r ))
        # _, dst_idx = scores.max(dim=-1) 
        # if training: 
            # b_idx = merge_idx[..., 1::2]
        # else:
        b_idx = merge_idx[..., r:]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls=x[:,0,:].unsqueeze(1)
            x=x[:,1:,:]
        else:
            x_cls = None

        B, T, C = x.shape
        protected = x[batch_idx, protected_idx, :]
        # src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
        # dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)
        dst = x[batch_idx,  b_idx, :]

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


