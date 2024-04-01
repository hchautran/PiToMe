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
) -> Tuple[Callable, Callable]:
    
    protected = 0
    if class_token:
        protected += 1
    if len(metric.shape) == 2:
        metric = metric[None,...]

    # We can only reduce by a maximum of 50% tokens
    T = metric.shape[1]
    
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
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)
    
    return merge, None

def pitome_vision(
    metric: torch.Tensor, 
    size:torch.Tensor=None,
    attn:torch.Tensor=None,
    r:int=0,
    ratio:float=1.0,
    margin:torch.Tensor=0.5,
    class_token: bool = False,
    prune:bool=False
    # dropout:nn.Module=None
):

    if margin >= 0.45 and not prune:
        return bipartite_soft_matching(metric, ratio=ratio, class_token=class_token)

    with torch.no_grad():
        if class_token:
            if attn is not None:
                attn = attn[:,:, 0, 1:]
            
            metric=metric[:,1:,:]
            if size is not None:
                size=size.squeeze()[:, 1:]
        else:
            attn = attn 

        B,T,C = metric.shape
        if r > 0:
            r = min(r, T // 2)
        elif ratio < 1.0:
            r = math.floor(T- T*ratio)
        else:
            return do_nothing, do_nothing
        metric_normed = F.normalize(metric, p=2, dim=-1) 
   
    with torch.no_grad():
        sim = metric_normed@metric_normed.transpose(-1,-2) 
        sim = F.elu((sim - margin)/0.1)
        isolation_score = sim.mean(dim=-1) 
        indices =  torch.argsort(isolation_score, descending=True)

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
            dst_idx = merge_idx[...,  1::2]
            dst = x[batch_idx,  dst_idx, :]

        if x_cls is not None:
            return torch.cat([x_cls, protected, dst], dim=1)
        else:
            return torch.cat([protected, dst], dim=1)
    
    


    if class_token:
        return merge, None 
    return merge, 1- F.softmax(isolation_score, dim=-1) 


def pitome_text(
    metric: torch.Tensor, 
    ratio:float=1.0,
    attn:torch.Tensor = None,
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
    if attn is not None:
        isolation_score = attn.mean(dim=1)
        indices =  torch.argsort(isolation_score, descending=False)
    else:
        isolation_score = sim.mean(dim=-1) + sim.sum(-1)
        indices =  torch.argsort(isolation_score, descending=True)

    with torch.no_grad():
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

    
def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """

    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v, dim=1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.view_as_complex(V)

    v = torch.fft.ifft(V, dim=1).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)



def dc_transform(x, ratio:float=None, k:int=None, class_token:bool=True ):
    # cufft doesn't accept fp16
    # dct along T dimension
    if class_token:
        x_cls = x[:,0,:].unsqueeze_(1)
        x = x[:,1:,:]
    x = x.type(torch.float32).permute(1,0,2)
    x_dct = dct(x.transpose(0,2), norm='ortho').transpose(0,2)
    T, B, C = x_dct.size()

    # feel free to play with any method here
    if ratio is not None: 
        x_dct = x_dct[:math.ceil(T * ratio), :, :]
    else:
        x_dct = x_dct[:T-k, :, :]

    x = idct(x_dct.transpose(0,2), norm='ortho').transpose(0,2).type(torch.half).permute(1,0,2)
    
    if class_token:
        return torch.cat([x_cls, x], dim=1)
    return x
    
