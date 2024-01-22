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
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    T = metric.shape[1]
    if r >0:
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
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge



def pitome(
    x: torch.Tensor, 
    ratio:float=1.0,
    margin:float=0.5,
    class_token: bool = False,
):
    if margin < 0.6:
        return bipartite_soft_matching(x)

    with torch.no_grad():
        if class_token:
            x=x[:,1:,:]
        B,T,C = x.shape
        if ratio < 1.0:
            r = math.floor(T- T*ratio)
        else:
            return do_nothing, do_nothing
        batch_idx = torch.arange(B).unsqueeze_(1).to(x.device)
        x = F.normalize(x, p=2, dim=-1)
        sim =x@x.transpose(-1,-2) - margin
        sim  = torch.where(sim > 0, sim, -margin)
        isolation_score = sim.mean(dim=-2)
        indices =  torch.argsort(isolation_score, descending=True)
        min_indices = indices[..., :2*r]
        protected_idx = indices[..., 2*r:]
        a_idx, b_idx = min_indices[..., ::2], min_indices[..., 1::2]
        a, b = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
        scores = a@b.transpose(-1,-2) 
        _, dst_idx = scores.max(dim=-1) 
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if class_token:
            x_cls=x[:,0,:].unsqueeze(1)
            x=x[:,1:,:]
        else:
            x_cls = None
        B, T, C = x.shape

        protected = x[batch_idx, protected_idx,:] 
        src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :]

        dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)
        if x_cls is not None:
            return torch.cat([x_cls, protected, dst], dim=1)
        else:
            return torch.cat([protected, dst], dim=1)

    return merge, isolation_score

def merge_mean(
    merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    x = merge(x, mode="mean")
    return x




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
