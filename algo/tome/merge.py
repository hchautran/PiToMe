# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple
import torch
# from hilbert_utils import get_hilbert_inverse, get_hilbert_order


def do_nothing(x, mode=None):
    return x



def hilbert_matching(
    metric: torch.Tensor,
    ratio: float = 1.0,
    class_token: bool = False, 
):
    if ratio >= 1.0:
        return do_nothing, do_nothing
    


def consecutive_soft_matching(
    metric: torch.Tensor,
    ratio: float = 1.0,
    class_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Merge every group of 4 consecutive tokens into their average.

    Designed for Hilbert-ordered sequences where groups of 4 consecutive tokens
    form a 2×2 spatially adjacent block.  No similarity scoring — fixed stride-4
    average pool forward, broadcast back on unmerge.

    T must be divisible by 4; trailing tokens that don't fill a group are kept.

    Args:
        metric      : [B, T, C] — only used to capture T and device; not scored
        ratio       : unused (kept for API compatibility); merging always reduces
                      by 4× on the grouped tokens
        class_token : reserved, not used
    """
    if ratio >= 1.0:
        return do_nothing, do_nothing

    if len(metric.shape) == 2:
        metric = metric[None]

    B, T, _ = metric.shape
    G = T // 4          # number of complete groups of 4
    tail = T - G * 4    # leftover tokens (0–3) that are kept as-is

    if G == 0:
        return do_nothing, do_nothing

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        n, _, c = x.shape
        grouped = x[:, :G * 4, :].view(n, G, 4, c)   # [B, G, 4, C]
        merged  = grouped.mean(dim=2)                  # [B, G,    C]
        if tail > 0:
            return torch.cat([merged, x[:, G * 4:, :]], dim=1)  # [B, G + tail, C]
        return merged

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        n, _, c = x.shape
        merged_out = x[:, :G, :]                       # [B, G,    C]
        # Broadcast each group average back to 4 positions
        out_grouped = merged_out.unsqueeze(2).expand(n, G, 4, c).reshape(n, G * 4, c)
        if tail > 0:
            return torch.cat([out_grouped, x[:, G:, :]], dim=1)  # [B, T, C]
        return out_grouped

    return merge, unmerge


def bipartite_soft_matching(
    metric: torch.Tensor,
    ratio:float=1.0,    
    class_token: bool = False,
) -> Tuple[Callable, Callable]:
    
    
    protected = 0
    if class_token:
        protected += 1
    if len(metric.shape) == 2:
        metric = metric[None,...]

    # We can only reduce by a maximum of 50% tokens
    T = metric.shape[1]
    
    if ratio < 1.0:
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
        indices = torch.arange(T).to(metric.device)
        a_idx = indices[::2].unsqueeze(0).unsqueeze(-1)  # (1, N/2, 1)
        b_idx = indices[1::2].unsqueeze(0).unsqueeze(-1)  # (1, N/2, 1)

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if len(x.shape) == 2:
            x.unsqueeze_(0)
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        if mode is not None:
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        unm_absolute_indices = torch.gather(
            a_idx.expand(n, a.shape[1], 1), dim=1,
            index=unm_idx.expand(n, -1, 1),
        ).squeeze(-1)
        # (B*num_heads, N_dst)
        dst_absolute_indices = b_idx.squeeze(-1).expand(n, -1)
        absolute_indices = torch.cat([unm_absolute_indices, dst_absolute_indices], dim=1)

        return torch.cat([unm, dst], dim=1), absolute_indices
    
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


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
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
