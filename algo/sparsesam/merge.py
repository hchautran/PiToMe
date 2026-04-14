# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple
import torch
from .hilbert_utils import get_hilbert_inverse, get_hilbert_order
import torch.nn.functional as F
order_1 = 2
order_2 = 4
order_3 = 8 
order_4 = 16 
orders = [order_1, order_2, order_3, order_4]

global_perm= get_hilbert_order(64, 64) 
global_inv = get_hilbert_inverse(64, 64) 

local_perm = get_hilbert_order(14, 14) 
local_inv = get_hilbert_inverse(14, 14) 

def do_nothing(x, mode=None):
    return x


def compute_grad(metric:torch.Tensor):
    sobel_x = torch.tensor([[[[1.,  0., -1.],
                           [2.,  0., -2.],
                           [1.,  0., -1.]]]])  # horizontal edges

    sobel_y = torch.tensor([[[[1.,  2.,  1.],
                            [0.,  0.,  0.],
                            [-1., -2., -1.]]]])  # vertical edges
    sobel_x = sobel_x.view(1, 1, 3, 3).expand(-1, metric.shape[1], -1, -1)
    sobel_y = sobel_y.view(1, 1, 3, 3).expand(-1, metric.shape[1], -1, -1)

    # Apply convolution
    x_pad = F.pad(metric, (1, 1, 1, 1), mode='reflect')
    edge_x = F.conv2d(x_pad, sobel_x, padding=0)
    edge_y = F.conv2d(x_pad, sobel_y, padding=0)

    # Gradient magnitude
    magnitude = torch.sqrt(edge_x**2 + edge_y**2)
    return magnitude

def get_hilbert_orders(
    size=64,
    order=0
):
    perm     = get_hilbert_order(size, size)     # [T]  raster→Hilbert
    inv_perm = get_hilbert_inverse(size, size)   # [T]  Hilbert→raster
    
    
    


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
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        unm_absolute_indices = torch.gather(a_idx.expand(n, a.shape[1], 1), dim=1, index=unm_idx).squeeze(-1)
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


