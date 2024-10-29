# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
import torch
from typing import Callable, List, Tuple, Union


def mctf(x, info,attn=None, ratio=1.0, r=0, class_token=False):

    merge, _ = bipartite_soft_matching(
        x, attn = attn, r = r,ratio=ratio,
        class_token = class_token,
        tau_sim       = info["tau_sim"],
        tau_info      = info["tau_info"],
        tau_size      = info["tau_size"],
        size          = info["size"],
        bidirection   = info["bidirection"]
    )

    if info["trace_source"]:
        info["source"] = merge_source(
            merge, x, info["source"]
        )

    x, info["size"], attn_n = merge_wavg(merge, x, info["size"], attn, one_step_ahead=info["one_step_ahead"], pooling_type=info["pooling_type"])
    return x, attn_n

def bipartite_soft_matching(
        metric : torch.Tensor,
        class_token:   bool,
        tau_sim:       int=1,
        tau_info:      int=1,
        tau_size:      int=1,
        ratio:         float=1.0,
        r:             int=0,
        bidirection:  int=True,
        size   : torch.Tensor=None,
        attn   : torch.Tensor=None,
) -> Tuple[Callable, Callable]:

    
    protected = 0
    if class_token:
        protected += 1
    if len(metric.shape) == 2:
        metric = metric[None,...]

    T = metric.shape[1]

   
    if ratio < 1.0:
        r = math.floor(T- T*ratio)
    else:
        return do_nothing

    if bidirection:
        r1, r2 = r // 2, r - r // 2
    else:
        r1, r2 = r, 0

    B, T, _ = metric.shape  # (4(B), 197(T), 384(4))
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)  # (12, 197, 64)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]  # (12, 99, 64), (12, 98, 64)

        if tau_sim:
            W_sim = a @ b.transpose(-1, -2)
            W_sim = ((W_sim + 1) / 2) ** (1 / tau_sim)
        else:
            W_sim = torch.ones((a.shape[0], a.shape[1], b.shape[1]), device=a.device)

        if tau_info > 0 and attn is not None:
            attn = 1 / attn.mean(dim=[1, 2])
            attn = attn / attn.max(1, keepdim=True)[0]  # (4(B), 197(T))
            attn_a, attn_b = attn[..., ::2, None], attn[..., 1::2, None].transpose(1, 2)

            W_info = (attn_a * attn_b) ** (1 / tau_info)
        else:
            W_info = 1

        if tau_size and size is not None:
            size = 1 / size
            size = size / size.max(1, keepdim=True)[0]  # (4(B), 197(T), 1)
            size_a, size_b = size[..., ::2, :], size[..., 1::2, :].transpose(1, 2)

            W_size = (size_a * size_b) ** (1 / tau_size)
        else:
            W_size = 1

        scores = W_sim * W_info * W_size

        if class_token:
            scores[..., 0, :] = -math.inf

        n, t1, t2 = scores.shape

        node_max, node_idx = scores.max(dim=-1)  # (12, 99), (12, 99)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # (12, 99, 1)
        unm_idx = edge_idx[..., r1:, :]  # Unmerged Tokens (12, 83, 1)
        src_idx = edge_idx[..., :r1, :]  # Merged Tokens   (12, 16, 1)

        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # (12, 16, 1)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

        #### bidirection:
        new_scores = scores.gather(dim=-2, index=unm_idx.repeat(1, 1, t2)).transpose(1, 2)  # (12, 98, 91)

        node_max2, node_idx2 = new_scores.max(dim=-1)  # (12, 98), (12, 98)
        edge_max2, edge_idx2 = node_max2.sort(dim=-1, descending=True)  # (12, 98, 1)
        edge_idx2 = edge_idx2[..., None]
        unm_idx2 = edge_idx2[..., r2:, :]  # Unmerged Tokens (12, 90, 1)
        src_idx2 = edge_idx2[..., :r2, :]  # Merged Tokens   (12, 8, 1)
        dst_idx2 = node_idx2[..., None].gather(dim=-2, index=src_idx2)  # (12, 8, 1)

    def dim_match(src, dim=1, dim_num=5):
        while len(src.shape) < dim_num:
            src = src.unsqueeze(dim)
        return src

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # ori_dtype = x.dtype
        # x = x.to(dtype=torch.float32)
        src, dst = x[..., ::2, :], x[..., 1::2, :]  # (12, 99, 197), (12, 98, 197)

        b, mid, c = src.shape[0], src.shape[1:-2], src.shape[-1]
        dim_num = len(src.shape)
        unm_idx_ = dim_match(unm_idx, dim=1, dim_num=dim_num)
        src_idx_ = dim_match(src_idx, dim=1, dim_num=dim_num)
        dst_idx_ = dim_match(dst_idx, dim=1, dim_num=dim_num)
        unm = src.gather(dim=-2, index=unm_idx_.expand(b, *mid, t1 - r1, c))  # (..., 91, 197)
        src = src.gather(dim=-2, index=src_idx_.expand(b, *mid, r1, c))  # (..., 8, 197)
        dst = dst.scatter_reduce(-2, dst_idx_.expand(b, *mid, r1, c), src, reduce=mode)  # (..., 98, 197)

        ### bidirection 
        unm_idx2_ = dim_match(unm_idx2, dim=1, dim_num=dim_num)
        src_idx2_ = dim_match(src_idx2, dim=1, dim_num=dim_num)
        dst_idx2_ = dim_match(dst_idx2, dim=1, dim_num=dim_num)
        src2, dst2 = dst, unm
        unm2 = src2.gather(dim=-2, index=unm_idx2_.expand(b, *mid, t2 - r2, c))  # (12, 90, 197)
        src2 = src2.gather(dim=-2, index=src_idx2_.expand(b, *mid, r2, c))  # (12, 8, 197)
        dst2 = dst2.scatter_reduce(-2, dst_idx2_.expand(b, *mid, r2, c), src2, reduce=mode)  # (12, 91, 197)
        x = torch.cat([dst2, unm2], dim=-2)  # (12(B), 2(T_1) + 3(T_2), 384(D))
        return x

    return merge


def merge_wavg(
        merge: Callable, x: torch.Tensor, attn:torch.Tensor, size: torch.Tensor = None, one_step_ahead=0,
    ):
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    attn_n = False
    if size is None:  # WAVER
        size = torch.ones_like(x[..., 0, None])  # (12, 197, 1)

    size_max = size.amax(dim=-2, keepdim=True)
    with torch.no_grad():
        if len(attn.shape) == 4: 
            attn_m = attn.mean(dim=[1, 2]).unsqueeze(-1)
        else:
            attn_m = attn.mean(dim=[1]).unsqueeze(-1)
        norm = merge(attn_m * (size / size_max), mode="sum") # (1, 197, 1)

    if one_step_ahead:
        attn_n = merge(attn[..., None], mode="sum").squeeze(-1)  # (6(B), 6(H), 197(T_Q), 181(T_K))
        attn_n = merge(attn_n * attn_m[:, None] * (size / size_max)[:, None], mode="sum").squeeze(-1) # (1(B), 6(H), 181(T_Q), 181(T_K))
        attn_n = attn_n / norm[:, None]

    x = merge(x * attn_m * (size / size_max), mode="sum")
    with torch.no_grad():
        size = merge(size, mode="sum")
    if isinstance(x, tuple):
        x = x[0] / norm[0]
    else:
        x = x / norm


    return x, size, attn_n


# def merge_wavg(
#     merge: Callable, x: torch.Tensor, size: torch.Tensor = None
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Applies the merge function by taking a weighted average based on token size.
#     Returns the merged tensor and the new token sizes.
#     """
#     if size is None:
#         size = torch.ones_like(x[..., 0, None])
#     # print(x.shape)

#     x = merge(x, mode="sum")
#     size = merge(size, mode="sum")
#     if isinstance(x, tuple):
#         x = x[0] / size[0]
#     else:
#         x = x / size

#     return x, None 

@torch.no_grad()
def merge_source(
        merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)  # (12, 197, 197)

    source = merge(source, mode="amax")
    return source


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


def do_nothing(x, mode=None):
    return x, None

    

def merge_attention_mask(
    merge, attention_mask: torch.Tensor
): 

    attention_mask = merge(attention_mask, mode="amax")
    return attention_mask 
