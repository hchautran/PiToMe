from typing import Callable, Tuple
import torch
import torch.nn.functional as F
import math


track_merge_and_unmerge_funcs = []
def make_merge_info(grad, n_tokens, a_idx, b_idx, protected_idx, merge, unmerge):
    info = {
        'grad': grad,
        'n_tokens': n_tokens,
        'a_idx': a_idx,
        'b_idx': b_idx,
        'protected_idx': protected_idx,
        'merge': merge,
        'unmerge': unmerge
    }
    return info

# with stride partition, so larger merge rate

def get_central_difference_gradient(x: torch.Tensor) -> torch.Tensor:
    '''
    Compute the gradient magnitude for token embeddings using central difference approximation.

    Args:
        x: token embeddings of shape (B, H, W, C)

    Returns:
        grad_mag(torch.Tensor): Gradient magnitude of embeddings (B, H, W)
    '''
    padded = F.pad(x, (0, 0, 1, 1, 1, 1), mode='replicate') # (B, H+2, W+2)
    diff_x = (padded[:, 1:-1, 2:, :] - padded[:, 1:-1, :-2, :])/2.0 # Difference along width # (B, H, W, C)
    diff_y = (padded[:, 2:, 1:-1, :] - padded[:, :-2, 1:-1, :])/2.0 # Difference along height # (B, H, W, C)
    grad_mag = torch.sqrt((diff_x ** 2).sum(dim=-1) + (diff_y ** 2).sum(dim=-1)) # (B, H, W)

    return grad_mag

def get_sobel_gradient(x: torch.Tensor) -> torch.Tensor:
    '''
    Compute the gradient magnitude for token embeddings using a Sobel operator.

    Args:
        x: token embeddings of shape (B, H, W, C)

    Returns:
        grad_mag(torch.Tensor): Gradient magnitude of embeddings (B, H, W)
    '''
    x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
    B, C, H, W = x.shape

    sobel_x = torch.tensor([[-1., 0, 1],
                            [-2., 0, 2],
                            [-1., 0, 1]], device=x.device, dtype=x.dtype)

    sobel_y = torch.tensor([[-1., -2, 1],
                            [0, 0, 0],
                            [1., 2, 1]], device=x.device, dtype=x.dtype)

    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # (C, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # (C, 1, 3, 3)

    grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)  # (B, C, H, W)
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)  # (B, C, H, W)

    # grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).mean(dim=1)  # (B, H, W) # per-channel
    grad_mag = torch.sqrt((grad_x ** 2 + grad_y ** 2).mean(dim=1)) # (B, H, W) # L2


    return grad_mag


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

def do_nothing(x, mode=None):
    return x

def generate_src_and_dst_idx(grad: torch.Tensor,
                             sx: int=2,
                             sy: int=2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Divide the grid into small cells(sx*sy), and choose the one with the least gradient magnitude as the dst tokens,
    and the left ones as src tokens.

    Args:
        grad: records gradient magnitude for each input token (B, H, W)
        sx: stride in the x dimension for choosing dst
        sy: stride in the y dimension for choosing dst

    Returns:
        src_idx, dst_idx: two disjoint sets for src and dst token indices, respectively
    """

    B, H, W = grad.shape
    assert H % sy == 0 and W % sx == 0, 'sy must devide H and sx must devide W'

    hsy, hsw = H // sy, W // sx
    num_dst = hsy * hsw

    grad_cells = grad.view(B, hsy, sy, hsw, sx)
    grad_cells = grad_cells.permute(0, 1, 3, 2, 4).contiguous().view(B, num_dst, sy*sx)

    _, max_idx = grad_cells.min(dim=-1) # max gradient in each cell

    grid_row_indices = torch.arange(hsy, device=grad.device).repeat_interleave(torch.tensor(hsw, device=grad.device)) # (H*W) flattened row indices for each cell
    grid_col_indices = torch.arange(hsw, device=grad.device).repeat(torch.tensor(hsy, device=grad.device)) # (H*W) flattened column indices for each cell

    max_local_row_idx = max_idx // sx
    max_local_col_idx = max_idx % sx

    max_global_row_idx = grid_row_indices * sy + max_local_row_idx # (B, num_dst)
    max_global_col_idx = grid_col_indices * sx + max_local_col_idx # (B, num_dst)
    dst_idx = max_global_row_idx * W + max_global_col_idx # (B, num_dst)

    idx_buffer = torch.zeros(B, H*W, device=grad.device, dtype=torch.int64) # (B, H*W)
    idx_buffer.scatter_(dim=-1, index=dst_idx, src=-torch.ones_like(dst_idx)) # (B, num_dst)
    src_idx = idx_buffer.argsort(dim=-1)[:, num_dst:] # (B, H*W - num_dst)

    return src_idx, dst_idx

def grad_bipartite_soft_matching(metric: torch.Tensor,
                                 H: int, W: int,
                                 sx: int=2,
                                 sy: int=2,
                                 grad_method: str='sobel',
                                 r: int=0) -> Tuple[Callable, Callable]:
 
    if r == 0: 
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    T = metric.shape[1]
    
    with torch.no_grad():
        B, N, C = metric.shape

        grad = get_sobel_gradient(metric.view(B, H, W, C)) # (B, H, W)
        a_idx, b_idx = generate_src_and_dst_idx(grad, sx=sx, sy=sy) # (B, T-num_dst), (B, num_dst)

        def split(x):
            n, _, C = x.shape
            src = gather(x, dim=1, index=a_idx.unsqueeze(-1).expand(n, a_idx.shape[1], C))
            dst = gather(x, dim=1, index=b_idx.unsqueeze(-1).expand(n, b_idx.shape[1], C))
            return src, dst

        metric = F.normalize(metric, p=2, dim=-1)
        a, b = split(metric)
        # breakpoint()
        sim = a @ b.transpose(-1, -2)
        r = min(a.shape[1], r)
        node_max, node_idx = sim.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :] # lower pairwise similarity - unmerged tokens
        src_idx = edge_idx[..., :r, :] # higher pairwise similarity - merged tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = split(x)
        n, t1, c = src.shape
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))

        if mode is not None:
            src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        merged_tokens = torch.cat([unm, dst], dim=1)
        unm_absolute_indices = gather(a_idx, dim=1, index=unm_idx.squeeze(-1))
        absolute_indices = torch.cat([unm_absolute_indices, b_idx], dim=1)
        return merged_tokens, absolute_indices

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]

        src = gather(dst, dim=-2, index=dst_idx.expand(n, r, c))
        # Combine back to the original shape
        out = torch.zeros(n, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.unsqueeze(-1).expand(n, b_idx.shape[1], c), src=dst)
        out.scatter_(dim=-2,
                     index=gather(a_idx, dim=1, index=unm_idx.squeeze(-1)).unsqueeze(-1).expand(n, unm_len, c),
                     src=unm)
        out.scatter_(dim=-2, index=gather(a_idx, dim=1, index=src_idx.squeeze(-1)).unsqueeze(-1).expand(n, r, c),
                     src=src)

        return out

    return merge, unmerge


