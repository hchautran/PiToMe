from typing import Callable, Tuple
import torch
import torch.nn.functional as F
import math
from .hilbert_utils import get_hilbert_order


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

def tile_stride_matching(
    x:          torch.Tensor,   # (B, N, C) in raster order, N = H*W
    H:          int,
    W:          int,
    r:      float,          # fraction of tokens to KEEP
    group_size: int = 4,        # tokens per Hilbert group; 4 (2×2) is fine-grained, 64 (8×8) is coarse
) -> Tuple[Callable, Callable]:
    """
    Partition tokens into Hilbert groups, rank groups by spatial gradient
    magnitude, and return (merge, unmerge) callables.

    merge(x_in)  → (x_merged, None)
        x_in   : (B, N, C)
        x_merged: (B, N', C)  where N' = n_keep*group_size + n_merge

    unmerge(x_out) → (B, N, C)
        Restores full spatial resolution; merge-group members receive the
        group representative's value (mean of the original group).
    """
    N = H * W
    assert N % group_size == 0, (
        f"N={N} (H={H}, W={W}) must be divisible by group_size={group_size}"
    )

    n_groups = N // group_size
    gs       = group_size

    # How many groups to merge  (each merge removes gs-1 tokens)
    if r <= 0 or n_groups <= 1:
        def _identity(x_in, mode=None):
            if mode == 'permute':
                B_in = x_in.shape[0]
                return torch.arange(N, device=x_in.device, dtype=torch.int32).unsqueeze(0).expand(B_in, -1)
            return x_in, None
        return _identity, lambda x_out, mode=None: x_out

    n_merge = min(n_groups - 1, math.ceil(r / max(gs - 1, 1)))
    n_keep  = n_groups - n_merge

    with torch.no_grad():
        B, _ , C = x.shape
        device  = x.device

        # ── 1. Full-channel Sobel gradient (matches merge.py get_sobel_gradient) ──
        # Flatten batch×channel into a single batch dim so we need only one kernel.
        x_bchw = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x_flat = x_bchw.reshape(B * C, 1, H, W)                       # (B*C, 1, H, W)

        sobel_x_k = torch.tensor(
            [[-1., 0, 1], [-2., 0, 2], [-1., 0, 1]],
            device=device, dtype=torch.float16
        ).view(1, 1, 3, 3)
        sobel_y_k = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            device=device, dtype=torch.float16
        ).view(1, 1, 3, 3)

        gx = F.conv2d(x_flat, sobel_x_k, padding=1).reshape(B, C, H, W)
        gy = F.conv2d(x_flat, sobel_y_k, padding=1).reshape(B, C, H, W)
        # L2 across channels → (B, N) raster
        sobel_flat = torch.sqrt((gx**2 + gy**2).mean(dim=1)).reshape(B, N)

        # ── 2. Reorder to Hilbert curve ───────────────────────────────────────
        # perm[hilbert_pos] = raster_idx  →  x_hilbert = x_raster[:, perm]
        perm = get_hilbert_order(H, W, device=device)       # (N,)
        sobel_z = sobel_flat[:, perm]                       # (B, N) Hilbert order

        # ── 3. Per-group mean Sobel magnitude ─────────────────────────────────
        grp_sobel = sobel_z.view(B, n_groups, gs).mean(-1)  # (B, n_groups)

        # ── 4. Rank groups ascending → lowest gradient = merge candidates ─────
        grp_rank   = grp_sobel.argsort(dim=-1)                    # (B, n_groups)
        merge_grps = grp_rank[:, :n_merge]                         # (B, n_merge)
        keep_grps  = grp_rank[:, n_merge:]                         # (B, n_keep)


        # ── 5. Raster token indices per Hilbert group ─────────────────────────
        # Hilbert group g occupies Hilbert positions [g*gs, (g+1)*gs).
        # perm[g*gs : (g+1)*gs] gives the raster indices for group g.
        group_raster = perm.view(n_groups, gs)                     # (n_groups, gs)
        breakpoint()

        # Merge group raster indices: (B, n_merge, gs)
        merge_raster = group_raster[merge_grps.reshape(-1)].reshape(B, n_merge, gs)
        merge_flat   = merge_raster.reshape(B, n_merge * gs)       # (B, n_merge*gs)

        # Keep group raster indices: (B, n_keep*gs)
        keep_raster = group_raster[keep_grps.reshape(-1)].reshape(B, n_keep, gs)
        keep_flat   = keep_raster.reshape(B, n_keep * gs)          # (B, n_keep*gs)

        # Permute order: all z-order groups sorted ascending by gradient (lowest first),
        # each group's tokens contiguous in z-order — no transpose/interleave.
        full_perm     = group_raster[grp_rank.reshape(-1)].reshape(B, N)          # (B, N)
        full_inv_perm = full_perm.argsort(dim=1)                                  # (B, N)
        breakpoint()

    n_keep_flat = n_keep * gs

    def merge(x_in: torch.Tensor, mode: str = None):
        Bx, Nx, Cx = x_in.shape
        if mode == 'permute':
            return full_perm                                                        # (B, N)

        k_idx = keep_flat.unsqueeze(-1).expand(Bx, n_keep_flat, Cx)
        m_idx = merge_flat.unsqueeze(-1).expand(Bx, n_merge * gs, Cx)
        if mode is None:
            merge_tokens = x_in.gather(1, m_idx).reshape(Bx, n_merge, gs, Cx)
            merge_repr = merge_tokens[:, :, 0, :]                     # (B, n_merge, C)
        else:
            merge_tokens = x_in.gather(1, m_idx).reshape(Bx, n_merge, gs, Cx)
            merge_repr = merge_tokens.mean(dim=2)                     # (B, n_merge, C)

        keep_tokens = x_in.gather(1, k_idx)                        # (B, n_keep*gs, C)
        return torch.cat([keep_tokens, merge_repr], dim=1)          # (B, N', C)

    def unmerge(x_out: torch.Tensor, mode: str = None) -> torch.Tensor:
        Bx, _, Cx = x_out.shape

        if mode == 'permute':
            idx = full_inv_perm.unsqueeze(-1).expand(Bx, N, Cx)
            return x_out.gather(1, idx)                             # (B, N, C)

        keep_out  = x_out[:, :n_keep_flat, :]                      # (B, n_keep*gs, C)
        merge_out = x_out[:, n_keep_flat:, :]                      # (B, n_merge, C)

        out = torch.zeros(Bx, N, Cx, device=x_out.device, dtype=x_out.dtype)

        # Scatter keep tokens back to their original raster positions
        k_idx = keep_flat.unsqueeze(-1).expand(Bx, n_keep_flat, Cx)
        out.scatter_(1, k_idx, keep_out)

        # Broadcast each group representative to all gs group members
        m_exp = merge_out.unsqueeze(2).expand(Bx, n_merge, gs, Cx)
        m_idx = merge_flat.unsqueeze(-1).expand(Bx, n_merge * gs, Cx)
        out.scatter_(1, m_idx, m_exp.reshape(Bx, n_merge * gs, Cx))

        return out

    return merge, unmerge



