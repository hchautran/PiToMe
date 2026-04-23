"""
Z-curve (Morton order) token ordering for 2-D image grids.

Provides:
    get_z_order(H, W, device)  ->  perm  (H*W,)  int64
        perm[i] = raster index of the i-th token in Z-order
        i.e. tokens[perm]  reorders a raster-flat sequence to Z-order

    get_z_inverse(H, W, device)  ->  inv_perm  (H*W,)  int64
        inverse permutation: given Z-ordered tokens, put them back in
        raster order  (tokens[inv_perm])

Both functions cache results (keyed on (H, W, device)) so repeated calls are
free after the first.

Usage example
-------------
    from z_utils import get_z_order

    # tokens: (B, H*W, C) in raster order
    perm = get_z_order(H, W, device=tokens.device)
    tokens_z = tokens[:, perm, :]          # Z order

    # ... process ...

    inv_perm = get_z_inverse(H, W, device=tokens.device)
    tokens_raster = tokens_z[:, inv_perm, :]  # back to raster
"""

import math
from functools import lru_cache

import torch


# ─────────────────────────────────────────────────────────────────
# Low-level Z-curve encode / decode (Morton coding)
# ─────────────────────────────────────────────────────────────────

def encode(coords: torch.Tensor, num_dims: int, num_bits: int) -> torch.Tensor:
    """
    (N, num_dims) integer coordinates → Z-curve (Morton) indices.
    
    Interleaves bits of coordinates. For 2D (r, c), it interleaves bits such that:
    index = ... r1 c1 r0 c0
    """
    z = torch.zeros(coords.shape[0], dtype=torch.int64, device=coords.device)
    for i in range(num_bits):
        for j in range(num_dims):
            # Extract i-th bit of j-th coordinate
            bit = (coords[:, j] >> i) & 1
            # Place it at position (i * num_dims + j)
            z |= bit << (i * num_dims + (num_dims - 1 - j))
    return z


def decode(z_indices: torch.Tensor, num_dims: int, num_bits: int) -> torch.Tensor:
    """
    Z-curve (Morton) index → (N, num_dims) coordinates.
    """
    coords = torch.zeros(z_indices.shape[0], num_dims, dtype=torch.int64, device=z_indices.device)
    for i in range(num_bits):
        for j in range(num_dims):
            # Extract (i * num_dims + j)-th bit of z
            bit = (z_indices >> (i * num_dims + (num_dims - 1 - j))) & 1
            # Place it at position i of j-th coordinate
            coords[:, j] |= bit << i
    return coords


# ─────────────────────────────────────────────────────────────────
# Grid-level helpers
# ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _compute_z_order(H: int, W: int, device_str: str):
    """
    Cached computation. Returns (perm, inv_perm) on the requested device.
    """
    device = torch.device(device_str)
    num_bits = max(1, math.ceil(math.log2(max(H, W))))

    # All (row, col) pairs in raster order  →  shape (H*W, 2)
    rows = torch.arange(H, dtype=torch.int64, device=device)
    cols = torch.arange(W, dtype=torch.int64, device=device)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")   # (H, W)
    coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=1)  # (H*W, 2)

    z_idx = encode(coords, num_dims=2, num_bits=num_bits)  # (H*W,)

    # perm[i] = which raster token goes at position i in Z-order
    perm = torch.argsort(z_idx)          # (H*W,) : sort raster by Z-curve key
    inv_perm = torch.argsort(perm)       # inverse permutation

    return perm, inv_perm


def get_z_order(H: int, W: int, device=None) -> torch.Tensor:
    """
    Return a 1-D index tensor `perm` of shape (H*W,) such that
    
        tokens_z = tokens_raster[perm]      # for a 1-D sequence
        tokens_z = tokens_raster[:, perm]   # for (B, H*W, C)

    reorders raster-flattened tokens into Z-curve order.
    """
    if device is None:
        device = torch.device("cpu")
    device_str = str(torch.device(device))
    perm, _ = _compute_z_order(H, W, device_str)
    return perm


def get_z_inverse(H: int, W: int, device=None) -> torch.Tensor:
    """
    Return the inverse permutation: given Z-ordered tokens, restore
    raster order.
    """
    if device is None:
        device = torch.device("cpu")
    device_str = str(torch.device(device))
    _, inv_perm = _compute_z_order(H, W, device_str)
    return inv_perm


@lru_cache(maxsize=64)
def _compute_z_attn_index(H: int, W: int, device_str: str):
    """Cached 2-D index tensor for permuting both axes of an attention matrix."""
    perm, inv_perm = _compute_z_order(H, W, device_str)
    row_idx = perm.unsqueeze(1).expand(H * W, H * W)   # (N, N)
    col_idx = perm.unsqueeze(0).expand(H * W, H * W)   # (N, N)
    return row_idx, col_idx, inv_perm


def get_z_attn_index(H: int, W: int, device=None):
    """
    Return (row_idx, col_idx, inv_perm) for permuting an [*, N, N] attention
    matrix to Z-order and then inverse-permuting the output.
    """
    if device is None:
        device = torch.device("cpu")
    device_str = str(torch.device(device))
    return _compute_z_attn_index(H, W, device_str)
