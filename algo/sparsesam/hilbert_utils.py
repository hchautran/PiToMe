"""
Hilbert-curve token ordering for 2-D image grids.

Provides:
    get_hilbert_order(H, W, device)  ->  perm  (H*W,)  int64
        perm[i] = raster index of the i-th token in Hilbert order
        i.e. tokens[perm]  reorders a raster-flat sequence to Hilbert order

    get_hilbert_inverse(H, W, device)  ->  inv_perm  (H*W,)  int64
        inverse permutation: given Hilbert-ordered tokens, put them back in
        raster order  (tokens[inv_perm])

Both functions cache results (keyed on (H, W, device)) so repeated calls are
free after the first.

Usage example
-------------
    from hilbert_utils import get_hilbert_order

    # tokens: (B, H*W, C) in raster order
    perm = get_hilbert_order(H, W, device=tokens.device)
    tokens_hilbert = tokens[:, perm, :]          # Hilbert order

    # ... process ...

    inv_perm = get_hilbert_inverse(H, W, device=tokens.device)
    tokens_raster = tokens_hilbert[:, inv_perm, :]  # back to raster
"""

import math
from functools import lru_cache

import torch


# ─────────────────────────────────────────────────────────────────
# Low-level Hilbert encode / decode  (Skilling's algorithm)
# ─────────────────────────────────────────────────────────────────

def _h_to_transposed(H: torch.Tensor, n: int, p: int) -> torch.Tensor:
    X = torch.zeros(*H.shape, n, dtype=torch.int64, device=H.device)
    for i in range(p):
        for j in range(n):
            bit = (H >> (n * p - 1 - (i * n + j))) & 1
            X[..., j] |= bit << (p - 1 - i)
    return X


def _transposed_to_h(X: torch.Tensor, n: int, p: int) -> torch.Tensor:
    H = torch.zeros(X.shape[0], dtype=torch.int64, device=X.device)
    for i in range(p):
        for j in range(n):
            bit = (X[:, j] >> (p - 1 - i)) & 1
            H |= bit << (n * p - 1 - (i * n + j))
    return H


def _inverse_hilbert_transform(X: torch.Tensor, n: int, p: int) -> torch.Tensor:
    X = X.clone()
    M = 1 << (p - 1)
    t = X[:, n - 1] >> 1
    for i in range(n - 1, 0, -1):
        X[:, i] ^= X[:, i - 1]
    X[:, 0] ^= t
    Q = 2
    while Q != M << 1:
        P = Q - 1
        for i in range(n - 1, -1, -1):
            hi  = (X[:, i] & Q) != 0
            swp = (X[:, 0] ^ X[:, i]) & P
            X[:, 0] = torch.where(hi, X[:, 0] ^ P, X[:, 0] ^ swp)
            X[:, i] = torch.where(hi, X[:, i],      X[:, i] ^ swp)
        Q <<= 1
    return X


def _forward_hilbert_transform(X: torch.Tensor, n: int, p: int) -> torch.Tensor:
    X = X.clone()
    M = 1 << (p - 1)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            hi  = (X[:, i] & Q) != 0
            swp = (X[:, 0] ^ X[:, i]) & P
            X[:, 0] = torch.where(hi, X[:, 0] ^ P, X[:, 0] ^ swp)
            X[:, i] = torch.where(hi, X[:, i],      X[:, i] ^ swp)
        Q >>= 1
    for i in range(1, n):
        X[:, i] ^= X[:, i - 1]
    t = torch.zeros(X.shape[0], dtype=torch.int64, device=X.device)
    Q = M
    while Q > 1:
        t = torch.where((X[:, n - 1] & Q) != 0, t ^ (Q - 1), t)
        Q >>= 1
    for i in range(n):
        X[:, i] ^= t
    return X


def decode(hilbert_indices: torch.Tensor, num_dims: int, num_bits: int) -> torch.Tensor:
    """Hilbert index → (N, num_dims) coordinates."""
    X = _h_to_transposed(hilbert_indices.to(torch.int64), num_dims, num_bits)
    return _inverse_hilbert_transform(X, num_dims, num_bits)


def encode(coords: torch.Tensor, num_dims: int, num_bits: int) -> torch.Tensor:
    """(N, num_dims) integer coordinates → Hilbert indices."""
    X = _forward_hilbert_transform(coords.to(torch.int64), num_dims, num_bits)
    return _transposed_to_h(X, num_dims, num_bits)


# ─────────────────────────────────────────────────────────────────
# Grid-level helpers
# ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _compute_hilbert_order(H: int, W: int, device_str: str):
    """
    Cached computation.  Returns (perm, inv_perm) on the requested device.

    num_bits is chosen to be ceil(log2(max(H, W))) so the Hilbert grid
    covers every coordinate.  Coordinates outside [0, H) x [0, W) never
    appear because we enumerate the grid explicitly.
    """
    device = torch.device(device_str)
    num_bits = max(1, math.ceil(math.log2(max(H, W))))

    # All (row, col) pairs in raster order  →  shape (H*W, 2)
    rows = torch.arange(H, dtype=torch.int64, device=device)
    cols = torch.arange(W, dtype=torch.int64, device=device)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")   # (H, W)
    coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=1)  # (H*W, 2)

    h_idx = encode(coords, num_dims=2, num_bits=num_bits)  # (H*W,)

    # perm[i] = which raster token goes at position i in Hilbert order
    perm = torch.argsort(h_idx)          # (H*W,) : sort raster by Hilbert key
    inv_perm = torch.argsort(perm)       # inverse permutation

    return perm, inv_perm
    

def get_hilbert_order(H: int, W: int, device=None) -> torch.Tensor:

    if device is None:
        device = torch.device("cpu")
    device_str = str(torch.device(device))
    perm, _ = _compute_hilbert_order(H, W, device_str)
    return perm


def get_hilbert_inverse(H: int, W: int, device=None) -> torch.Tensor:

    if device is None:
        device = torch.device("cpu")
    device_str = str(torch.device(device))
    _, inv_perm = _compute_hilbert_order(H, W, device_str)
    return inv_perm


@lru_cache(maxsize=64)
def _compute_hilbert_attn_index(H: int, W: int, device_str: str):

    perm, inv_perm = _compute_hilbert_order(H, W, device_str)
    # attn_hilbert = attn_raster[perm[:, None], perm[None, :]]
    # Building the meshgrid once avoids re-broadcasting on every forward pass.
    row_idx = perm.unsqueeze(1).expand(H * W, H * W)   # (N, N)
    col_idx = perm.unsqueeze(0).expand(H * W, H * W)   # (N, N)
    return row_idx, col_idx, inv_perm


def get_hilbert_attn_index(H: int, W: int, device=None):

    if device is None:
        device = torch.device("cpu")
    device_str = str(torch.device(device))
    return _compute_hilbert_attn_index(H, W, device_str)
