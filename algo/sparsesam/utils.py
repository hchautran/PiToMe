# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import time
from typing import List, Tuple, Union

import torch
from tqdm import tqdm


def benchmark(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model (channels, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    input = torch.rand(batch_size, *input_size, device=device)
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()

                model(input)
                total += batch_size

    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start

    throughput = total / elapsed

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")

    return throughput


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






def _h_to_transposed(H: torch.Tensor, n: int, p: int) -> torch.Tensor:
    """Unpack Hilbert integer(s) into Skilling's transposed representation."""
    X = torch.zeros(*H.shape, n, dtype=torch.int64, device=H.device)
    for i in range(p):
        for j in range(n):
            bit = (H >> (n * p - 1 - (i * n + j))) & 1
            X[..., j] |= bit << (p - 1 - i)
    return X


def _transposed_to_h(X: torch.Tensor, n: int, p: int) -> torch.Tensor:
    """Pack Skilling's transposed representation into Hilbert integer(s)."""
    H = torch.zeros(X.shape[0], dtype=torch.int64, device=X.device)
    for i in range(p):
        for j in range(n):
            bit = (X[:, j] >> (p - 1 - i)) & 1
            H |= bit << (n * p - 1 - (i * n + j))
    return H


def _inverse_hilbert_transform(X: torch.Tensor, n: int, p: int) -> torch.Tensor:
    """Skilling inverse transform: transposed Hilbert form → coordinates."""
    X = X.clone()
    M = 1 << (p - 1)

    # Gray decode by prefix scan
    t = X[:, n - 1] >> 1
    for i in range(n - 1, 0, -1):
        X[:, i] ^= X[:, i - 1]
    X[:, 0] ^= t

    # Undo excess work
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
    """Skilling forward transform: coordinates → transposed Hilbert form."""
    X = X.clone()
    M = 1 << (p - 1)

    # Inverse undo
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            hi  = (X[:, i] & Q) != 0
            swp = (X[:, 0] ^ X[:, i]) & P
            X[:, 0] = torch.where(hi, X[:, 0] ^ P, X[:, 0] ^ swp)
            X[:, i] = torch.where(hi, X[:, i],      X[:, i] ^ swp)
        Q >>= 1

    # Gray encode
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


def decode(hilbert_indices, num_dims: int, num_bits: int,
           device: str = "cpu") -> torch.Tensor:
    """Decode Hilbert indices → coordinates.

    Args:
        hilbert_indices : array-like or 1-D int64 Tensor.
        num_dims        : number of spatial dimensions.
        num_bits        : bits per axis  (grid size = 2**num_bits per axis).
        device          : torch device (used only when input is not a Tensor).
    Returns:
        int64 Tensor of shape (N, num_dims).
    """
    if not isinstance(hilbert_indices, torch.Tensor):
        hilbert_indices = torch.tensor(hilbert_indices, dtype=torch.int64, device=device)
    X = _h_to_transposed(hilbert_indices.to(torch.int64), num_dims, num_bits)
    return _inverse_hilbert_transform(X, num_dims, num_bits)


def encode(coords, num_dims: int, num_bits: int,
           device: str = "cpu") -> torch.Tensor:
    """Encode coordinates → Hilbert indices.

    Args:
        coords   : array-like or Tensor of shape (N, num_dims), integer coords.
        num_dims : number of spatial dimensions.
        num_bits : bits per axis.
        device   : torch device (used only when input is not a Tensor).
    Returns:
        int64 Tensor of shape (N,).
    """
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.int64, device=device)
    X = _forward_hilbert_transform(coords.to(torch.int64), num_dims, num_bits)
    return _transposed_to_h(X, num_dims, num_bits)
