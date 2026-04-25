
import sys
import os
import types
from typing import Optional, Tuple
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack
import torch
import torch.nn.functional as F
import math

_here = os.path.dirname(__file__)
_sam_root = os.path.normpath(os.path.join(_here, '..', '..', '..', '..', 'sam-hq'))
if _sam_root not in sys.path:
    sys.path.insert(0, _sam_root)

_bsa_cute = os.path.normpath(os.path.join(_here, '..', '..', '..', '..', 'cute'))
if _bsa_cute not in sys.path:
    sys.path.insert(0, _bsa_cute)
from flash_attn import FlashAttentionForwardAmpere

from segment_anything.modeling.image_encoder import (
    ImageEncoderViT,
    Block,
    Attention,
    window_partition,
    window_unpartition,
)
from ..z_utils import get_z_inverse, get_z_order
from typing import Callable


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:

    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()].half()



def aggregate_over_head(x: torch.Tensor, num_heads: int, option: str = "mean") -> torch.Tensor:

    B, N, _ = x.shape
    metric = x.view(B, N, num_heads, -1)

    if option == "max":
        metric = metric.max(dim=2).values
    elif option == "mean":
        metric = metric.mean(dim=2)
    elif option == "sum":
        metric = metric.sum(dim=2)
    else:
        raise ValueError(f"Unknown aggregation option: {option}")

    return metric


_FA2_M_BLOCK_LOCAL  = 64 
_FA2_N_BLOCK_LOCAL  = 64

_FA2_M_BLOCK_GLOBAL = 64
_FA2_N_BLOCK_GLOBAL = 64

_FA2_THREADS_LOCAL  = 64
_FA2_THREADS_GLOBAL = 128

_FA2_DTYPE_FP16 = cutlass.dtype("Float16")
_SPARSE_MASK_DTYPE = cutlass.dtype("Int32")
_FA2_COMPILED: dict = {}
_FA2_CAN_IMPL: dict = {}
_SPARSE_MASK_CACHE: dict = {}


def compute_rel_bias(
    q_bshd: torch.Tensor,
    Rh: torch.Tensor,
    Rw: torch.Tensor,
    win: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, D = q_bshd.shape
    r_q   = q_bshd.reshape(B , win, win, D)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).reshape(B, N, win)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).reshape(B, N, win)
    to_fa2 = lambda t: t.unsqueeze(1).permute(0, 2, 1, 3).contiguous()
    return to_fa2(rel_h), to_fa2(rel_w)


def _wrap_qkvo(t: torch.Tensor, dtype) -> "cute.Tensor":
    return (from_dlpack(t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(
                mode=3,
                stride_order=t.dim_order(),
                divisibility=128 // dtype.width))


def _wrap_bias(t: torch.Tensor) -> "cute.Tensor":
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=3)


def _wrap_perm(t: torch.Tensor) -> "cute.Tensor":
    ct = from_dlpack(t, assumed_align=4)
    if t.dim() == 2:
        ct = ct.mark_layout_dynamic(leading_dim=1)
    return ct


def _fa2_can_implement(
    D: int, m_block: int, n_block: int, threads: int) -> bool:
    key = (D, m_block, n_block, threads)
    if key not in _FA2_CAN_IMPL:
        _FA2_CAN_IMPL[key] = FlashAttentionForwardAmpere.can_implement(
            _FA2_DTYPE_FP16, D, m_block, n_block, threads
        )
    return _FA2_CAN_IMPL[key]


def make_A_mask(B, H, T, ratio, m_block, n_block, device="cuda"):
    num_m_blocks = math.ceil(T / m_block)
    num_n_blocks = math.ceil(T / n_block)

    t = torch.zeros(B, H, num_m_blocks, num_n_blocks, dtype=torch.int32, device=device)
    t = t + torch.eye(num_m_blocks, num_n_blocks, dtype=torch.int32, device=device)

    t[:, :, :, :int(ratio * num_n_blocks)] = 1
    ct = from_dlpack(t, assumed_align=4)
    return ct, t

def _get_fa2_compiled(
    B, H ,q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c,
    win, scale, cu_stream,
    D: int, m_block: int, n_block: int, threads: int,
    ratio:float
):
    key = (win, D, m_block, n_block, threads, ratio)
    mask_key = (B, H, win)
    if mask_key not in _SPARSE_MASK_CACHE:
        ct, t = make_A_mask(B, H, win**2, ratio, m_block, n_block)
        _SPARSE_MASK_CACHE[mask_key] = (ct, t)

    ct_mask, t_mask = _SPARSE_MASK_CACHE[mask_key]

    if key not in _FA2_COMPILED:
        _FA2_COMPILED[key] = cute.compile(
            FlashAttentionForwardAmpere(D, m_block, n_block, threads, win),
            q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c, ct_mask, scale, cu_stream,
        )
    return _FA2_COMPILED[key], ct_mask


# ── Ablation: random permutation ─────────────────────────────────────────────
def tile_stride_matching_random(
    x:          torch.Tensor,   # (B, N, C)
    H:          int,
    W:          int,
    ratio:      float = 0.0,    # unused — kept for API compatibility
    group_size: int   = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:

    N = H * W
    assert N % group_size == 0, f"N={N} must be divisible by group_size={group_size}"

    n_groups = N // group_size
    gs       = group_size
    ratio_n  = round(ratio * N)                      # exact primary-token target
    n_keep   = min(ratio_n // gs, n_groups)          # full groups in keep (floor → n_keep*gs ≤ ratio_n)
    n_extra  = ratio_n - n_keep * gs if n_keep < n_groups else 0  # tokens borrowed from merge_flat start
    n_merge  = n_groups - n_keep                     # low-gradient groups, interleaved
    device  = x.device

    perm_1d = torch.stack([
        torch.randperm(N, device=device) for _ in range(x.shape[0])
        ], dim=0)  
    inv_perm_1d = torch.argsort(perm_1d, dim=1)               # (B, N) int64
    return perm_1d, inv_perm_1d
    


class ToMeSAMAttentionRandom(Attention):

    def forward(self, x: torch.Tensor, ratio: float, use_fa2: bool = True,
                m_block: int = _FA2_M_BLOCK_LOCAL,
                n_block: int = _FA2_N_BLOCK_LOCAL,
                threads: int = _FA2_THREADS_LOCAL,
                sparsity: float = 0.0,
                custom_mask: "cute.Tensor | None" = None,
                return_perm: bool = False) -> torch.Tensor:
        B, H, W, _ = x.shape
        Sq  = H * W
        D   = _ // self.num_heads
        win = H

        BH = B * self.num_heads
        qkv = self.qkv(x.view(B, Sq, -1))
        qkv = qkv.view(B, Sq, 3, self.num_heads, D).permute(2, 0, 3, 1, 4).reshape(3, BH, Sq, D)
        q, k, v = qkv.unbind(0)
        o = torch.empty_like(q)

        if not hasattr(self, '_Rh') or self._Rh is None:
            self._Rh = get_rel_pos(win, win, self.rel_pos_h)
            self._Rw = get_rel_pos(win, win, self.rel_pos_w)
        Rh, Rw = self._Rh, self._Rw

        rel_h, rel_w = compute_rel_bias(q, Rh, Rw, win)

        # Random permutation — NOT cached across layers (each layer gets fresh random order)
        perm, inv_perm = tile_stride_matching_random(k, win, win, ratio=ratio)

        perm_e = perm.unsqueeze(-1).expand(-1, -1, D)
        q_p = q.gather(1, perm_e)
        k_p = k.gather(1, perm_e)
        v_p = v.gather(1, perm_e)

        perm_q_t = perm.to(torch.int32)
        perm_k_t = perm.to(torch.int32)

        cu_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        q_c = _wrap_qkvo(q_p.unsqueeze(2), _FA2_DTYPE_FP16)
        k_c = _wrap_qkvo(k_p.unsqueeze(2), _FA2_DTYPE_FP16)
        v_c = _wrap_qkvo(v_p.unsqueeze(2), _FA2_DTYPE_FP16)
        o_c = _wrap_qkvo(o.unsqueeze(2), _FA2_DTYPE_FP16)
        rh_c = _wrap_bias(rel_h)
        rw_c = _wrap_bias(rel_w)
        perm_q_c = _wrap_perm(perm_q_t)
        perm_k_c = _wrap_perm(perm_k_t)

        compiled, default_mask = _get_fa2_compiled(
            BH, 1, q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c,
            win, self.scale, cu_stream, D, m_block, n_block, threads,
            ratio=ratio,
        )
        mask = custom_mask if custom_mask is not None else default_mask
        compiled(q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c, mask, self.scale, cu_stream)

        inv_perm_e = inv_perm.unsqueeze(-1).expand(-1, -1, D)
        o_out = o.gather(1, inv_perm_e).reshape(B, self.num_heads, Sq, D).permute(0, 2, 1, 3).reshape(B, Sq, -1)
        out = self.proj(o_out).reshape(B, H, W, -1)
        if return_perm:
            return out, perm, inv_perm
        return out


@torch.no_grad()
def _compute_window_dense_random(
    x: torch.Tensor,   # (B, H, W, C)
    ws: int,
    ratio: float,
) -> torch.Tensor:      # (B*nW,) bool — True = dense attention
    """Random window selection ablation: assigns dense/sparse labels uniformly at random."""
    B, H, W, C = x.shape

    H_pad = (ws - H % ws) % ws
    W_pad = (ws - W % ws) % ws
    nH = (H + H_pad) // ws
    nW_tiles = (W + W_pad) // ws
    nWin = nH * nW_tiles

    n_dense = max(1, round(ratio * nWin))

    dense = torch.zeros(B, nWin, dtype=torch.bool, device=x.device)
    for b in range(B):
        idx = torch.randperm(nWin, device=x.device)[:n_dense]
        dense[b, idx] = True
    return dense.reshape(B * nWin)


class ToMeSAMBlockRandom(Block):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H_sp, W_sp, C = x.shape
        info  = self._tome_info
        ratio = info["ratio"].pop(0)
        sparsity = info.get("sparsity", 1.0)

        shortcut = x
        x_n = self.norm1(x)
        if self.window_size > 0:
            ws = self.window_size
            H_w, W_w = x_n.shape[1], x_n.shape[2]
            x_n_win, pad_hw = window_partition(x_n, ws)

            if info["window_dense"] is None:
                info["window_dense"] = _compute_window_dense_random(x, ws, ratio)

            dense_flat = info["window_dense"]
            nh  = self.attn.num_heads
            nW  = x_n_win.shape[0] // B
            BH  = B * nW * nh
            T   = ws * ws
            num_m = math.ceil(T / _FA2_M_BLOCK_LOCAL)
            num_n = math.ceil(T / _FA2_N_BLOCK_LOCAL)

            # if info["local_mask_buf"] is None:
            #     num_init = max(1, int(ratio * num_m)) 
            #     base = torch.eye(num_m, num_n, dtype=torch.int32, device=x.device)
            #     base[:, :num_init] = 1
            #     buf  = base.unsqueeze(0).unsqueeze(0).expand(BH, 1, -1, -1).contiguous()

            #     info["local_mask_diag"] = buf.clone()
            #     info["local_mask_buf"]  = buf
            #     info["local_ct_mask"]   = from_dlpack(buf, assumed_align=4)

            # if info["local_dense_bh"] is None:
            #     info["local_dense_bh"] = dense_flat.repeat_interleave(nh)

            # buf      = info["local_mask_buf"]
            # # dense_bh = info["local_dense_bh"]
            # buf.copy_(info["local_mask_diag"])
            # # buf[dense_bh] = 1
            # ct_mask  = info["local_ct_mask"]

            x_attn, perm, inv_perm = self.attn(
                x_n_win, ratio,
                use_fa2=True,
                custom_mask=None,
                return_perm=True,
            )
            x_attn = window_unpartition(x_attn, ws, pad_hw, (H_w, W_w))
        else:
            x_attn, perm, inv_perm = self.attn(
                x_n, ratio,
                m_block=_FA2_M_BLOCK_GLOBAL,
                n_block=_FA2_N_BLOCK_GLOBAL,
                threads=_FA2_THREADS_GLOBAL,
                sparsity=sparsity,
                return_perm=True,
            )

        x = shortcut + x_attn
        x_seq = x.reshape(B, H_sp * W_sp, C)

        if self.window_size == 0 and ratio < 1.0:
            keep_n = max(1, round(0.25 * x_seq.shape[1]))
            perm = perm.view(B, self.attn.num_heads, -1)[:, 0, :]
            inv_perm = inv_perm.view(B, self.attn.num_heads, -1)[:, 0, :]
            perm_e = perm.unsqueeze(-1).expand(-1, -1, C)
            inv_perm_e = inv_perm.unsqueeze(-1).expand(-1, -1, C)

            x_perm = x_seq.gather(1, perm_e)
            x_kept = x_perm[:, :keep_n, :]
            x_kept = x_kept + self.mlp(self.norm2(x_kept))
            x_perm[:, :keep_n, :] = x_kept

            x_seq = x_perm.gather(1, inv_perm_e)
        else:
            x_seq = x_seq + self.mlp(self.norm2(x_seq))

        return x_seq.reshape(B, H_sp, W_sp, C)


def _warmup_fa2_kernels(encoder: ImageEncoderViT, sparsity) -> None:

    device = next(encoder.parameters()).device
    seen: set = set()

    for blk in encoder.blocks:
        attn      = blk.attn
        is_global = (blk.window_size == 0)

        win = (attn.rel_pos_h.shape[0] + 1) // 2
        D   = attn.rel_pos_h.shape[1]

        if not hasattr(attn, '_Rh') or attn._Rh is None:
            attn._Rh = get_rel_pos(win, win, attn.rel_pos_h)
            attn._Rw = get_rel_pos(win, win, attn.rel_pos_w)

        if not is_global:
            continue

        m_block = _FA2_M_BLOCK_GLOBAL
        n_block = _FA2_N_BLOCK_GLOBAL
        threads = _FA2_THREADS_GLOBAL

        compile_key = (win, D, m_block, n_block, threads, sparsity)
        if compile_key in seen or not _fa2_can_implement(D, m_block, n_block, threads):
            seen.add(compile_key)
            continue
        seen.add(compile_key)

        Sq = win * win
        H  = attn.num_heads
        B  = 1

        q = torch.zeros(B, Sq, H, D, dtype=torch.float16, device=device)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        o = torch.zeros_like(q)
        rel_h = torch.zeros(B, Sq, H, win, dtype=torch.float16, device=device)
        rel_w = torch.zeros_like(rel_h)

        cu_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        q_c = _wrap_qkvo(q, _FA2_DTYPE_FP16)
        k_c = _wrap_qkvo(k, _FA2_DTYPE_FP16)
        v_c = _wrap_qkvo(v, _FA2_DTYPE_FP16)
        o_c = _wrap_qkvo(o, _FA2_DTYPE_FP16)
        rh_c = _wrap_bias(rel_h)
        rw_c = _wrap_bias(rel_w)
        identity = torch.arange(Sq, device=q.device, dtype=torch.int32).unsqueeze(0)
        perm_q_c = _wrap_perm(identity)
        perm_k_c = _wrap_perm(identity)

        print(
            f"[ToMe-SAM-Random] compiling FA2 kernel  global  "
            f"win={win}  D={D}  m={m_block}  n={n_block}  T={threads}  "
            f"sparsity={sparsity:.3f}   ...",
            end=" ", flush=True,
        )
        _get_fa2_compiled(
            B, H, q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c,
            win, attn.scale, cu_stream,
            D, m_block, n_block, threads,
            sparsity,
        )
        print("done")


def apply_patch(
    encoder: ImageEncoderViT,
    algo: str = "sparsesam_random",
    ratio: float = 0.9,
    margin: float = 0.5,
    trace_source: bool = False,
    sparsity: float = 0.0,
) -> ImageEncoderViT:
    print('sparsity', sparsity)

    assert 0 < ratio <= 1.0, "ratio must be in (0, 1]"
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1)"

    tome_info = {
        "algo": algo,
        "ratio": ratio,
        "margin": margin,
        "x_attn": None,
        "metric": None,
        "sparsity": sparsity,
        "local_mask_buf":  None,
        "local_mask_diag": None,
        "local_ct_mask":   None,
        "local_dense_bh":  None,
    }
    encoder.tome_info = tome_info

    _orig_forward = encoder.__class__.forward

    def _patched_forward(self, x: torch.Tensor):
        n = len(self.blocks)
        r = self.tome_info["ratio"]
        self.tome_info["ratio"] = [r] * n
        self.tome_info["perm_cache"]     = {}
        self.tome_info["mlp_perm_cache"] = {}
        self.tome_info["window_dense"]   = None
        self.tome_info["local_dense_bh"] = None

        result = _orig_forward(self, x)

        self.tome_info["ratio"] = r
        return result

    encoder.forward = types.MethodType(_patched_forward, encoder)

    for module in encoder.modules():
        if isinstance(module, Block) and not isinstance(module, ToMeSAMBlockRandom):
            module.__class__  = ToMeSAMBlockRandom
            module._tome_info = tome_info
        elif isinstance(module, Attention) and not isinstance(module, ToMeSAMAttentionRandom):
            module.__class__  = ToMeSAMAttentionRandom
            module._tome_info = tome_info

    n_blocks = len(encoder.blocks)
    n_global = sum(1 for blk in encoder.blocks if blk.window_size == 0)
    print(
        f"[ToMe-SAM-Random] patched  algo={algo}  ratio={ratio}"
        + (f"  margin={margin}" if algo == "pitome" else "")
        + f"  sparsity={sparsity:.2f}"
        + f"  blocks={n_blocks} (global={n_global} local={n_blocks-n_global})"
        + "  strategy=post-attn-merge / post-mlp-unmerge (all blocks)"
        + "  token-order=random (ablation: no gradient-based ordering)"
    )

    _warmup_fa2_kernels(encoder, sparsity=sparsity)

    return encoder
