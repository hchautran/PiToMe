
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

_here = os.path.dirname(__file__)
_sam_root = os.path.normpath(os.path.join(_here, '..', '..', '..', '..', 'sam-hq'))
if _sam_root not in sys.path:
    sys.path.insert(0, _sam_root)

_bsa_cute = os.path.normpath(os.path.join(_here, '..', '..', '..', '..', 'Block-Sparse-Attention', 'cute'))
if _bsa_cute not in sys.path:
    sys.path.insert(0, _bsa_cute)
from flash_attn import FlashAttentionForwardAmpere

from segment_anything.modeling.image_encoder import (
    ImageEncoderViT,
    Block,
    Attention,
    add_decomposed_rel_pos,
    window_partition,
    window_unpartition,
)
from ..merge import merge_wavg, bipartite_soft_matching
from ..hilbert_utils import get_hilbert_inverse, get_hilbert_order


    
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


_FA2_M_BLOCK_LOCAL  = 32
_FA2_N_BLOCK_LOCAL  = 32 

_FA2_M_BLOCK_GLOBAL = 64 
_FA2_N_BLOCK_GLOBAL = 64 

_FA2_THREADS_LOCAL  = 64 
_FA2_THREADS_GLOBAL = 128 

_FA2_DTYPE_FP16 = cutlass.dtype("Float16")

_FA2_COMPILED: dict = {}

_FA2_CAN_IMPL: dict = {}
_HILBERT_CACHE: dict = {}


def compute_rel_bias(
    q_bshd: torch.Tensor,
    Rh: torch.Tensor,
    Rw: torch.Tensor,
    win: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, Sq, H, D = q_bshd.shape
    r_q   = q_bshd.permute(0, 2, 1, 3).reshape(B * H, win, win, D)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).reshape(B * H, Sq, win)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).reshape(B * H, Sq, win)
    to_fa2 = lambda t: t.reshape(B, H, Sq, win).permute(0, 2, 1, 3).contiguous()
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
    return from_dlpack(t, assumed_align=4)


def _get_hilbert_perm(win: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if win not in _HILBERT_CACHE:
        order = get_hilbert_order(win, win).to(device=device, dtype=torch.int32)
        inv   = get_hilbert_inverse(win, win).to(device=device, dtype=torch.int64)
        _HILBERT_CACHE[win] = (order, inv)
    return _HILBERT_CACHE[win]


def _fa2_can_implement(
    D: int, m_block: int, n_block: int, threads: int) -> bool:
    key = (D, m_block, n_block, threads)
    if key not in _FA2_CAN_IMPL:
        _FA2_CAN_IMPL[key] = FlashAttentionForwardAmpere.can_implement(
            _FA2_DTYPE_FP16, D, m_block, n_block, threads
        )
    return _FA2_CAN_IMPL[key]


def _get_fa2_compiled(
    q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c,
    win, scale, cu_stream,
    D: int, m_block: int, n_block: int, threads: int
):
    key = (win, D, m_block, n_block, threads)
    if key not in _FA2_COMPILED:
        _FA2_COMPILED[key] = cute.compile(
            FlashAttentionForwardAmpere(D, m_block, n_block, threads, win),
            q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c, scale, cu_stream,
            options="",
        )
    return _FA2_COMPILED[key]


class ToMeSAMAttention(Attention):

    def forward(self, x: torch.Tensor, ratio, use_fa2=True,
                m_block: int = _FA2_M_BLOCK_LOCAL,
                n_block: int = _FA2_N_BLOCK_LOCAL,
                threads: int = _FA2_THREADS_LOCAL,
                ) -> torch.Tensor:
        B, H, W, _ = x.shape
        Sq  = H * W
        D   = _ // self.num_heads  
        win = H  

        qkv = self.qkv(x.view(B, Sq, -1))
        qkv = qkv.view(B, Sq, 3, self.num_heads, D).permute(2,0,1,3,4).contiguous()
        q, k, v = qkv.unbind(0)

        if not hasattr(self, '_Rh') or self._Rh is None:
            self._Rh = get_rel_pos(win, win, self.rel_pos_h)
            self._Rw = get_rel_pos(win, win, self.rel_pos_w)
        Rh, Rw = self._Rh, self._Rw

        rel_h, rel_w = compute_rel_bias(q, Rh, Rw, win)

        hilbert_order, inv_hilbert = _get_hilbert_perm(win, x.device)
        q = q[:, hilbert_order, :, :].contiguous()
        k = k[:, hilbert_order, :, :].contiguous()
        v = v[:, hilbert_order, :, :].contiguous()

        o = torch.empty_like(q)
        cu_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        q_c = _wrap_qkvo(q, _FA2_DTYPE_FP16)
        k_c = _wrap_qkvo(k, _FA2_DTYPE_FP16)
        v_c = _wrap_qkvo(v, _FA2_DTYPE_FP16)
        o_c = _wrap_qkvo(o, _FA2_DTYPE_FP16)
        rh_c = _wrap_bias(rel_h)
        rw_c = _wrap_bias(rel_w)
        perm_q_c = _wrap_perm(hilbert_order)
        perm_k_c = _wrap_perm(hilbert_order)  # same perm for self-attention

        compiled = _get_fa2_compiled(
            q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c,
            win, self.scale, cu_stream, D, m_block, n_block, threads,
        )
        compiled(q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c, self.scale, cu_stream)

        o_orig = o[:, inv_hilbert, :, :].contiguous()
        return self.proj(o_orig.reshape(B, Sq, -1)).reshape(B, H, W, -1)


class ToMeSAMBlock(Block):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H_sp, W_sp, C = x.shape
        info  = self._tome_info
        ratio = info["ratio"].pop(0)

        shortcut = x
        x_n = self.norm1(x)
        if self.window_size > 0:
            ws = self.window_size
            H_w, W_w = x_n.shape[1], x_n.shape[2]
            x_n_win, pad_hw = window_partition(x_n, ws)
            x_attn = self.attn(x_n_win, ratio, use_fa2=True)
            x_attn = window_unpartition(x_attn, ws, pad_hw, (H_w, W_w))
        else:
            x_attn = self.attn(
                x_n, ratio,
                m_block=_FA2_M_BLOCK_GLOBAL,
                n_block=_FA2_N_BLOCK_GLOBAL,
                threads=_FA2_THREADS_GLOBAL,
            )

        x = shortcut + x_attn                         

        x_seq = x.reshape(B, H_sp * W_sp, C)          

        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        return x_seq.reshape(B, H_sp, W_sp, C)




def _warmup_fa2_kernels(encoder: ImageEncoderViT) -> None:

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

        compile_key = (win, D, m_block, n_block, threads)
        if compile_key in seen or not _fa2_can_implement(D, m_block, n_block, threads):
            seen.add(compile_key)
            continue
        seen.add(compile_key)

        Sq = win * win
        H  = attn.num_heads
        B  = 1  # batch dim is dynamic; B=1 is sufficient to drive compilation

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
        # Use Hilbert perm for warmup so the compiled signature matches inference
        hilbert_order, _ = _get_hilbert_perm(win, device)
        perm_q_c = _wrap_perm(hilbert_order)
        perm_k_c = _wrap_perm(hilbert_order)

        print(
            f"[ToMe-SAM] compiling FA2 kernel  global  "
            f"win={win}  D={D}  m={m_block}  n={n_block}  T={threads} ...",
            end=" ", flush=True,
        )
        _get_fa2_compiled(
            q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c,
            win, attn.scale, cu_stream,
            D, m_block, n_block, threads,
        )
        print("done")



def apply_patch(
    encoder: ImageEncoderViT,
    algo: str = "tome",
    ratio: float = 0.9,
    margin: float = 0.5,
    trace_source: bool = False,
) -> ImageEncoderViT:

    assert algo in ("tome", "pitome"), f"algo must be 'tome' or 'pitome', got {algo!r}"
    assert 0 < ratio <= 1.0, "ratio must be in (0, 1]"

    tome_info = {
        "algo":   algo,
        "ratio":  ratio,   
        "margin": margin,
        "x_attn": None,
        "metric": None 
    }
    encoder.tome_info = tome_info

    _orig_forward = encoder.__class__.forward

    def _patched_forward(self, x: torch.Tensor):
        n = len(self.blocks)
        r = self.tome_info["ratio"]
        self.tome_info["ratio"] = [r] * n

        result = _orig_forward(self, x)

        self.tome_info["ratio"] = r   # restore scalar for next call
        return result

    encoder.forward = types.MethodType(_patched_forward, encoder)

    for module in encoder.modules():
        if isinstance(module, Block) and not isinstance(module, ToMeSAMBlock):
            module.__class__  = ToMeSAMBlock
            module._tome_info = tome_info
        elif isinstance(module, Attention) and not isinstance(module, ToMeSAMAttention):
            module.__class__  = ToMeSAMAttention
            module._tome_info = tome_info

    n_blocks = len(encoder.blocks)
    n_global = sum(1 for blk in encoder.blocks if blk.window_size == 0)
    print(
        f"[ToMe-SAM] patched  algo={algo}  ratio={ratio}"
        + (f"  margin={margin}" if algo == "pitome" else "")
        + f"  blocks={n_blocks} (global={n_global} local={n_blocks-n_global})"
        + "  strategy=post-attn-merge / post-mlp-unmerge (all blocks)"
        + "  token-order=hilbert"
    )

    _warmup_fa2_kernels(encoder)

    return encoder
