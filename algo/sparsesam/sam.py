
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
_sam_root = os.path.normpath(os.path.join(_here, '..', '..', '..', 'sam-hq'))
if _sam_root not in sys.path:
    sys.path.insert(0, _sam_root)

_bsa_cute = os.path.normpath(os.path.join(_here, '..', '..', '..', 'cute'))
if _bsa_cute not in sys.path:
    sys.path.insert(0, _bsa_cute)
from flash_attn_rel_pos import FlashAttentionForwardAmpere

from segment_anything.modeling.image_encoder import (
    ImageEncoderViT,
    Block,
    Attention,
    window_partition,
    window_unpartition,
)
from .hilbert_utils import get_hilbert_inverse, get_hilbert_order
from .z_utils import get_z_inverse, get_z_order
from typing import Callable
# from .merge import get_z_order, get_z_inverse


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


_FA2_M_BLOCK_LOCAL = 64
_FA2_N_BLOCK_LOCAL = 64
_FA2_M_BLOCK_GLOBAL = 64
_FA2_N_BLOCK_GLOBAL = 64
_FA2_THREADS_LOCAL = 128
_FA2_THREADS_GLOBAL = 128

# A-mask = banded diagonal (width _DIAG_BAND_WIDTH) + vertical "keep" bar
# (first floor(ratio * num_n_blocks * _KEEP_BAR_SCALE) columns).
_DIAG_BAND_WIDTH = 1
_KEEP_BAR_SCALE  = 1.0

_FA2_DTYPE_FP16 = cutlass.dtype("Float16")
_SPARSE_MASK_DTYPE = cutlass.dtype("Int32")
_FA2_COMPILED: dict = {}
_FA2_CAN_IMPL: dict = {}
_HILBERT_CACHE: dict = {}
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


def make_A_mask(B, H, T, ratio, m_block, n_block,
                band_width: int = _DIAG_BAND_WIDTH,
                keep_bar_scale: float = _KEEP_BAR_SCALE,
                device="cuda"):
    num_m_blocks = math.ceil(T / m_block)
    num_n_blocks = math.ceil(T / n_block)

    t = torch.zeros(B, H, num_m_blocks, num_n_blocks, dtype=torch.int32, device=device)

    # Banded diagonal of width `band_width` (centered on the identity).
    half = band_width // 2
    for k in range(-half, band_width - half):
        if k == 0:
            i_m = torch.arange(min(num_m_blocks, num_n_blocks), device=device)
            t[:, :, i_m, i_m] = 1
        else:
            i_m = torch.arange(max(0, -k), min(num_m_blocks, num_n_blocks - k), device=device)
            t[:, :, i_m, i_m + k] = 1

    # Reduced vertical "keep" bar.
    n_keep_cols = int(ratio * num_n_blocks * keep_bar_scale)
    if n_keep_cols > 0:
        t[:, :, :, :(n_keep_cols - band_width +1 ) ] = 1

    ct = from_dlpack(t, assumed_align=4)
    return ct, t

def _get_fa2_compiled(
    B, H ,q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c,
    win, scale, cu_stream,
    D: int, m_block: int, n_block: int, threads: int,
    ratio:float,
    is_global: bool = False,
):
    key = (win, D, m_block, n_block, threads, ratio)
    band_width = _DIAG_BAND_WIDTH if is_global else 0
    # ratio = 1.0 if not is_global else ratio
    mask_key = (B, H, win, ratio, m_block, n_block, band_width, _KEEP_BAR_SCALE)
    if mask_key not in _SPARSE_MASK_CACHE:
        ct, t = make_A_mask(B, H, win**2, ratio, m_block, n_block, band_width=band_width)
        _SPARSE_MASK_CACHE[mask_key] = (ct, t)
        
    ct_mask, t_mask = _SPARSE_MASK_CACHE[mask_key]

    if key not in _FA2_COMPILED:
        # fa2 = FlashAttentionForwardAmpere(D, m_block, n_block, threads, win)
        _FA2_COMPILED[key] = cute.compile(
            FlashAttentionForwardAmpere(D, m_block, n_block, threads, win),
            q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c, ct_mask, scale, cu_stream,
        )
    return _FA2_COMPILED[key], ct_mask


def tile_stride_matching(
    x: torch.Tensor, H: int, W: int,
    ratio: float = 0.0,
    group_size: int = 4,
    n_block: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort Z-curve groups by per-group score; return (perm, inv_perm).

    Keep groups (grouped layout 111222...) followed by merge groups
    (interleaved 12341234...). Returned perms are int64, shape (B, N)."""
    N = H * W
    assert N % group_size == 0, f"N={N} must be divisible by group_size={group_size}"

    n_groups = N // group_size
    gs       = group_size

    with torch.no_grad():
        B, N, C = x.shape
        device  = x.device
        z_perm = get_z_order(H, W, device=device)
        x_z    = x[:, z_perm, :].float()
        x_grp  = x_z.view(B, n_groups, gs, C)

        grp_std  = x_grp.std(dim=2).mean(dim=-1)
        grp_mean = x_grp.mean(dim=2)
        gm_norm  = F.normalize(grp_mean, dim=-1)
        sim      = gm_norm @ gm_norm.transpose(1, 2)
        avg_sim  = (sim.sum(dim=-1) - 1.0) / max(n_groups - 1, 1)
        dissim   = -avg_sim

        eps = 1e-6
        std_n    = (grp_std - grp_std.mean(dim=-1, keepdim=True)) / (grp_std.std(dim=-1, keepdim=True) + eps)
        dissim_n = (dissim  - dissim.mean(dim=-1, keepdim=True))  / (dissim.std(dim=-1, keepdim=True)  + eps)
        grp_score = std_n + dissim_n

        grp_rank = grp_score.argsort(dim=-1, descending=True)
        group_raster = z_perm.view(n_groups, gs)
        all_raster = group_raster[grp_rank.reshape(-1)].reshape(B, n_groups, gs)
        perm_1d = all_raster.permute(0, 2, 1).reshape(B, N)
    inv_perm_1d = torch.argsort(perm_1d, dim=1)
    return perm_1d, inv_perm_1d


_STRIDE_PERM_CACHE: dict = {}


def tile_stride_matching_uniform(
    x: torch.Tensor, H: int, W: int,
    ratio: float = 0.0,
    group_size: int = 4,
    n_block: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stride-uniform variant: keep-set groups picked at uniform stride over
    the Z-order, so the global keep prefix has even spatial coverage."""
    B, N, _ = x.shape
    device  = x.device
    assert N == H * W and N % group_size == 0

    n_groups = N // group_size
    gs       = group_size
    num_n_blocks   = math.ceil(N / n_block)
    n_dense_blocks = int(ratio * num_n_blocks)
    n_keep_tokens  = min(n_dense_blocks * n_block, (N // gs) * gs)
    n_keep  = n_keep_tokens // gs
    n_merge = n_groups - n_keep

    cache_key = (H, W, ratio, group_size, n_block, str(device))
    cached = _STRIDE_PERM_CACHE.get(cache_key)
    if cached is None:
        z_perm = get_z_order(H, W, device=device)
        group_raster = z_perm.view(n_groups, gs)                       # (n_groups, gs)

        if n_keep > 0 and n_keep < n_groups:
            keep_idx = torch.round(
                torch.arange(n_keep, device=device, dtype=torch.float32)
                * (n_groups / n_keep)
            ).long().clamp_(0, n_groups - 1)
            mask = torch.ones(n_groups, dtype=torch.bool, device=device)
            mask[keep_idx] = False
            merge_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        elif n_keep == 0:
            keep_idx  = torch.empty(0, dtype=torch.long, device=device)
            merge_idx = torch.arange(n_groups, device=device)
        else:
            keep_idx  = torch.arange(n_groups, device=device)
            merge_idx = torch.empty(0, dtype=torch.long, device=device)

        keep_raster = group_raster[keep_idx].reshape(n_keep * gs)
        if n_merge > 0:
            merge_raster = group_raster[merge_idx].permute(1, 0).reshape(n_merge * gs)
            perm_1d_1 = torch.cat([keep_raster, merge_raster], dim=0)
        else:
            perm_1d_1 = keep_raster

        inv_1 = torch.argsort(perm_1d_1, dim=0)
        _STRIDE_PERM_CACHE[cache_key] = (perm_1d_1, inv_1)
        cached = _STRIDE_PERM_CACHE[cache_key]

    perm_1d_1, inv_1 = cached
    perm_1d     = perm_1d_1.unsqueeze(0).expand(B, -1).contiguous()
    inv_perm_1d = inv_1.unsqueeze(0).expand(B, -1).contiguous()
    return perm_1d, inv_perm_1d


class ToMeSAMAttention(Attention):

    def forward(self, x: torch.Tensor, ratio: float, use_fa2: bool = True,  # noqa: ARG002
                m_block: int = _FA2_M_BLOCK_LOCAL,
                n_block: int = _FA2_N_BLOCK_LOCAL,
                threads: int = _FA2_THREADS_LOCAL,
                custom_mask: "cute.Tensor | None" = None,
                is_global: bool = False,
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

        perm_cache = self._tome_info.setdefault("perm_cache", {})
        cache_key = (win, ratio, n_block)
        if cache_key not in perm_cache:
            perm_cache[cache_key] = tile_stride_matching(k, win, win, ratio=ratio, n_block=n_block)
        perm, inv_perm = perm_cache[cache_key]

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
            is_global=is_global,
        )
        mask = custom_mask if custom_mask is not None else default_mask
        compiled(q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c, mask, self.scale, cu_stream)
        inv_perm_e = inv_perm.unsqueeze(-1).expand(-1, -1, D)
        o_out = o.gather(1, inv_perm_e).reshape(B, self.num_heads, Sq, D).permute(0, 2, 1, 3).reshape(B, Sq, -1)
        out = self.proj(o_out).reshape(B, H, W, -1)
        if return_perm:
            return out, perm, inv_perm
        return out


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
            x_attn, perm, inv_perm = self.attn(
                x_n_win, ratio, use_fa2=True, custom_mask=None, return_perm=True,
            )
            x_attn = window_unpartition(x_attn, ws, pad_hw, (H_w, W_w))
        else:
            x_attn, perm, inv_perm = self.attn(
                x_n, ratio,
                m_block=_FA2_M_BLOCK_GLOBAL,
                n_block=_FA2_N_BLOCK_GLOBAL,
                threads=_FA2_THREADS_GLOBAL,
                is_global=True,
                return_perm=True,
            )

        x = shortcut + x_attn
        x_seq = x.reshape(B, H_sp * W_sp, C)

        mlp_merge = info.get("mlp_merge", True)
        if mlp_merge and ratio < 1.0 and self.window_size > 0:
            global_cached = info.get("perm_cache", {}).get((H_sp, ratio, _FA2_N_BLOCK_GLOBAL))
            if global_cached is not None:
                g_perm, g_inv_perm = global_cached
                nh = self.attn.num_heads
                keep_n = max(1, round(ratio * x_seq.shape[1]))
                # Avg head rank → topk avoids 2× full-N argsort+gather.
                avg_rank = g_inv_perm.view(B, nh, -1).float().mean(dim=1)
                top_idx  = avg_rank.topk(keep_n, dim=1, largest=False).indices
                idx_e    = top_idx.unsqueeze(-1).expand(-1, -1, C)
                x_kept   = x_seq.gather(1, idx_e)
                x_kept   = x_kept + self.mlp(self.norm2(x_kept))
                x_seq    = x_seq.scatter(1, idx_e, x_kept)
            else:
                x_seq = x_seq + self.mlp(self.norm2(x_seq))
        else:
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
        identity = torch.arange(Sq, device=q.device, dtype=torch.int32).unsqueeze(0)  # (1, Sq)
        perm_q_c = _wrap_perm(identity)
        perm_k_c = _wrap_perm(identity)

        print(
            f"[ToMe-SAM] compiling FA2 kernel  global  "
            f"win={win}  D={D}  m={m_block}  n={n_block}  T={threads}   ...",
            end=" ", flush=True,
        )
        _get_fa2_compiled(
            B, H, q_c, k_c, v_c, o_c, rh_c, rw_c, perm_q_c, perm_k_c,
            win, attn.scale, cu_stream,
            D, m_block, n_block, threads,
            ratio=0.5,
            is_global=True,
        )
        print("done")


def apply_patch(
    encoder: ImageEncoderViT,
    algo: str = "tome",
    ratio: float = 0.9,
    margin: float = 0.5,
    mlp_merge: bool = True,
    **_,
) -> ImageEncoderViT:
    """SparseSAM patch for the SAM-HQ image encoder.

    `mlp_merge`: True (default) runs the per-block MLP on the top
    floor(ratio*N) "keep" tokens only; False runs it on the full token
    set. `**_` swallows extras forwarded by the registry."""
    assert 0 < ratio <= 1.0, "ratio must be in (0, 1]"

    tome_info = {
        "ratio":     ratio,
        "mlp_merge": bool(mlp_merge),
    }
    encoder.tome_info = tome_info

    _orig_forward = encoder.__class__.forward

    def _patched_forward(self, x: torch.Tensor):
        n = len(self.blocks)
        r = self.tome_info["ratio"]
        self.tome_info["ratio"]      = [r] * n
        self.tome_info["perm_cache"] = {}   # reset per-image-forward

        result = _orig_forward(self, x)

        self.tome_info["ratio"] = r
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
        + f"  mlp_merge={tome_info['mlp_merge']}"
        + f"  blocks={n_blocks} (global={n_global} local={n_blocks-n_global})"
        + ("  strategy=post-attn-merge / post-mlp-unmerge"
           if tome_info['mlp_merge'] else "  strategy=full-MLP (no partial)")
    )

    _warmup_fa2_kernels(encoder)

    return encoder
