
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
import numpy as np
import math
PITOME = 'pitome'
TOME = 'tome'
TOFU = 'tofu'
LTMP = 'ltmp'
TPS = 'tps'
DIFFRATE = 'diffrate'
DCT = 'dct'
NONE = 'none'


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """

    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v, dim=1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.view_as_complex(V)

    v = torch.fft.ifft(V, dim=1).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

def dc_transform(x, r=0.9):
    x = x.type(torch.float32).permute(1,0,2)
    x_dct = dct(x.transpose(0,2), norm='ortho').transpose(0,2)
    T, B, C = x_dct.size()

    x_dct = x_dct[:math.ceil(T * r), :, :]

    return idct(x_dct.transpose(0,2), norm='ortho').transpose(0,2).type(torch.half).permute(1,0,2), x_dct.permute(1,0,2)


class CompressedBlock(nn.Module):
    def __init__(self, compress_method='dct', r=0.95, k=2, use_k=False):
        super().__init__()
        self.r = r
        self.k = k
        self.use_k = use_k
        self.compress_method = compress_method
    
    def do_nothing(self, x, mode='none'):
        return x

    def bipartite_soft_matching(
        self, 
        metric: torch.Tensor,
        class_token: bool = False,
        distill_token: bool = False,
    ) -> Tuple[Callable, Callable]:
     
        protected = 0
        if class_token:
            protected += 1
        if distill_token:
            protected += 1

        # We can only reduce by a maximum of 50% tokens
        T = metric.shape[1]
     
        if self.use_k:
            k = min(self.k, T // 2)
        else:
            k = math.floor(T- T*self.r)


        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)

            if class_token:
                scores[..., 0, :] = -math.inf
            if distill_token:
                scores[..., :, 0] = -math.inf

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., k:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :k, :]  # Merged Tokens
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

            if class_token:
                unm_idx = unm_idx.sort(dim=1)[0]

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - k, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, k, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, k, c), src, reduce=mode)

            if distill_token:
                return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
            else:
                return torch.cat([unm, dst], dim=1)


        return merge, None

    def pitome(
        self,
        metric: torch.Tensor, 
        margin:torch.Tensor=0.5,
        class_token: bool = False,
    ):
        with torch.no_grad():
            if class_token:
                metric=metric[:,1:,:]
            B,T,C = metric.shape

            if self.use_k:
                k = min(self.k, T // 2)
            else:
                k = math.floor(T- T*self.r)

            if margin >=0.45:
                return self.bipartite_soft_matching(metric=metric, class_token=class_token)
            batch_idx = torch.arange(B).unsqueeze_(1).to(metric.device)

        sim = F.elu((F.cosine_similarity(metric) - margin)/0.001)
        isolation_score = sim.mean(dim=-1)

        with torch.no_grad():
            indices =  torch.argsort(isolation_score, descending=True)
            merge_idx = indices[..., :2*k]
            protected_idx = indices[..., 2*k:]
            a_idx, b_idx = merge_idx[..., ::2], merge_idx[..., 1::2]

            scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, k)) 
            scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, k, k))

            _, dst_idx = scores.max(dim=-1) 
        
        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            if class_token:
                x_cls=x[:,0,:].unsqueeze(1)
                x=x[:,1:,:]
            else:
                x_cls = None

            B, _, C = x.shape
            protected = x[batch_idx, protected_idx, :]
            src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
            dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, k, C), src, reduce=mode)

            if x_cls is not None:
                return torch.cat([x_cls, protected, dst], dim=1)
            else:
                return torch.cat([protected, dst], dim=1)

        isolation_score = 1 - F.softmax(isolation_score, dim=-1) 

        if class_token:
            return merge, torch.cat([torch.ones(B, 1).to(metric.device), isolation_score], dim=-1)[..., None]
        return merge, isolation_score[..., None] 
    
    def merge_wavg(
        self, merge: Callable, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x = merge(x*size, mode="sum")
        size = merge(size, mode="sum")
        x = x / size
        return x
    
    def merge_source(
        self, merge: Callable, x: torch.Tensor, source: torch.Tensor = None
    ) -> torch.Tensor:
       
        if source is None:
            n, t, _ = x.shape
            source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

        source = merge(source, mode="amax")
        return source

    def dc_transform(self, x):
        # cufft doesn't accept fp16
        x = x.permute(1,0,2)
        x_dct = dct(x.transpose(0,2), norm='ortho').transpose(0,2)
        T, B, C = x_dct.size()
        k = math.ceil(self.r * T)
        x_dct = x_dct[:k, :, :]
        x = idct(x_dct.transpose(0,2), norm='ortho').transpose(0,2)
        return x.permute(1,0,2)
    
    def compress_hidden_state(self, metric, x, trace_source=False, margin=0.5, source=None, class_token=True ):
        if self.compress_method == NONE:
            return x

        if (self.use_k and self.r < 1.0) or (self.use_k and self.k > 0):
            if self.compress_method == DCT:
                x = self.dc_transform(x)
            else:
                if self.compress_method == PITOME:
                    merge, isolation_score = self.pitome(metric, margin=margin, class_token=class_token) 
                elif self.compress_method == TOME:
                    merge = self.bipartite_soft_matching(metric) 


                if isolation_score is not None and self._tome_info["size"] is not None:
                    x, self._tome_info["size"] = self.merge_wavg(merge, x, isolation_score + self._tome_info["size"])
                else:
                    x, self._tome_info["size"] = self.merge_wavg(merge, x, self._tome_info["size"])
                if trace_source:
                    self._tome_info["source"] = self.merge_source(merge, x, source)

        return  x



