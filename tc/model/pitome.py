
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers.modeling_utils import ModuleUtilsMixin 
from dct import dct, idct
import math



class CompressedModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, compress_method='dct', r=0.95):
        super().__init__()
        self.r = r
        self.compress_method = compress_method
        self.num_reduced_token = 12 
    

    def pitome(self, x: torch.Tensor, r: int=None, margin=None):
        # if margin>=0.6:
            # return self.bipartite_soft_matching(x, r)
       
        with torch.no_grad():
            B,T,_ = x.shape
            r = math.floor(T- T*self.r)
            batch_idx = torch.arange(B)[..., None].to(x.device)

            x = F.normalize(x, p=2, dim=-1)
            
            if margin is not None:
                sim = F.elu((x@x.transpose(-1,-2) - margin)/0.01)
            else:
                sim = x@x.transpose(-1,-2) 
            isolation_score = sim.mean(-1) 
            indices =  torch.argsort(isolation_score, descending=True)
            min_indices = indices[..., :2*r]
            protected_idx = indices[..., 2*r:]
            a_idx, b_idx = min_indices[..., :r], min_indices[..., r:]
            scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
            scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r))
            _, dst_idx = scores.max(dim=-1) 

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            B, _, C = x.shape

            # mask = torch.ones_like(x, dtype=torch.bool).to(x.device)
            # mask[batch_idx, min_indices] = False
            # protected = torch.masked_select(x, mask).view(B, -1, C)

            protected = x[batch_idx, protected_idx,:] 
            src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
            dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)
            return torch.cat([protected, dst], dim=1)

        isolation_score = (1 - F.softmax(isolation_score, dim=-1))[..., None]
        return merge, isolation_score  

    def bipartite_soft_matching(
        self,
        x: torch.Tensor,
        r: int=None,
    ):

        T = x.shape[1]
        r = math.floor(T- T*self.r)

        with torch.no_grad():
            x = F.normalize(x, p=2, dim=-1) 
            a, b = x[..., ::2, :], x[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)


        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            return torch.cat([unm, dst], dim=1)

        return merge, None

    def merge_wavg(
        self, merge, x: torch.Tensor, size: torch.Tensor = None
    ): 
        """
        Applies the merge function by taking a weighted average based on token size.
        Returns the merged tensor and the new token sizes.
        """
        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x = merge(x * size, mode="sum")
        size = merge(size, mode="sum")

        x = x / size
        return x, size 
    
    def merge_attention_mask(
        self, merge, attention_mask: torch.Tensor
    ): 

        attention_mask = merge(attention_mask, mode="amax")
        return attention_mask 


    def dc_transform(self, x):
        # cufft doesn't accept fp16
        x = x.permute(1,0,2)
        x_dct = dct(x.transpose(0,2), norm='ortho').transpose(0,2)
        T, B, C = x_dct.size()
        k = math.ceil(self.r * T)
        x_dct = x_dct[:k, :, :]
        x = idct(x_dct.transpose(0,2), norm='ortho').transpose(0,2)
        return x.permute(1,0,2), x_dct.permute(1,0,2)

    def compress_hidden_state(self, x:torch.Tensor, attention_mask:torch.Tensor, margin):

        if self.compress_method == 'dct':
            x, _ = self.dc_transform(x)
        elif self.compress_method == 'pitome':
            merge, w = self.pitome(x, margin=margin) 
            x, _ = self.merge_wavg(merge, x, w) 
            attention_mask = self.merge_attention_mask(merge, attention_mask) 
        elif self.compress_method == 'tome':
            merge, w = self.bipartite_soft_matching(x, None) 
            x, _ = self.merge_wavg(merge, x, w) 
            attention_mask = self.merge_attention_mask(merge, attention_mask) 

        return  x, attention_mask.squeeze_()

