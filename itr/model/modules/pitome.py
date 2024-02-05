
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .utils import dct, idct 
from typing import Callable, Tuple
import math
EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'

from .vis import make_visualization

class CompressedModel(nn.Module):
    def __init__(self, compress_method='dct', r=0.95, window_size=2):
        super().__init__()
        self.r = r
        self.window_size=window_size
        self.compress_method = compress_method
        self.num_reduced_token = 32 
        self.temp = nn.Parameter(torch.tensor(0.01))
    
    def do_nothing(self, x, mode='none'):
        return x

    def dist_func(self, x:torch.Tensor, y:torch.Tensor): 
        dis = 0
        x = F.normalize(x,p=2, dim=-1) 
        y = F.normalize(y,p=2, dim=-1) 
        dis = torch.matmul(x, y.transpose(-1,-2)) 
        return dis 
    

    def bipartite_soft_matching(
        self,
        x: torch.Tensor,
        r: int=None,
    ):
        if self.r == 1.0:
            return self.do_nothing
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

        return merge

    def estimate_flop(self, shape:tuple):
        with torch.no_grad():
            flops = 0
            _, T, C = shape 
            mhsa_flops = 4*T*C*C + 2*T*T*C
            ffn_flops = 8*T*C*C
            flops = mhsa_flops + ffn_flops 
            return flops
    
    def do_nothing(self, x, mode=None):
        return x

    def pitome(self, x: torch.Tensor, r: int=None, margin:torch.Tensor=None):

        if self.r == 1.0:
            return self.do_nothing, None

        if margin >= 0.45 :
            return self.bipartite_soft_matching(x, r), None

        with torch.no_grad():
            B,T,_ = x.shape
            r = math.floor(T- T*self.r)
            batch_idx = torch.arange(B, pin_memory=True)[..., None].to(x.device)
            x = F.normalize(x, p=2, dim=-1)

        sim =x@x.transpose(-1,-2)
        isolation_score = F.softmin(sim, dim=-1).mean(-2) 

        with torch.no_grad():
            indices =  torch.argsort(isolation_score)
            merge_idx = indices[..., :2*r]
            protected_idx = indices[..., 2*r:]
            a_idx, b_idx = merge_idx[..., ::2], merge_idx[...,1::2]
            scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
            scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r))
            _, dst_idx = scores.max(dim=-1) 

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            B, _, C = x.shape
            protected = x[batch_idx, protected_idx,:] 
            src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :] 
            dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)
            return torch.cat([protected, dst], dim=1)


        # isolation_score = 1 - F.softmax(isolation_score, dim=-1)
        return merge, isolation_score[..., None] 
    
    def merge_wavg(
        self, merge: Callable, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the merge function by taking a weighted average based on token size.
        Returns the merged tensor and the new token sizes.
        """
        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x = merge(x*size, mode="sum")
        size = merge(size, mode="sum")
        x = x / size

        return x
    
    def merge_source(
        self, merge: Callable, x: torch.Tensor, source: torch.Tensor = None
    ) -> torch.Tensor:
        """
        For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
        x is used to find out how many tokens there are in case the source is None.
        """
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
   
    def get_vision_features(self, pixel_values, return_attention_map=False):
        raise NotImplementedError("This method is not implemented yet")

    def get_text_features(self, input_ids, attention_mask):
        raise NotImplementedError("This method is not implemented yet")
    
    def compress_hidden_state(self, x, return_source=False, margin=0.5, source=None ):
        if self.compress_method == 'dct':
            x_reconstructed = self.dc_transform(x)
        elif self.compress_method == 'PiToMe':
            merge, isolation_score = self.pitome(x, None, margin=margin) 
            x_reconstructed = self.merge_wavg(merge, x, isolation_score) 
        elif self.compress_method == 'ToMe':
            merge = self.bipartite_soft_matching(x, None) 
            x_reconstructed = self.merge_wavg(merge, x) 
        else:
            return x, x
        if return_source:
            source = self.merge_source(merge, x, source)

        return  x_reconstructed, source

    def make_visualization(self, img, source, attention_score, patch_size=16, class_token=True):
        return make_visualization(img, source, patch_size=patch_size, class_token=class_token, attention_score=attention_score)

    def forward(
        self,
        input_ids: torch.LongTensor=None,
        pixel_values: torch.FloatTensor=None,
        attention_mask: Optional[torch.LongTensor] = None,
        
    ):
        if input_ids is not None:
            return self.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        else:
            return self.get_vision_features(pixel_values=pixel_values)
        


