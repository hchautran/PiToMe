
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers.modeling_utils import ModuleUtilsMixin 
from dct import dct, idct
import math



class CompressedModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, compress_method='dct', r=0.95, window_size=2):
        super().__init__()
        self.r = r
        self.window_size=window_size
        self.compress_method = compress_method
        self.num_reduced_token = 12 
    
    

    def pitome(self, x: torch.Tensor, r: int=None, margin:float=0.5):
        # if margin >= 0.45 :
            # return self.bipartite_soft_matching(x, r)
        with torch.no_grad():
            B,T,_ = x.shape
            r = math.floor(T- T*self.r)
            batch_idx = torch.arange(B)[..., None].to(x.device)

            x = F.normalize(x, p=2, dim=-1)
            sim =x@x.transpose(-1,-2) 
            isolation_score = torch.where(sim > margin, 1.0, -1.0).mean(dim=-1) + sim.mean(-1)
            # isolation_score = sim.mean(dim=-1)

            isolation_score = 1 - F.softmax(isolation_score, dim=-1)
            indices =  torch.argsort(isolation_score)
            min_indices = indices[..., :2*r]
            protected_idx = indices[..., 2*r:]

            a_idx, b_idx = min_indices[..., ::2], min_indices[..., 1::2]
            scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
            scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r))
    #         scores = a@b.transpose(-1,-2) 
            _, dst_idx = scores.max(dim=-1) 

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            B, _, C = x.shape

            mask_to_keep = torch.zeros_like(x, dtype=torch.bool).to(x.device)
            mask_to_keep[batch_idx, protected_idx] = True
            protected = torch.masked_select(x, mask_to_keep).view(B, -1, C)

            # protected = x[batch_idx, protected_idx,:] 
            src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
            dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)
            return torch.cat([protected, dst], dim=1)

        return merge, isolation_score[..., None]
    
    

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

        return merge



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
            
    def random_filter_with_r(self, x, use_mean=False, k = 2):        
        B, T, D = x.shape
        with torch.no_grad():
            if k is None:
                k = math.floor((T- T*self.r)/2)
            batch_idx = torch.arange(x.shape[0]).unsqueeze(1)

            first_x = x[:,:(T%self.window_size),:]
            remain_x = x[:,(T%self.window_size):,:]
            remain_x = remain_x.view(B, -1, 2, D)
       
            min_indices = torch.randint(0, remain_x.shape[1], (1, k)).squeeze(0)
            mask_to_keep = torch.ones_like(remain_x, dtype=torch.bool)
            mask_to_keep[batch_idx, min_indices, :, :] = False

            filtered_tensor = torch.masked_select(remain_x, mask_to_keep).view(remain_x.size(0), -1, remain_x.size(2), remain_x.size(3))
            reduced_tensor = remain_x[batch_idx, min_indices, :, :].mean(dim=2, keepdim=True)
            output = torch.cat([first_x, filtered_tensor.view(B,-1,D), reduced_tensor.view(B,-1,D).mean(dim=1, keepdim=True)], dim=1)
      
        return output, None 
    
    
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        pixel_values: torch.FloatTensor=None,
        attention_mask: Optional[torch.LongTensor] = None,
        use_compressed_hidden_state: Optional[torch.LongTensor] = True,
        
    ):
        if input_ids is not None:
            return self.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        else:
            return self.get_vision_features(pixel_values=pixel_values, use_compressed_hidden_state=use_compressed_hidden_state)

    def dc_transform(self, x, use_reconstucted_state=False, threshold=None):
        # cufft doesn't accept fp16
        x = x.permute(1,0,2)
        x_dct = dct(x.transpose(0,2), norm='ortho').transpose(0,2)
        T, B, C = x_dct.size()
        k = math.ceil(self.r * T)

        if use_reconstucted_state:
            x_dct = x_dct[:k, :, :]
            x = idct(x_dct.transpose(0,2), norm='ortho').transpose(0,2)
            # print(x)
   
        return x.permute(1,0,2), x_dct.permute(1,0,2)

    def direct(self, x, use_reconstucted_state = False):
        k = math.ceil(0.90 * x.shape[1])
        if use_reconstucted_state:
            x = x[:,:k,:]  
        return x, x
    
    def std_based_compress(self, x, use_reconstucted_state, threshold=0.7,filter_strategy='std'):
        if use_reconstucted_state:
            x = self.std_filter(x, threshold, filter_strategy=filter_strategy) 
        return x, x
   
    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_hidden_state=False):
        raise NotImplementedError("This method is not implemented yet")

    def get_text_features(self, input_ids, attention_mask):
        raise NotImplementedError("This method is not implemented yet")
    
    def compress_hidden_state(self, x, use_compressed_hidden_state, margin=0.5):
        if self.compress_method == 'dct':
            x_reconstructed, energy = self.dc_transform(x ,use_compressed_hidden_state ) 
        elif self.compress_method == 'random-mean-merge':
            # x_reconstructed, energy = self.random_filter_with_r(x, k=self.num_reduced_token, use_mean=True) 
            x_reconstructed, energy = self.random_filter_with_r(x, k=None)  
        elif self.compress_method == 'random-std-merge':
            # x_reconstructed, energy = self.random_filter_with_r(x, k=self.num_reduced_token, use_mean=False) 
            x_reconstructed, energy = self.random_filter_with_r(x, k=None) 
        elif self.compress_method == 'pitome':
            merge, w = self.pitome(x, margin=margin) 
            x_reconstructed, energy = self.merge_wavg(merge, x, w) 
        elif self.compress_method == 'tome':
            merge = self.bipartite_soft_matching(x, None) 
            x_reconstructed, energy = self.merge_wavg(merge, x) 
        else: 
            return x, x

        return  x_reconstructed, energy

