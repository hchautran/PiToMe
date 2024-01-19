
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import (
    BlipConfig, 
)
import math
from lavis import BlipRetrieval, Blip2Qformer
from transformers import BertForSequenceClassification 
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import ModuleUtilsMixin 
from dct import dct, idct
from typing import Union, Tuple



class CompressedModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, compress_method='dct', r=0.95, window_size=2):
        super().__init__()
        self.r = r
        self.window_size=window_size
        self.compress_method = compress_method
        self.num_reduced_token = 12 
    

    def std_filter_with_r(self, x, k = 2):        
        B, T, D = x.shape
        batch_idx = torch.arange(B).unsqueeze(1)
        min_indices = torch.argsort(x.std(-1),dim=-1)
        x = x[batch_idx, min_indices, :]

        if k is None or k * 2 > T:
            k = math.floor(T- T*self.r)
        else:
            k = k 

        first_x = x[:,self.window_size*(T//self.window_size):,:]
        remain_x = x[:,:self.window_size*(T//self.window_size),:]
        remain_x = remain_x.view(B, -1, self.window_size, D)
        # print(remain_x.shape)
        std_array = remain_x.std(-1)
        max_std = std_array.max(-1)[0] 
    
        # min_std_array, min_indices = torch.topk(max_std, k=k, dim=-1, largest=False)
        with torch.no_grad():
            min_indices = torch.argsort(max_std, dim=-1)[:, :k]
            mask_to_keep = torch.ones_like(remain_x, dtype=torch.bool)
            mask_to_keep[batch_idx, min_indices, :, :] = False
        # print(min_std_array.shape)
        filtered_tensor = torch.masked_select(remain_x, mask_to_keep).view(remain_x.size(0), -1, remain_x.size(2), remain_x.size(3))
        reduced_tensor = remain_x[batch_idx, min_indices, :, :]
        reduced_tensor = reduced_tensor.mean(dim=2)
        output = torch.cat([first_x, filtered_tensor.view(B,-1,D), reduced_tensor.view(B,-1,D)], dim=1)

        return output, None 

    
    def std_based(self, x:torch.Tensor, k:int=None):        
        B, T, D = x.shape
        if k is None:
            k = math.floor(T- T*self.r)
    
        with torch.no_grad():
            std_array = x.std(-1)
            batch_idx = torch.arange(x.shape[0]).unsqueeze(1)
            min_indices = torch.argsort(std_array,dim=-1)[:, :2*k ]

            mask_to_keep = torch.ones_like(x, dtype=torch.bool)
            mask_to_keep[batch_idx, min_indices,  :] = False

        filtered_tensor = torch.masked_select(x, mask_to_keep).view(x.size(0), -1, x.size(2))
        reduced_tensor = x[batch_idx, min_indices, :]
        reduced_tensor = (reduced_tensor[..., k:,:] + reduced_tensor[..., :k,:].flip([1])) /2
        # reduced_tensor =  reduced_tensor.view(B, -1, 2, D).mean(dim=2)
        # print(filtered_tensor.shape)

        output = torch.cat([filtered_tensor, reduced_tensor], dim=1)
      
        return output, None 
    
    def pitome(self, x: torch.Tensor, r: int=None, margin:float=0.5):
        B,T,C = x.shape
        r = math.floor(T- T*self.r)
        with torch.no_grad():
            batch_idx = torch.arange(B).unsqueeze_(1)
            x = F.normalize(x, p=2, dim=-1)
            x_std = x.std(-1, keepdim=True)
            ori_score =x@x.transpose(-1,-2) 
            ori_score = torch.where(ori_score > margin, ori_score - margin, -1.0 * x_std)
            min_indices =  torch.argsort(ori_score.mean(dim=-2), descending=True)[..., :2*r]
            mask_to_keep = torch.ones_like(x, dtype=torch.bool).to(x.device)
            mask_to_keep[batch_idx, min_indices,  :] = False
            a_idx, b_idx = min_indices[..., ::2], min_indices[..., 1::2]
            a, b = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
            scores = a@b.transpose(-1,-2) 
            _, dst_idx = scores.max(dim=-1) 

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            ori = torch.masked_select(x, mask_to_keep).view(B, -1, C)
            src, dst = x[batch_idx, a_idx, :], x[batch_idx,  b_idx, :]
            dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)
            return torch.cat([ori, dst], dim=1)

        return merge
    

    def bipartite_soft_matching(
        self,
        x: torch.Tensor,
        r: int=None,
    ):
        T = x.shape[1]

        protected = 0
        if r is None:
            r = math.floor(T- T*self.r)
            # print(r)
    
        # We can only reduce by a maximum of 50% tokens
        r = min(r, (T - protected) // 2)

        if r <= 0:
            return x, x

        with torch.no_grad():
            x = F.normalize(x, p=2, dim=-1) 
            a, b = x[..., ::2, :], x[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)

       

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
            # print(node_max)

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

        x = merge(x * size, mode="mean")
        # size = merge(size, mode="sum")

        # x = x / size
        return x, None 
            
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
            merge = self.pitome(x, None) 
            x_reconstructed, energy = self.merge_wavg(merge, x) 
        elif self.compress_method == 'std-mean-merge':
            x_reconstructed, energy = self.std_filter_with_r(x,k=None) 
        elif self.compress_method == 'tome':
            merge = self.bipartite_soft_matching(x, None) 
            x_reconstructed, energy = self.merge_wavg(merge, x) 
        else: 
            return x, x

        return  x_reconstructed, energy

    
class CompressedBERT(CompressedModel):
    config_class = BlipConfig

    def __init__(self, model:BertForSequenceClassification, compress_method='dct', r=0.9):
        super(CompressedBERT, self).__init__(compress_method, r=r)
        self.model = model.bert
        self.dropout = model.dropout 
        self.classifier = model.classifier
        self.config = model.config

        self.compress_layers = [i for i in range(1, len(self.model.encoder.layer)-1)]
        self.model_len = len(self.model.encoder.layer) 
     
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = []
        energy = []
        real_mem = 0
        total_mem = 0
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False


        _, seq_length = input_ids.shape 
        ori_size = seq_length

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)

        hidden_states = self.model.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.model.encoder.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i in self.compress_layers:    
                cls = hidden_states[:, 0, :].unsqueeze(1)
                state, cur_energy = self.compress_hidden_state(
                    hidden_states[:, 1:, :], 
                    use_compressed_hidden_state=True,
                    margin=(0.5 if i< self.model_len//2 else  0.5-0.5*i/self.model_len)
                )
                hidden_states = torch.cat([cls, state], dim=1)
                real_mem += hidden_states.shape[1]
                total_mem += ori_size 
                attention_mask =  attention_mask[:,:,:, :hidden_states.shape[1]]
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                )
            else:

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

      
        pooled_output = self.model.pooler(hidden_states) if self.model.pooler is not None else None

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
       
        return (
            hidden_states,
            logits,
            all_hidden_states,
            all_self_attentions,
        )


