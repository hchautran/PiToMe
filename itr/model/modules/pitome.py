
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import (
    BlipConfig, 
)
import math
from lavis import BlipRetrieval, Blip2Qformer
from transformers import AutoModel 
from .utils import dct, idct 
from typing import Callable, Tuple
EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'



class CompressedModel(nn.Module):
    def __init__(self, compress_method='dct', r=0.95, window_size=2, manifold=None):
        super().__init__()
        self.r = r
        self.window_size=window_size
        self.compress_method = compress_method
        self.num_reduced_token = 32 
        self.temp = nn.Parameter(torch.tensor(0.01))
    
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

    def pitome(self, x: torch.Tensor, r: int=None, margin:torch.Tensor=0.5):

        if margin >= 0.45 :
            return self.bipartite_soft_matching(x, r), None

        with torch.no_grad():
            B,T,_ = x.shape
            r = math.floor(T- T*self.r)
            batch_idx = torch.arange(B, pin_memory=True)[..., None].to(x.device)
            x = F.normalize(x, p=2, dim=-1)
            sim =x@x.transpose(-1,-2) 
            margin = margin.clamp_(min=0.0, max=0.9)
            temp= self.temp.clamp(min=0.001, max=0.05)
        sim = F.elu((sim - margin)/temp)
        isolation_score = sim.mean(dim=-1) 
        with torch.no_grad():
            indices =  torch.argsort(isolation_score, descending=True)
            merge_idx = indices[..., :2*r]
            protected_idx = indices[..., 2*r:]
            a_idx, b_idx = merge_idx[..., :r], merge_idx[...,r:]
            scores = sim.gather(dim=-1, index=b_idx.unsqueeze(-2).expand(B, T, r)) 
            scores = scores.gather(dim=-2, index=a_idx.unsqueeze(-1).expand(B, r, r))
            _, dst_idx = scores.max(dim=-1) 

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            B, _, C = x.shape
            protected = x[batch_idx, protected_idx,:] 
            src, dst = x[batch_idx, a_idx, :], x[batch_idx, b_idx, :] 
            dst = dst.scatter_reduce(-2, dst_idx.unsqueeze(2).expand(B, r, C), src, reduce=mode)
            return torch.cat([protected, dst], dim=1)


        isolation_score = 1 - F.softmax(isolation_score, dim=-1)
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
   
        return x.permute(1,0,2)

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
            x_reconstructed = self.dc_transform(x ,use_compressed_hidden_state ) 
        elif self.compress_method == 'PiToMe':
            merge, isolation_score = self.pitome(x, None, margin=margin) 
            x_reconstructed = self.merge_wavg(merge, x, isolation_score) 
        elif self.compress_method == 'ToMe':
            merge = self.bipartite_soft_matching(x, None) 
            x_reconstructed = self.merge_wavg(merge, x) 
        else: 
            return x

        return  x_reconstructed

    
class CompressedHFBLIP(CompressedModel):
    config_class = BlipConfig

    def __init__(self, model:AutoModel, compress_method='dct', r=0.9):
        super(CompressedHFBLIP, self).__init__(compress_method, r=r)
        self.vision_model = model.vision_model
        self.text_model = model.text_model 
        self.vision_proj = model.visual_projection 
        self.text_proj = model.text_projection 
        self.compress_layers = [i for i in range(1,len(self.vision_model.blocks))]
        self.model_len = len(self.vision_model.blocks)
        self.margins = nn.ParameterList([0.9 - i/self.model_len * 0.9 for i in range(self.model_len)])

    
    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_hidden_state=False):
        hidden_states = self.vision_model.embeddings(pixel_values)
        all_hidden_states = []
        real_mem = 0
        total_mem = 0
        flop = 0
        ori_size = hidden_states.shape[1]

        for i, layer in enumerate(self.vision_model.encoder.layers):
            if i in self.compress_layers:    
                cls = hidden_states[:, 0, :].unsqueeze(1)
                state = self.compress_hidden_state(
                    hidden_states[:, 1:, :], 
                    use_compressed_hidden_state=use_compressed_hidden_state,
                    margin=self.margins[i]
                )
                hidden_states = torch.cat([cls, state], dim=1)

            if return_all_hidden_state or i == len(self.vision_model.encoder.layers)-1:
                all_hidden_states.append(hidden_states)
            real_mem += hidden_states.shape[1]
            total_mem += ori_size 

            flop += self.estimate_flop(hidden_states.shape)

            hidden_states = layer(
                hidden_states,
                None,
                None
            )[0]


        last_hidden_state = self.vision_model.post_layernorm(hidden_states)
        pooled_output = last_hidden_state[:, 0, :]
        vision_embed = self.vision_proj(pooled_output)
       
        return hidden_states, vision_embed, all_hidden_states, flop, real_mem/total_mem

    def get_text_features(self, input_ids, attention_mask):
        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = text_output[0] 
        text_embed = self.text_proj(text_output[1])

        return  last_hidden_state, text_embed


class CompressedLAVISBLIP(CompressedModel):

    def __init__(self, model:BlipRetrieval, compress_method='dct',r=0.9):
        super(CompressedLAVISBLIP, self).__init__(compress_method, r=r)

        self.vision_model = model.visual_encoder
        self.text_model = model.text_encoder 
        self.vision_proj = model.vision_proj 
        self.text_proj = model.text_proj 
        self.compress_layers = [i for i in range(1,len(self.vision_model.blocks))]
        self.model_len = len(self.vision_model.blocks)
        self.margins = nn.ParameterList([
            nn.Parameter(torch.tensor(0.75 - i/self.model_len * 0.75), requires_grad=True) for i in range(self.model_len)])

   
    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_hidden_state=False):
        B = pixel_values.shape[0]
        x = self.vision_model.patch_embed(pixel_values)
        hidden_states = []
        cls_tokens = self.vision_model.cls_token.expand(
            B, -1, -1
        ) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.vision_model.pos_embed[:, : x.size(1), :]
        x = self.vision_model.pos_drop(x)
        ori_size = x.shape[1]
        real_mem = 0
        total_mem = 0
        flop = 0
        for i, blk in enumerate(self.vision_model.blocks):
            if i in self.compress_layers: 
                cls = x[:, 0, :].unsqueeze(1)
                state = self.compress_hidden_state(
                    x[:, 1:, :], 
                    use_compressed_hidden_state=use_compressed_hidden_state,
                    margin = self.margins[i]
                )
                x = torch.cat([cls, state], dim=1)

            if return_all_hidden_state or i == len(self.vision_model.blocks)-1:
                hidden_states.append(state)
            real_mem += x.shape[1]
            total_mem += ori_size 

            flop += self.estimate_flop(x.shape)
            x = blk(x)

        # with torch.no_grad():
        x = self.vision_model.norm(x)
        vision_embed = self.vision_proj(x[:,0,:])
        return x, vision_embed, hidden_states, flop, real_mem/total_mem

    def get_text_features(self, input_ids, attention_mask):
        # with torch.no_grad():
        class Text(object):
            pass
        text = Text() 
        text.input_ids=input_ids
        text.attention_mask=attention_mask
        text_output = self.text_model.forward_text(text)
        last_hidden_state = text_output[0] 
        text_embed = self.text_proj(last_hidden_state[:,0,:])

        return  last_hidden_state, text_embed


class CompressedHFCLIP(CompressedModel):

    def __init__(self, model:AutoModel, compress_method='dct',r=0.9):
        super(CompressedHFCLIP, self).__init__(compress_method, r=r)

        self.vision_model = model.vision_model
        self.text_model = model.text_model 
        self.vision_proj = model.visual_projection 
        self.text_proj = model.text_projection 
        self.compress_layers = [i for i in range(1,len(self.vision_model.encoder.layers))]
        self.model_len = len(self.vision_model.encoder.layers)
        self.margins = nn.ParameterList([0.9 - i/self.model_len * 0.9 for i in range(self.model_len)])

    def get_vision_features(self, pixel_values, use_compressed_hidden_state=True, return_all_hidden_state=False):
        all_hidden_states = []
        hidden_states = self.vision_model.embeddings(pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)
        real_mem = 0
        total_mem = 0
        flop = 0
        ori_size = hidden_states.shape[1]
        for i, layer in enumerate(self.vision_model.encoder.layers):
            if i in self.compress_layers:
                cls = hidden_states[:, 0, :].unsqueeze(1)
                state = self.compress_hidden_state(
                    hidden_states[:, 1:, :], 
                    use_compressed_hidden_state=use_compressed_hidden_state,
                    margin=self.margins[i]
                )
                hidden_states = torch.cat([cls, state], dim=1)
                # print(hidden_states.shape)
            if return_all_hidden_state or i == len(self.vision_model.encoder.layers)-1:
             
                all_hidden_states.append(hidden_states)

            real_mem += hidden_states.shape[1]
            total_mem += ori_size 
            flop += self.estimate_flop(hidden_states.shape)

            hidden_states = layer(
                hidden_states,
                None,
                None
            )[0]

        last_hidden_state = hidden_states
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)
        vision_embed = self.vision_proj(pooled_output)

        return hidden_states, vision_embed, all_hidden_states, flop, real_mem/total_mem

    def get_text_features(self, input_ids, attention_mask):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = text_outputs[1]
        text_embed = self.text_proj(pooled_output)

        return  text_outputs[0], text_embed

        
class CompressedLAVISBLIP2(CompressedModel):

    def __init__(self, model:Blip2Qformer, compress_method='dct',r=0.9):
        super(CompressedLAVISBLIP2, self).__init__(compress_method,r=r)

        self.ln_vision = model.ln_vision
        self.visual_encoder = model.visual_encoder
        self.query_tokens = model.query_tokens
        self.vision_proj = model.vision_proj
        self.text_proj = model.text_proj
        self.Qformer = model.Qformer
        self.itm_head = model.itm_head
        self.model_len = len(self.visual_encoder.blocks)
        self.margins = nn.ParameterList([0.75 - i/self.model_len * 0.5 for i in range(self.model_len)])
        self.compress_layers = [i for i in range(1, self.model_len)]

   
    def get_vision_features(self, pixel_values:torch.Tensor, use_compressed_hidden_state=True, return_all_hidden_state=False):
        all_hidden_states = []
       
        total_mem=0
        real_mem=0
        flop = 0
        with torch.no_grad():
            x = self.visual_encoder.patch_embed(pixel_values.squeeze(0))
            batch_size, seq_len, _ = x.size()

            cls_tokens = self.visual_encoder.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            if self.visual_encoder.pos_embed is not None:
                x = x + self.visual_encoder.pos_embed
            x = self.visual_encoder.pos_drop(x)
            ori_size = x.shape[1]

            rel_pos_bias = self.visual_encoder.rel_pos_bias() if self.visual_encoder.rel_pos_bias is not None else None
            for i, blk in enumerate(self.visual_encoder.blocks):
                if i in self.compress_layers:
                    x = self.compress_hidden_state(
                        x, 
                        use_compressed_hidden_state=use_compressed_hidden_state,
                        margin=self.margins[i]
                    )
                x = blk(x, rel_pos_bias)
                if return_all_hidden_state or i == len(self.visual_encoder.blocks) - 1:
                    all_hidden_states.append(x)
                real_mem += x.shape[1]
                total_mem += ori_size 
                flop += self.estimate_flop(x.shape)

            vit_embeds = self.ln_vision(x)

        image_atts = torch.ones(vit_embeds.size()[:-1], dtype=torch.long).to(
            pixel_values.device
        )
        query_tokens = self.query_tokens.expand(vit_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=vit_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        pooled_output = self.vision_proj(query_output.last_hidden_state)
        return vit_embeds, pooled_output, all_hidden_states, flop, real_mem/total_mem 

    def get_text_features(self, input_ids, attention_mask):
        # with torch.no_grad():
        text_output = self.Qformer.bert(
            input_ids=input_ids.squeeze(),
            attention_mask=attention_mask.squeeze(),
            return_dict=True,
        )

        pooled_output = self.text_proj(text_output.last_hidden_state[:, 0, :])
        return text_output.last_hidden_state, pooled_output