
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BlipConfig, 
)
from transformers import AutoModel 
from .pitome import CompressedModel
from lavis import BlipRetrieval
EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'






class CompressedLAVISBLIP(CompressedModel):

    def __init__(self, model:BlipRetrieval, compress_method='dct',r=0.9):
        super(CompressedLAVISBLIP, self).__init__(compress_method, r=r)

        self.vision_model = model.visual_encoder
        self.text_model = model.text_encoder 
        self.vision_proj = model.vision_proj 
        self.text_proj = model.text_proj 
        self.compress_layers = [i for i in range(1,len(self.vision_model.blocks))]
        self.model_len = len(self.vision_model.blocks)
        margin = 0.8
        alpha = 0.9
        self.margins = nn.ParameterList([
            nn.Parameter(torch.tensor(margin - i/self.model_len * alpha)) 
            for i in range(self.model_len)
        ])

   
    def get_vision_features(self, pixel_values, return_source:bool=False, return_attention_map=False):
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
        sources = [] 
        source = None
        for i, blk in enumerate(self.vision_model.blocks):
            if i in self.compress_layers: 
                cls = x[:, 0, :].unsqueeze(1)
                state, source = self.compress_hidden_state(
                    x[:, 1:, :], 
                    return_source=return_source,
                    margin=self.margins[i],
                    source=source,
                )
                x = torch.cat([cls, state], dim=1)

            if return_source:
                hidden_states.append(x)
                sources.append(source)
            real_mem += x.shape[1]
            total_mem += ori_size 

            flop += self.estimate_flop(x.shape)
            x = blk(x, register_hook=return_attention_map)

        x = self.vision_model.norm(x)
        vision_embed = self.vision_proj(x[:,0,:])
        return x, vision_embed, hidden_states, flop, real_mem/total_mem, sources

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
    
    def get_attention_scores(self):
        attn_score = []
        for i, blk in enumerate(self.vision_model.blocks): 
            attn_score.append(blk.attn.get_attention_map())
        return attn_score

        