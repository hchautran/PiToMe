
import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis import Blip2Qformer
from .pitome import CompressedModel



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
        self.margins = nn.ParameterList([
            nn.Parameter(torch.tensor(0.75 - i/self.model_len * 0.5)) for i in range(self.model_len)
        ])
        self.compress_layers = [i for i in range(1, self.model_len)]

   
    def get_vision_features(self, pixel_values:torch.Tensor, return_source=False, return_attention_map=False):
        all_hidden_states = []
       
        total_mem=0
        real_mem=0
        flop = 0
        source = None
        sources = []
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
                    x, source = self.compress_hidden_state(
                        x, 
                        return_source=return_source,
                        margin=self.margins[i],
                        source=source
                    )
                x = blk(x, rel_pos_bias)
                if return_attention_map or i == len(self.visual_encoder.blocks) - 1:
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
        return vit_embeds, pooled_output, all_hidden_states, flop, real_mem/total_mem, source

    def get_text_features(self, input_ids, attention_mask):
        # with torch.no_grad():
        text_output = self.Qformer.bert(
            input_ids=input_ids.squeeze(),
            attention_mask=attention_mask.squeeze(),
            return_dict=True,
        )

        pooled_output = self.text_proj(text_output.last_hidden_state[:, 0, :])
        return text_output.last_hidden_state, pooled_output