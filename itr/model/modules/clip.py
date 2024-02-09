
import torch
import torch.nn as nn
from transformers import AutoModel
from .pitome import CompressedModel

class CompressedHFCLIP(CompressedModel):

    def __init__(self, model:AutoModel, compress_method='dct',r=0.9, use_k=False, k=13):
        super(CompressedHFCLIP, self).__init__(compress_method, r=r, use_k=use_k, k=k)

        self.vision_model = model.vision_model
        self.text_model = model.text_model 
        self.vision_proj = model.visual_projection 
        self.text_proj = model.text_projection 
        self.compress_layers = [i for i in range(1,len(self.vision_model.encoder.layers))]
        self.model_len = len(self.vision_model.encoder.layers)
        self.margins = nn.ParameterList([
            nn.Parameter(torch.tensor(0.9 - i/self.model_len * 0.9)) for i in range(self.model_len)
        ])

    def get_vision_features(self, pixel_values, return_source:bool=False, return_attention_map=False):
        all_hidden_states = []
        hidden_states = self.vision_model.embeddings(pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)
        real_mem = 0
        total_mem = 0
        flop = 0
        sources = [] 
        source = None
        ori_size = hidden_states.shape[1]
        for i, layer in enumerate(self.vision_model.encoder.layers):
            if i in self.compress_layers:
                cls = hidden_states[:, 0, :].unsqueeze(1)
                state, source = self.compress_hidden_state(
                    hidden_states[:, 1:, :], 
                    source=source,
                    margin=self.margins[i],
                    return_source=return_source
                )
                hidden_states = torch.cat([cls, state], dim=1)
            if return_attention_map or i == len(self.vision_model.encoder.layers)-1:
                all_hidden_states.append(hidden_states)
                sources.append(source)

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

        return hidden_states, vision_embed, all_hidden_states, flop, real_mem/total_mem, sources

    def get_text_features(self, input_ids, attention_mask):
        text_output = self.text_model(input_ids, attention_mask)
        last_hidden_state = text_output[1] 
        text_embed = self.text_proj(last_hidden_state)

        return  last_hidden_state, text_embed