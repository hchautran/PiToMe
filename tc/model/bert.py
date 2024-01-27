    

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import (
    BlipConfig, 
)
import math
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Union, Tuple
from .pitome import CompressedModel




class CompressedBERT(CompressedModel):
    config_class = BlipConfig

    def __init__(self, model:BertForSequenceClassification, compress_method='dct', r=0.9):
        super(CompressedBERT, self).__init__(compress_method, r=r)
        self.model = model.bert
        self.dropout = model.dropout 
        self.classifier = model.classifier
        self.config = model.config

        self.compress_layers = [i for i in range(1, len(self.model.encoder.layer)-1, 3)]
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
                margin = 0.9
                state, cur_energy = self.compress_hidden_state(
                    hidden_states[:, 1:, :], 
                    use_compressed_hidden_state=True,
                    margin=(margin-margin*i/self.model_len)
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


