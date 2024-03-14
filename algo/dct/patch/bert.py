# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertLayer, BertEncoder, apply_chunking_to_forward
from ..merge import dc_transform 
from typing import Optional, Union 
import math
from transformers.modeling_utils import ModuleUtilsMixin 


class DCTBertLayer(BertLayer):
    def init_margin(self, margin):
        self.margin = margin
   
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # attn_size = self._dct_info["size"] if self._dct_info["prop_attn"] else None

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        ratio = self._dct_info["ratio"].pop()
        x = self_attention_outputs[0]

    


        x = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, x
        )


        if ratio < 1.0:
            x = dc_transform(
                x=x,
                ratio=ratio,
                class_token=self._dct_info["class_token"],
            )
            attention_mask = torch.ones((x.shape[0], x.shape[1])).to(x.device) 
        else:
            attention_mask.squeeze_()


        # print(x.isnan()._is_any_true())

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        outputs = (x, attention_mask, )  + outputs
        return outputs



def make_dct_class(transformer_class):
    class DCTBertEncoder(transformer_class, ModuleUtilsMixin):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
        ): 
            len_layers = len(self.layer)
            self._dct_info["ratio"] = [self.ratio if i in [
                len_layers - 1, 
                len_layers - 2,
                len_layers - 3,
                # len_layers - 6,
                # len_layers - 9,
            ] else 1.0 for i in range(len_layers) ]
            # self._dct_info["ratio"] = [self.ratio for i in range(len(self.layer))]
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            flops = 0

            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

   
                layer_head_mask = head_mask[i] if head_mask is not None else None

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
                hidden_states = layer_outputs[0]
                B, T, _ = hidden_states.shape
                attention_mask =   self.get_extended_attention_mask(
                    layer_outputs[1],
                    (B,T)
                )
                flops += self.calculate_block_flop(hidden_states.shape)

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[2],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            return (
                hidden_states,
                all_hidden_states,
                all_self_attentions,
                flops,
            )
    
        def calculate_block_flop(self, shape):
            flops = 0
            _, N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops


    return DCTBertEncoder



def apply_patch(
   model: BertEncoder, trace_source: bool = False, prop_attn: bool = True, margin=0.9, use_k=False):
    """
    Applies DCT to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._dct_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    DCTBertEncoder = make_dct_class(model.__class__)
    print('using', 'dct')

    model.__class__ = DCTBertEncoder
    model.ratio = 1.0 
    model.r=0.0
    
    # model.compress_method = 'dct' 
    model._dct_info = {
        "ratio": model.ratio,
        "margin":  [],
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }
    current_layer = 0
    margin = margin 
    num_layers = len(model.layer)
    margins = [0.75 - 0.25*(i/num_layers) for i in range(num_layers)]


    for module in model.modules():
        if isinstance(module, BertLayer):
            module.__class__ = DCTBertLayer
            module.init_margin(margins[current_layer])
            module._dct_info = model._dct_info
            current_layer +=1

