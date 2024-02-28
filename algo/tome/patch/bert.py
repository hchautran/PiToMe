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
from transformers.models.bert.modeling_bert import BertLayer, BertEncoder, BertSelfAttention, BertAttention, apply_chunking_to_forward
from ..merge import merge_source, bipartite_soft_matching,  merge_wavg, merge_attention_mask
from typing import Optional, Union 
import math
from transformers.modeling_utils import ModuleUtilsMixin 


class ToMeBertLayer(BertLayer):
    def init_margin(self, margin):
        self.margin = margin
   
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        ratio = self._tome_info["ratio"].pop()
        x = self_attention_outputs[0]
        key = self_attention_outputs[1]

    

        if ratio < 1.0:
            merge, _ = bipartite_soft_matching(
                ratio=ratio,
                metric=x,
                class_token=self._tome_info["class_token"]
            )

            weight = self._tome_info["size"] 
            x, self._tome_info["size"] = merge_wavg(merge, x, None)
            # print(attention_mask.shape)

            attention_mask = torch.where(attention_mask.squeeze_() >= 0, 1, 0)
            attention_mask = merge_attention_mask(merge, attention_mask=attention_mask[..., None]).squeeze_()
        else:
            attention_mask = torch.where(attention_mask.squeeze_() >= 0, 1, 0)


        x = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, x
        )



        # print(x.isnan()._is_any_true())

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        outputs = (x, attention_mask, )  + outputs
        return outputs



class ToMeBertAttention(BertAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs, key = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + (key,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class ToMeBertSelfAttention(BertSelfAttention):

   def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

      
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        

    
        return outputs, key_layer.mean(1)


def make_tome_class(transformer_class):
    class ToMeBertEncoder(transformer_class, ModuleUtilsMixin):
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
            # self._tome_info["ratio"] = [self.ratio if i in [len_layers-1,len_layers-6] else 1.0 for i in range(len_layers) ]
            self._tome_info["ratio"] = [self.ratio for i in range(len(self.layer))]
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
                B, T, _ = hidden_states.shape

                hidden_states = layer_outputs[0]
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


    return ToMeBertEncoder



def apply_patch(
   model: BertEncoder, trace_source: bool = False, prop_attn: bool = True, margin=0.9, use_k=False):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeBertEncoder = make_tome_class(model.__class__)
    print('using', 'tome')

    model.__class__ = ToMeBertEncoder
    model.ratio = 1.0 
    model.r=0.0
    
    # model.compress_method = 'tome' 
    model._tome_info = {
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
            module.__class__ = ToMeBertLayer
            module.init_margin(margins[current_layer])
            module._tome_info = model._tome_info
            current_layer +=1
        if isinstance(module, BertAttention):
            module.__class__ = ToMeBertAttention 
        if isinstance(module, BertSelfAttention):
            module.__class__ = ToMeBertSelfAttention 

