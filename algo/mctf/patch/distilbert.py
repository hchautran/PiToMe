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
from transformers.models.distilbert.modeling_distilbert import Transformer, TransformerBlock, MultiHeadSelfAttention, apply_chunking_to_forward
from ..merge import bipartite_soft_matching,  merge_wavg, merge_attention_mask
from typing import Optional, Union 
import math
from transformers.modeling_utils import ModuleUtilsMixin 


class MCTFDistilBertBlock(TransformerBlock):
   
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=True,
        )
        ratio = self._info["ratio"].pop()
        sa_output, metric ,sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
    
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
   

        if ratio < 1.0:
            merge  = bipartite_soft_matching(
                ratio=ratio,
                metric=metric,
                class_token   = self._info["class_token"],
                tau_sim       = self._info["tau_sim"],
                tau_info      = self._info["tau_info"],
                tau_size      = self._info["tau_size"],
                size          = self._info["size"],
                bidirection   = self._info["bidirection"]
            )
            sa_output, self._info["size"] = merge_wavg(merge, sa_output, sa_weights, None)

            attn_mask = merge_attention_mask(merge, attention_mask=attn_mask[..., None]).squeeze_()
        else:
            attn_mask = attn_mask


        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = ffn_output
        if output_attentions:
            output = (attn_mask, sa_weights, output)
        return attn_mask, output 



class MCTFDistilBertAttention(MultiHeadSelfAttention):

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, k.mean(1), weights)
        else:
            return (context, k.mean(1))


def make_mctf_class(transformer_class):
    class MCTFTransformers(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
        def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: Optional[bool] = None,
        ): 

            len_layers = len(self.layer)
            self._info["ratio"] = [self.ratio if i in [
                len_layers - 1, 
                len_layers - 2,
                len_layers - 3,
                # len_layers - 6,
                # len_layers - 9,
            ] else 1.0 for i in range(len_layers) ]
            # self._info["ratio"] = [self.ratio for i in range(len(self.layer))]
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            hidden_state = x
            flops = 0
            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_state,)

                layer_outputs = layer_module(
                    x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions
                )
                hidden_state = layer_outputs[-1]
                # B, T, _ = hidden_state.shape

                attn_mask = layer_outputs[0]
                # print('mask',attn_mask.shape)
                # print('x',hidden_state.shape)
                    
                flops += self.calculate_block_flop(hidden_state.shape)


            # Add last layer
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            return hidden_state, all_hidden_states, all_attentions, flops
        
        def calculate_block_flop(self, shape):
            flops = 0
            _, N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops


    return MCTFTransformers



def apply_patch(
   model: Transformer, trace_source: bool = False, prop_attn: bool = True, output_attn: bool = False): 

    MCTFTransformers = make_mctf_class(model.__class__)
    print('using', 'mctf')

    model.__class__ = MCTFTransformers
    model.ratio = 1.0 
    
    # model.compress_method = 'mctf' 
    model._info = {
        "trace_source"   : False,
        "prop_attn"      : 1,
        "one_step_ahead" : 1,
        "tau_sim"        : 1,
        "tau_info"       : 20,
        "tau_size"       : 40,
        "bidirection"    : 1,
        "pooling_type"   : 0,
        "size": None,
        "class_token"  : True,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "output_attn": output_attn,
    }

    for module in model.modules():
        if isinstance(module, TransformerBlock):
            module.__class__ = MCTFDistilBertBlock 
            module._info = model._info
        if isinstance(module, MultiHeadSelfAttention):
            module.__class__ = MCTFDistilBertAttention 

