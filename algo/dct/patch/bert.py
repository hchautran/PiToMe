import torch
from typing import Tuple
from transformers.models.bert.modeling_bert import BertLayer, BertEncoder, apply_chunking_to_forward
from transformers.modeling_utils import ModuleUtilsMixin 
from ..merge import dc_transform 
from typing import Optional


class DCTBertLayer(BertLayer):
   
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # attn_size = self._info["size"] if self._info["prop_attn"] else None

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        ratio = self._info["ratio"].pop()
        x = self_attention_outputs[0]

        x = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, x
        )


        if ratio < 1.0:
            x = dc_transform(
                x=x,
                ratio=ratio,
                class_token=self._info["class_token"],
            )


        # print(x.isnan()._is_any_true())

        outputs = (x,)+self_attention_outputs[1:] 
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
            self._info["ratio"] = [self.ratio if i in [
                len_layers - 1, 
                len_layers - 2,
                len_layers - 3,
                # len_layers - 6,
                # len_layers - 9,
            ] else 1.0 for i in range(len_layers) ]
            # self._info["ratio"] = [self.ratio for i in range(len(self.layer))]
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
                attention_mask = None 
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
   model: BertEncoder, trace_source: bool = False, prop_attn: bool = True):

    DCTBertEncoder = make_dct_class(model.__class__)
    print('using', 'dct')

    model.__class__ = DCTBertEncoder
    model.ratio = 1.0 
    
    # model.compress_method = 'dct' 
    model._info = {
        "ratio": model.ratio,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }
    current_layer = 0


    for module in model.modules():
        if isinstance(module, BertLayer):
            module.__class__ = DCTBertLayer
            module._info = model._info
            current_layer +=1

