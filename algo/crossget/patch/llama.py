from transformers.models.llama.modeling_llama import LlamaModel 
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch
from typing import List
from ..merge import merge_source, merge_attention_mask, merge_mean, crossget 
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (
    logging,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
logger = logging.get_logger(__name__)
from LLaVA.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


class CrossGetLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """
    def init_margin(self, margins):
        # self.margin = nn.Parameter(torch.tensor(margin)) 
        self.margins = margins 



    def compress_x(self, metric, x, attn, idx):
        ratio = self._cross_get_info["ratio"].pop()
        if ratio < 1.0:
            merge, _ = crossget(
                ratio=ratio,
                metric=metric,
                class_token=self._cross_get_info["class_token"],
            )

            x, self._cross_get_info["size"] = merge_mean(merge, x)
            attention_mask = torch.where(attention_mask.squeeze_() >= 0, 1, 0)
            attention_mask = merge_attention_mask(merge, attention_mask=attention_mask[..., None]).squeeze_()
        else:
            attention_mask = torch.where(attention_mask.squeeze_() >= 0, 1, 0)
        return x


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) :
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        # print('input', inputs.shape)
        # print('images', images.shape)
        # print('image sizes', images_sizes.shape)

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


    def calculate_block_flop(self, shape):
            flops = 0
            _,N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops


def apply_patch(
   model: LlamaModel, trace_source: bool = False, prop_attn: bool = True, margin=0.9, use_k=False):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._cross_get_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    print('using', 'cross_get')

    model.__class__ =  CrossGetLLamaModel 
    model.ratio = 1.0 
    model.r=0.0
    
    # model.compress_method = 'cross_get' 
    model._cross_get_info = {
        "ratio": model.ratio,
        "margin":  [],
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }
    num_layers = len(model.layers)
    # margins = [margin - margin*(i/num_layers) for i in range(num_layers)]
    margins = [.9 - .9*(i/num_layers) for i in range(num_layers)]
    model.init_margin(margins)
