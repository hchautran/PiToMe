from transformers.models.clip.modeling_clip import CLIPEncoder
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Union
import torch
from ..ddp import DiffRate
from ..merge import get_merge_func


class DiffRateCLIPEncoder(CLIPEncoder):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """
    def introduce_diffrate(self):
        self.prune_ddp = [DiffRate(576,1, True) for layer in self.layers]
        self.merge_ddp = [DiffRate(576,1, True) for layer in self.layers]



    def compress_x(self, metric, x, attn, i):
        B, _, _ = x.shape
        size = self._diffrate_info["size"]
        mask = self._diffrate_info["mask"]
        cls_attn = attn[:, :, 0, 1:].mean(1)
        _, idx = torch.sort(cls_attn, descending=True)
        cls_index = torch.zeros((B,1), device=idx.device).long()
        idx = torch.cat((cls_index, idx+1), dim=1)
        x = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        if self._diffrate_info["trace_source"]:
            self._diffrate_info["source"] = torch.gather(self._diffrate_info["source"], dim=1, index=idx.unsqueeze(-1).expand(-1, -1, self._diffrate_info["source"].shape[-1]))

             # pruning
        prune_kept_num = self.prune_ddp[i].kept_token_number
        x = x[:, :prune_kept_num]
        if self._diffrate_info["trace_source"]:
            self._diffrate_info["source"] = self._diffrate_info["source"][:, :prune_kept_num]
        # merging
        merge_kept_num = self.merge_ddp[i].kept_token_number
        if merge_kept_num < prune_kept_num:
            merge, _ = get_merge_func(x.detach(), kept_number=merge_kept_num)
            x = merge(x,mode='mean')
            if self._diffrate_info["trace_source"]:
                self._diffrate_info["source"] = merge(self._diffrate_info["source"], mode="amax")
        return x

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        B = inputs_embeds.shape[0]
        N = inputs_embeds.shape[1]
        self._diffrate_info["size"] = torch.ones([B,N +1,1], device=inputs_embeds.device)
        self._diffrate_info["mask"] =  torch.ones((B,N+1),device=inputs_embeds.device)
        self._diffrate_info["prune_kept_num"] = []
        self._diffrate_info["merge_kept_num"] = []
        if self._diffrate_info["trace_source"]:
            self._diffrate_info["source"] = torch.eye(self.patch_embed.num_patches+1, device=inputs_embeds.device)[None, ...].expand(B, self.patch_embed.num_patches+1, self.patch_embed.num_patches+1)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        self.total_flops = 0

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=True,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=True,
                )

            hidden_states = layer_outputs[0]
            self.total_flops += self.calculate_block_flop(hidden_states.shape)
            hidden_states= self.compress_x(hidden_states, hidden_states, layer_outputs[1], idx)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


    def calculate_block_flop(self, shape):
            flops = 0
            _,N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops
    
    
    def parameters(self):
        # original network parameter
        params = []
        for n, m in self.named_parameters():
            if n.find('ddp') > -1:
                continue
            params.append(m)
        return iter(params)     

    def arch_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('ddp') > -1:
                params.append(m)
        return iter(params)    

    def get_kept_num(self):
        prune_kept_num = []
        merge_kept_num = []
        for idx,_ in enumerate(self.layers):
            prune_kept_num.append(int(self.prune_ddp[idx].kept_token_number))
            merge_kept_num.append(int(self.merge_ddp[idx].kept_token_number))
        return prune_kept_num, merge_kept_num
            

    def set_kept_num(self, prune_kept_numbers, merge_kept_numbers):
        assert len(prune_kept_numbers) == len(self.blocks) and len(merge_kept_numbers) == len(self.blocks)
        for idx, block, prune_kept_number, merge_kept_number in enumerate(zip(self.blocks, prune_kept_numbers, merge_kept_numbers)):
            self.prune_ddp[idx].kept_token_number = prune_kept_number
            self.merge_ddp[idx].kept_token_number = merge_kept_number
    
    def init_kept_num_using_ratio(self, ratio):
        import math
        N = 577 
        for idx, _ in enumerate(self.layers):
            if idx % 2 == 0:
                r = math.floor(N - N*ratio)
                self.prune_ddp[idx].kept_token_number = N - 0 
                self.merge_ddp[idx].kept_token_number = N - r
                N -= r
        
    def init_kept_num_using_r(self, r):
        N = 577 
        for idx, _ in enumerate(self.layers):
            r = min(r, N // 2)
            self.prune_ddp[idx].kept_token_number = N - 0 
            self.merge_ddp[idx].kept_token_number = N - r
            N -= r


def apply_patch(
   model: CLIPEncoder, trace_source: bool = False, prop_attn: bool = True, margin=0.9, use_k=False):
    """
    Applies DiffRate to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._diffrate_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    print('using', 'diffrate')

    model.__class__ =  DiffRateCLIPEncoder 
    model.introduce_diffrate()
    model.ratio = 1.0 
    model.r=0.0
    
    # model.compress_method = 'diffrate' 
    model._diffrate_info = {
        "ratio": model.ratio,
        "margin":  [],
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }
    margin = margin 