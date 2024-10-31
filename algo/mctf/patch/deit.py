from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
# from timm.models.helpers import checkpoint_seq 
from .timm import MCTFBlock, MCTFBlock, MCTFAttention 

def make_tome_class(transformer_class):
    class MCTFVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, return_flop=True) -> torch.Tensor:

            self._info["ratio"] = [self.ratio] * len(self.blocks) 
            self._info["size"] = None
            self._info["source"] = None
            self.total_flop = 0

            x = super().forward(x)
            if return_flop:
                return x, self.total_flop
            else:
                return x


        def forward_features(self, x):
            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = self.pos_drop(x + self.pos_embed)
            for block in self.blocks:
                self.total_flop += self.calculate_block_flop(x.shape) 
                x = block(x)
            x = self.norm(x)
            if self.dist_token is None:
                return self.pre_logits(x[:, 0])
            else:
                return x[:, 0], x[:, 1]
 
        def calculate_block_flop(self, shape):
            flops = 0
            _, N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops


    return MCTFVisionTransformer



def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, output_attn=False
):

    MCTFVisionTransformer = make_tome_class(model.__class__)
    print('using', 'tome')

    model.__class__ = MCTFVisionTransformer
    model.ratio = 1.0 
    
    # model.compress_method = 'tome' 
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
        "class_token"  : model.cls_token is not None,
        "output_attn": output_attn,
        "prop_attn": prop_attn,
        "trace_source": trace_source,
    }


    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._info["distill_token"] = True

    for module in model.modules():

        if isinstance(module, Block):
            module.__class__ = MCTFBlock
            module._info = model._info
        elif isinstance(module, Attention):
            module.__class__ = MCTFAttention
