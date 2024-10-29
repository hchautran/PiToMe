from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from .timm import ToFuBlock, ToFuBlock, ToFuAttention 
# from timm.models.helpers import checkpoint_seq 

def make_tofu_class(transformer_class):
    class ToFuVisionTransformer(transformer_class):
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
            x = self.pos_embed(x)
            x = self.norm_pre(x)
            # if self.grad_checkpointing and not torch.jit.is_scripting():
                # self.total_flop += self.calculate_block_flop(x.shape) 
                # x = checkpoint_seq(self.blocks, x)
            # else:
            for block in self.blocks:
                self.total_flop += self.calculate_block_flop(x.shape) 
                x = block(x)
            x = self.norm(x)
            return x
 
 
        def calculate_block_flop(self, shape):
            flops = 0
            _, N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops


    return ToFuVisionTransformer



def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):

    ToFuVisionTransformer = make_tofu_class(model.__class__)
    print('using', 'tofu')

    model.__class__ = ToFuVisionTransformer
    model.ratio = 1.0 
    
    
    # model.compress_method = 'tofu' 
    model._info = {
        "ratio": model.ratio,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._info["distill_token"] = True

    num_layers = len(model.blocks)
    strategies = ['tofu' if i > num_layers//2 else 'prune' for i in range(num_layers)]
    current_layer = 0 
    for module in model.modules():

        if isinstance(module, Block):
            module.__class__ = ToFuBlock
            module.init_strategy(strategies[current_layer])
            module._info = model._info
            current_layer += 1
        elif isinstance(module, Attention):
            module.__class__ = ToFuAttention
