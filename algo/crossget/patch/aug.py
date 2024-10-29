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
from timm.models.vision_transformer import Attention, Block, VisionTransformer
# from timm.models.helpers import checkpoint_seq 
from .timm import CrossGetAttention, CrossGetBlock



def make_cross_get_class(transformer_class):
    class CrossGetVisionTransformer(transformer_class):
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


    return CrossGetVisionTransformer



def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    CrossGetVisionTransformer = make_cross_get_class(model.__class__)
    print('using', 'cross_get')

    model.__class__ = CrossGetVisionTransformer
    model.ratio = 1.0 
    
    # model.compress_method = 'tome' 
    model._info = {
        "ratio": model.ratio,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }
    current_layer = 0
    num_layers = len(model.blocks)

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            # module.__class__ = ToMeBlock if compress_method == 'tome' else CrossGetBlock 
            module.__class__ = CrossGetBlock
            module._info = model._info
            current_layer +=1
        elif isinstance(module, Attention):
            module.__class__ = CrossGetAttention
