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
from lib.timm import CustomViTBlock, CustomAttention


def make_custom_class(transformer_class):
    class CustomDeiT(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, return_flop=True) -> torch.Tensor:
      
            self._tome_info["r"] = [self.r]* len(self.blocks) 
            self._tome_info["ratio"] = [self.ratio] * len(self.blocks) 
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self.total_flop = 0

            x = super().forward(x)
            if return_flop:
                return x, self.calculate_flop()
            else:
                return x
                
        def calculate_flop(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1).to('cuda')
            flops = 0
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            with torch.cuda.amp.autocast(enabled=False):
                for block in self.blocks:
                    mhsa_flops = 4*N*C*C + 2*N*N*C
                    flops += mhsa_flops
                    N = N * self.r if self.use_k else  N - self.k 
                    ffn_flops = 8*N*C*C
                    flops += ffn_flops
            flops += patch_embedding_flops
            flops += classifier_flops
            return flops
        

    return CustomDeiT


def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, margin=0.9, use_k=False, algo='none', r=1.0, k=0
):
    CustomVisionTransformer = make_custom_class(model.__class__)
    print('using', algo)

    model.__class__ = CustomVisionTransformer
    
    model._tome_info = {
        "ratio": model.ratio,
        "margin":  [],
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }
    current_layer = 0
    margin = margin 
    num_layers = len(model.blocks)
    margins = [margin - margin*(i/num_layers) for i in range(num_layers)]

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = CustomViTBlock 
            module._tome_info = model._tome_info
            module.init_algo(algo=algo, r=r, k=k, use_k=use_k, margin=margins[current_layer])
            current_layer +=1
        elif isinstance(module, Attention):
            module.__class__ = CustomAttention 
