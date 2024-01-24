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
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from copy import copy

from pitome.merge import merge_source, pitome, merge_mean
from .deit import PiToMeAttention, PiToMeBlock


def make_pitome_class(transformer_class):
    class PiToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, return_flop=True) -> torch.Tensor:
            margin = 0.75
            self._tome_info["ratio"] = [self.ratio] * len(self.blocks) 
            margins = [margin if i < len(self.blocks)//2 else margin - margin*(i/len(self.blocks)) for i in range(len(self.blocks))]
            self._tome_info["margin"] = margins 
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            x = super().forward(x)
            if return_flop:
                if self.training:
                    flops = self.calculate_flop_training()
                else:
                    flops = self.calculate_flop_inference()
                return x, flops
            else:
                return x
                

        def calculate_flop_training(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1).to('cuda')
            flops = 0
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            with torch.cuda.amp.autocast(enabled=False):
                for block in (self.blocks):
                    # translate fp16 to fp32 for stable training
                    mhsa_flops = 4*N*C*C + 2*N*N*C
                    flops += mhsa_flops
                    N = N * self.ratio
                    ffn_flops = 8*N*C*C
                    flops += ffn_flops
            flops += patch_embedding_flops
            flops += classifier_flops
            return flops

        def calculate_flop_inference(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1).to('cuda')
            flops = 0
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            with torch.cuda.amp.autocast(enabled=False):
                for block in (self.blocks):
                    mhsa_flops = 4*N*C*C + 2*N*N*C
                    flops += mhsa_flops
                    N = N * self.ratio
                    ffn_flops = 8*N*C*C
                    flops += ffn_flops
            flops += patch_embedding_flops
            flops += classifier_flops
            return flops

    return PiToMeVisionTransformer



def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, margin=0.5):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    PiToMeVisionTransformer = make_pitome_class(model.__class__)
    print('using', 'pitome')

    model.__class__ = PiToMeVisionTransformer
    model.ratio = 1.0 
    
    # model.compress_method = 'tome' 
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
    num_layers = len(module.blocks)
    margins = [margin - margin*(i/num_layers) for i in range(num_layers)]

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            # module.__class__ = ToMeBlock if compress_method == 'tome' else PiToMeBlock 
            module.__class__ = PiToMeBlock 
            module.init_margin(margins[current_layer])
            module._tome_info = model._tome_info
            current_layer +=1
        elif isinstance(module, Attention):
            module.__class__ = PiToMeAttention
