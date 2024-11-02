# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# mae: https://github.com/facebookresearch/mae
# --------------------------------------------------------


import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from copy import copy
from .timm import PiToMeBlock, PiToMeAttention, PiToMeBlock
import torch.nn as nn
import math


def make_pitome_class(transformer_class):
    class PiToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        - For MAE: make global average pooling proportional to token size
        """

        def forward(self, x, return_flop=True) -> torch.Tensor:
            self._info["ratio"] = [self.ratio] * len(self.blocks) 
            num_bsm_layers = math.ceil(len(self.blocks) * 0.5) 
            self._info["use_bsm_pitome"] = [False] * (num_bsm_layers) + [True] * (len(self.blocks)-num_bsm_layers)
            self._info["size"] = None
            self._info["source"] = None
            self._info["isolate_score"] = None
            self.total_flop = 0

            x = super().forward(x)
            if return_flop:
                return x, self.calculate_flop()
            else:
                return x

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            # From the MAE implementation
            B = x.shape[0]
            T = x.shape[1]

            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                self.total_flop += self.calculate_block_flop(x.shape) 
                x = blk(x)

            if self.global_pool:
                # ---- ToMe changes this ----
                # Global average pool proportional to token size
                if self._info["size"] is not None:
                    x = (x * self._info["size"])[:, 1:, :].sum(dim=1) / T
                else:
                    x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                # ---- End of change ----

                outcome = self.fc_norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]

            return outcome

        
        def calculate_block_flop(self, shape):
            flops = 0
            _, N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops


        def calculate_flop(self):
            C = self.embed_dim
            patch_number = float(self.patch_embed.num_patches)
            N = torch.tensor(patch_number+1).to('cuda')
            flops = 0
            flops += self.total_flop 
            return flops
        

    return PiToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = False, margin=0.9
):


    PiToMeVisionTransformer = make_pitome_class(model.__class__)
    print('using', 'pitome')

    current_layer = 0
    model.__class__ = PiToMeVisionTransformer
    model.ratio = 1.0
    model._info = {
        "ratio": model.ratio,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": False,
        "class_token": model.cls_token is not None,
        "distill_token": False,

    }
    current_layer = 0
    num_layers = len(model.blocks)
    # margins = [0.9- 0.9*(i/num_layers) for i in range(num_layers)]
    margins = [.9 - 0.9*(i/num_layers) for i in range(num_layers)]

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = PiToMeBlock
            module.init_margin(margins[current_layer])
            module._info = model._info
            current_layer +=1
        elif isinstance(module, Attention):
            module.__class__ = PiToMeAttention