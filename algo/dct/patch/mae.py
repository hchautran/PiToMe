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

from .timm import  DCTBlockUsingRatio, DCTBlock


def make_dct_class(transformer_class):
    class DCTVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        - For MAE: make global average pooling proportional to token size
        """

        def forward(self, x, return_flop=True) -> torch.Tensor:
            margin = 0.95
            self._dct_info["r"] = [self.r]* len(self.blocks) 
            self._dct_info["ratio"] = [self.ratio] * len(self.blocks) 
            self._dct_info["size"] = None
            self._dct_info["source"] = None
            self.total_flop = 0
            x = super().forward(x)
            if return_flop:
                return x, self.calculate_flop()
            else:
                return x

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            # From the MAE implementation
            B = x.shape[0]
            x = self.patch_embed(x)

            T = x.shape[1]

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            for blk in self.blocks:
                self.total_flop += self.calculate_block_flop(x.shape) 
                x = blk(x)

            if self.global_pool:
                # ---- DCT changes this ----
                # Global average pool proportional to token size
                x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
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
            patch_embedding_flops = N*C*(self.patch_embed.patch_size[0]*self.patch_embed.patch_size[1]*3)
            classifier_flops = C*self.num_classes
            flops += patch_embedding_flops
            flops += self.total_flop 
            flops += classifier_flops
            return flops
        

    return DCTVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = False, use_k=True
):
    """
    Applies DCT to this MAE transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._dct_info["source"] afterward.

    For MAE models, prop_attn should be set to false.
    """
    DCTVisionTransformer = make_dct_class(model.__class__)
    print('using', 'dct')

    model.__class__ = DCTVisionTransformer
    model.r = 0
    model.ratio = 1.0
    model._dct_info = {
        "r": model.r,
        "ratio": model.ratio,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": False,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._dct_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = DCTBlockUsingRatio if not use_k else DCTBlock
            # module.__class__ = DCTBlock if compress_method == 'dct' else PiDCTBlock 
            module._dct_info = model._dct_info
