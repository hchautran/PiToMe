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


from .timm import PiToMeBlock, ToMeBlock, ToMeAttention, ToMeBlockUsingRatio


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        - For MAE: make global average pooling proportional to token size
        """

        def forward(self, x, return_flop=True) -> torch.Tensor:
            # self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            margin = 0.90
            self._tome_info["r"] = [self.r]* len(self.blocks) 
            self._tome_info["ratio"] = [self.ratio] * len(self.blocks) 
            # margins = [margin if i < len(self.blocks)//2 else margin - margin*(i/len(self.blocks)) for i in range(len(self.blocks))]
            margins = [margin - margin*(i/len(self.blocks)) for i in range(len(self.blocks))]
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
                x = blk(x)

            if self.global_pool:
                # ---- ToMe changes this ----
                # Global average pool proportional to token size
                if self._tome_info["size"] is not None:
                    x = (x * self._tome_info["size"])[:, 1:, :].sum(dim=1) / T
                else:
                    x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                # ---- End of change ----

                outcome = self.fc_norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]

            return outcome
        
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
        

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, compress_method='tome', trace_source: bool = False, prop_attn: bool = False
):
    """
    Applies ToMe to this MAE transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For MAE models, prop_attn should be set to false.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)
    print('using', compress_method)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model.ratio = 1.0
    model._tome_info = {
        "r": model.r,
        "ratio": model.ratio,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlockUsingRatio if compress_method == 'tome' else PiToMeBlock 
            # module.__class__ = ToMeBlock if compress_method == 'tome' else PiToMeBlock 
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention