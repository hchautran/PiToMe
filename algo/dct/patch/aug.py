from typing import Tuple

import torch
from timm.models.vision_transformer import Block, VisionTransformer
from copy import copy

from .timm import DCTBlockUsingRatio

def make_dct_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x, return_flop=True) -> torch.Tensor:

            self._dct_info["r"] = [self.r] * len(self.blocks) 
            self._dct_info["ratio"] = [self.ratio] * len(self.blocks) 
            self._dct_info["size"] = None
            self._dct_info["source"] = None
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
            for blk in self.blocks:
                self.total_flop += self.calculate_block_flop(x.shape) 
                x = blk(x)
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


    return ToMeVisionTransformer



def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, use_k=False
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._dct_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_dct_class(model.__class__)
    print('using', 'dct')

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model.ratio = 1.0 
    model.use_k = use_k
    
    # model.compress_method = 'dct' 
    model._dct_info = {
        "ratio": model.ratio,
        "margin":  [],
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._dct_info["distill_token"] = True

    for module in model.modules():

        if isinstance(module, Block):
            module.__class__ = DCTBlockUsingRatio
            module._dct_info = model._dct_info
