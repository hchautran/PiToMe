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
from pitome.merge import merge_source, pitome, merge_mean, merge_wavg



class PiToMeBlockUsingRatio(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def init_margin(self, margin=0.5):
        self.margin = nn.Parameter(torch.tensor(margin)) 

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        ratio = self._tome_info["ratio"].pop(0)
        if ratio < 1.0:
            merge, isolated_score = pitome(
                metric=metric,
                ratio=ratio,
                margin=self.margin,
                class_token=self._tome_info["class_token"]
            )

            if isolated_score is not None and self._tome_info["size"] is not None:
                x, self._tome_info["size"] = merge_wavg(merge, x, isolated_score + self._tome_info["size"])
            else:
                x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))

        return x 

class PiToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def init_margin(self, margin=0.5):
        self.margin = nn.Parameter(torch.tensor(margin))

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            merge, isolated_score = pitome(
                x=metric,
                r=r,
                margin=self.margin,
                class_token=self._tome_info["class_token"]
            )

            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            # x = merge_mean(merge, x)
            x, self._tome_info["size"] = merge_wavg(merge, x, isolated_score)

        x = x + self._drop_path2(self.mlp(self.norm2(x)))

        return x 




class PiToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, k.mean(1)


def make_pitome_class(transformer_class):
    class PiToMeVisionTransformer(transformer_class):
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
                

         #     flops = 0
        #     _, N, C = shape
        #     mhsa_flops = 4*N*C*C + 2*N*N*C
        #     flops += mhsa_flops
        #     ffn_flops = 8*N*C*C
        #     flops += ffn_flops
        #     return flops

      # def calculate_block_flop(self, shape):
 
        def calculate_flop(self):
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
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, margin=0.9, use_r=False):
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
    num_layers = len(model.blocks)
    margins = [margin - margin*(i/num_layers) for i in range(num_layers)]

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            # module.__class__ = ToMeBlock if compress_method == 'tome' else PiToMeBlock 
            module.__class__ = PiToMeBlock if use_r else PiToMeBlockUsingRatio 
            module.init_margin(margins[current_layer])
            module._tome_info = model._tome_info
            current_layer +=1
        elif isinstance(module, Attention):
            module.__class__ = PiToMeAttention
