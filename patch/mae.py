

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from lib.timm import CustomViTBlock, CustomAttention 

def make_pitome_class(transformer_class):
    class PiToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        - For MAE: make global average pooling proportional to token size
        """

        def forward(self, x, return_flop=True) -> torch.Tensor:
            self._tome_info["r"] = [self.r]* len(self.blocks) 
            self._tome_info["ratio"] = [self.ratio] * len(self.blocks) 
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self._tome_info["isolate_score"] = None
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

            cls_tokens = self.cls_token.expand(B, -1, -1)  
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)
                self.total_flop += self.calculate_block_flop(x.shape) 

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

        
        def calculate_block_flop(self, shape):
            flops = 0
            _, N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops


        def calculate_flop(self):
            return self.total_flop 
        

    return PiToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, margin=0.9, use_k=False, algo='none', r=1.0, k=0
):
    """
    Applies ToMe to this MAE transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For MAE models, prop_attn should be set to false.
    """

    PiToMeVisionTransformer = make_pitome_class(model.__class__)
    print('using', 'pitome')

    current_layer = 0
    model.__class__ = PiToMeVisionTransformer

    model._tome_info = {
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
    margins = [0.9- 0.9*(i/num_layers) for i in range(num_layers)]
    print(margins)


    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = CustomViTBlock 
            module._tome_info = model._tome_info
            module.init_algo(algo=algo, r=r, k=k, use_k=use_k, margin=margins[current_layer])
            current_layer +=1
        elif isinstance(module, Attention):
            module.__class__ = CustomAttention 
