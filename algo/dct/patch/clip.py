from typing import Optional, Callable
import torch.nn as nn
import torch
from lavis.models.clip_models.model import Transformer, ResidualAttentionBlock
from ..merge import dc_transform 


class DCTBlock(ResidualAttentionBlock):

    def compress_x(self, x):
        ratio = self._info["ratio"].pop()
        if ratio < 1.0:
            x= dc_transform(
                x=x,
                ratio=ratio,
                class_token=self._info["class_token"]
            )
            # print(x.shape)
        return x 

    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        x.transpose_(1,0)
        x = self.compress_x(x).transpose_(1,0)
        return x


class DCTTransformer(Transformer):
    def __init__(
        self, width: int, layers: int, heads: int, act_layer: Callable = nn.GELU
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, act_layer=act_layer)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        self._info["ratio"] = [self.ratio] * len(self.resblocks) 
        self._info["size"] = None
        self._info["source"] = None
        self.total_flop = 0

        for r in self.resblocks:
            self.total_flop += self.calculate_block_flop(x.shape)
            x = r(x, attn_mask=attn_mask)
        return x

    def calculate_block_flop(self, shape):
            flops = 0
            N ,_, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops

        



def apply_patch(
   model: Transformer, trace_source: bool = False, prop_attn: bool = True):
    """
    Applies DCT to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    print('using', 'dct')

    model.__class__ = DCTTransformer 
    model.ratio = 1.0 
    
    # model.compress_method = 'dct' 
    model._info = {
        "ratio": model.ratio,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }
    current_layer = 0

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, ResidualAttentionBlock):
            # module.__class__ = DCTBlock if compress_method == 'dct' else DCTBlock 
            module.__class__ = DCTBlock
            module._info = model._info
            current_layer +=1
        # elif isinstance(module, Attention):
        #     module.__class__ = DCTAttention
