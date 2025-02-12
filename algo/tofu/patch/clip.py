from lavis.models.clip_models.model import Transformer, ResidualAttentionBlock
from typing import Optional, Callable
import torch.nn as nn
import torch
from ..merge import merge_source, bipartite_soft_matching, merge_wavg


class ToFuBlock(ResidualAttentionBlock):

    def init_strategy(self, strategy='mean'):
        self.strategy = strategy 

    def compress_x(self, metric, x):
        ratio = self._info["ratio"].pop()
        if ratio < 1.0:
            merge = bipartite_soft_matching(
                ratio=ratio,
                metric=metric,
                class_token=self._info["class_token"]
            )

            if self._info["trace_source"]:
                self._info["source"] = merge_source(
                    merge, x, self._info["source"]
                )

            weight = self._info["size"] 
            x, self._info["size"] = merge_wavg(merge, x, weight)

        return x

    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        x.transpose_(1,0)
        x = self.compress_x(x, x).transpose_(1,0)
        return x


class ToFuTransformer(Transformer):
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
   model: Transformer, trace_source: bool = False, prop_attn: bool = True ):

    print('using', 'tofu')

    model.__class__ = ToFuTransformer 
    model.ratio = 1.0 
    
    # model.compress_method = 'tofu' 
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
    num_layers = len(model.resblocks)
    strategies = ['tofu' if i > num_layers//2 else 'prune' for i in range(num_layers)]

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, ResidualAttentionBlock):
            # module.__class__ = ToFuBlock if compress_method == 'tofu' else ToFuBlock 
            module.__class__ = ToFuBlock
            module._info = model._info
            module.init_strategy(strategies[current_layer])
            current_layer +=1
        # elif isinstance(module, Attention):
        #     module.__class__ = ToFuAttention
