from lavis.models.clip_models.model import Transformer, ResidualAttentionBlock
from typing import Optional, Callable
import torch.nn as nn
import torch
from ..merge import merge_source, bipartite_soft_matching, merge_wavg


class ToMeBlock(ResidualAttentionBlock):
    def init_margin(self, margin=0.5):
        # self.margin = nn.Parameter(torch.tensor(margin)) 
        self.margin = margin

    def compress_x(self, metric, x):
        ratio = self._tome_info["ratio"].pop()
        if ratio < 1.0:
            merge, isolated_score = bipartite_soft_matching(
                ratio=ratio,
                metric=metric,
                class_token=self._tome_info["class_token"]
            )

            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            if isolated_score is not None and self._tome_info["size"] is not None:
                weight = self._tome_info["size"] + isolated_score
                x, self._tome_info["size"] = merge_wavg(merge, x, weight)
            else:
                weight = self._tome_info["size"] 
                x, self._tome_info["size"] = merge_wavg(merge, x, weight)
        return x

    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        x.transpose_(1,0)
        x = self.compress_x(x, x).transpose_(1,0)
        return x


class ToMeTransformer(Transformer):
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
        self._tome_info["r"] = [self.r]* len(self.resblocks) 
        self._tome_info["ratio"] = [self.ratio] * len(self.resblocks) 
        self._tome_info["size"] = None
        self._tome_info["source"] = None
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
   model: Transformer, trace_source: bool = False, prop_attn: bool = True, margin=0.9, use_k=False):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    print('using', 'tome')

    model.__class__ = ToMeTransformer 
    model.ratio = 1.0 
    model.r=0.0
    
    # model.compress_method = 'tome' 
    model._tome_info = {
        "ratio": model.ratio,
        "margin":  [],
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }
    current_layer = 0
    margin = margin 
    num_layers = len(model.resblocks)
    # margins = [margin - margin*(i/num_layers) for i in range(num_layers)]
    margins = [.9 - .9*(i/num_layers) for i in range(num_layers)]

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, ResidualAttentionBlock):
            # module.__class__ = ToMeBlock if compress_method == 'tome' else ToMeBlock 
            module.__class__ = ToMeBlock
            module.init_margin(margins[current_layer])
            module._tome_info = model._tome_info
            current_layer +=1
        # elif isinstance(module, Attention):
        #     module.__class__ = ToMeAttention
