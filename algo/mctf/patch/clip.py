from lavis.models.clip_models.model import Transformer, ResidualAttentionBlock
from typing import Optional, Callable
import torch.nn as nn
import torch
from ..merge import merge_source, bipartite_soft_matching, merge_wavg


class MCTFBlock(ResidualAttentionBlock):

    def compress_x(self, metric, x, attn):
        ratio = self._info["ratio"].pop()
        if ratio < 1.0:
            merge = bipartite_soft_matching(
                ratio=ratio,
                metric=metric,
                class_token   = self._info["class_token"],
                tau_sim       = self._info["tau_sim"],
                tau_info      = self._info["tau_info"],
                tau_size      = self._info["tau_size"],
                size          = self._info["size"],
                bidirection   = self._info["bidirection"]
            )

            if self._info["trace_source"]:
                self._info["source"] = merge_source(
                    merge, x, self._info["source"]
                )
            x, self._info["size"], _ = merge_wavg(
                merge=merge, 
                x=x, 
                attn=attn,
                size=self._info["size"]
            )
        return x

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=True, attn_mask=attn_mask)   
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        _, attn = self.attention(self.ln_1(x), attn_mask=attn_mask)
        # print(attn.shape)
        x.transpose_(1,0)
        x = self.compress_x(x, x, attn).transpose_(1,0)
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class MCTFTransformer(Transformer):
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

    print('using', 'mctf')

    model.__class__ = MCTFTransformer 
    model.ratio = 1.0 
    
    # model.compress_method = 'mctf' 
    model._info = {
        "trace_source"   : False,
        "prop_attn"      : 1,
        "one_step_ahead" : 0,
        "tau_sim"        : 1,
        "tau_info"       : 20,
        "tau_size"       : 40,
        "bidirection"    : 1,
        "pooling_type"   : 0,
        "size": None,
        "class_token"  : True,
        "output_attn": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, ResidualAttentionBlock):
            # module.__class__ = MCTFBlock if compress_method == 'mctf' else MCTFBlock 
            module.__class__ = MCTFBlock
            module._info = model._info
        # elif isinstance(module, Attention):
        #     module.__class__ = MCTFAttention
