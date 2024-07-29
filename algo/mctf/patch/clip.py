from lavis.models.clip_models.model import Transformer, ResidualAttentionBlock
from typing import Optional, Callable
import torch.nn as nn
import torch
from ..merge import merge_source, bipartite_soft_matching, merge_wavg


class MCTFBlock(ResidualAttentionBlock):
    def init_margin(self, margin=0.5):
        # self.margin = nn.Parameter(torch.tensor(margin)) 
        self.margin = margin

    def compress_x(self, metric, x, attn):
        ratio = self._mctf_info["ratio"].pop()
        if ratio < 1.0:
            merge, _ = bipartite_soft_matching(
                ratio=ratio,
                metric=metric,
                class_token   = self._mctf_info["class_token"],
                tau_sim       = self._mctf_info["tau_sim"],
                tau_info      = self._mctf_info["tau_info"],
                tau_size      = self._mctf_info["tau_size"],
                size          = self._mctf_info["size"],
                bidirection   = self._mctf_info["bidirection"]
            )

            if self._mctf_info["trace_source"]:
                self._mctf_info["source"] = merge_source(
                    merge, x, self._mctf_info["source"]
                )
            x, self._mctf_info["size"], _ = merge_wavg(
                merge=merge, 
                x=x, 
                attn=attn,
                size=self._mctf_info["size"]
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
        self._mctf_info["r"] = [self.r]* len(self.resblocks) 
        self._mctf_info["ratio"] = [self.ratio] * len(self.resblocks) 
        self._mctf_info["size"] = None
        self._mctf_info["source"] = None
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
    Applies MCTF to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._mctf_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    print('using', 'mctf')

    model.__class__ = MCTFTransformer 
    model.ratio = 1.0 
    model.r=0.0
    
    # model.compress_method = 'mctf' 
    model._mctf_info = {
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
    current_layer = 0
    margin = margin 
    num_layers = len(model.resblocks)
    # margins = [margin - margin*(i/num_layers) for i in range(num_layers)]
    margins = [.9 - .9*(i/num_layers) for i in range(num_layers)]

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._mctf_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, ResidualAttentionBlock):
            # module.__class__ = MCTFBlock if compress_method == 'mctf' else MCTFBlock 
            module.__class__ = MCTFBlock
            module.init_margin(margins[current_layer])
            module._mctf_info = model._mctf_info
            current_layer +=1
        # elif isinstance(module, Attention):
        #     module.__class__ = MCTFAttention
