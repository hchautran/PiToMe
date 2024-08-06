from lavis.models.clip_models.model import Transformer, ResidualAttentionBlock
from typing import Optional, Callable
import torch.nn as nn
import torch
from ..merge import merge_source, crossget, merge_wavg


class CrossGetBlock(ResidualAttentionBlock):
    def init_margin(self, margin=0.5):
        # self.margin = nn.Parameter(torch.tensor(margin)) 
        self.margin = margin

    def compress_x(self, metric, x, attn):
        ratio = self._cross_get_info["ratio"].pop()
        if ratio < 1.0:
            merge, _ = crossget(
                ratio=ratio,
                metric=metric,
                class_token=self._cross_get_info["class_token"]
            )

            if self._cross_get_info["trace_source"]:
                self._cross_get_info["source"] = merge_source(
                    merge, x, self._cross_get_info["source"]
                )
            x, self._cross_get_info["size"] = merge_wavg(merge, x, size=self._cross_get_info["size"])
        return x

    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        attn_x, attn = self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + attn_x 
        x.transpose_(1,0)
        x = self.compress_x(x, x, attn).transpose_(1,0)
        x = x + self.mlp(self.ln_2(x))
        return x


    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x_attn, attn = self.attn(x, x, x, need_weights=True, attn_mask=attn_mask)
        return x_attn, attn


class CrossGetTransformer(Transformer):


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        self._cross_get_info["r"] = [self.r]* len(self.resblocks) 
        self._cross_get_info["ratio"] = [self.ratio] * len(self.resblocks) 
        self._cross_get_info["size"] = None
        self._cross_get_info["source"] = None
        self.total_flop = 0

        for r in self.resblocks:
            self.total_flop += self.calculate_block_flop(x.shape)
            x = r(x, attn_mask=attn_mask)
        return x

    def calculate_block_flop(self, shape):
            flops = 0
            N,_, C = shape
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
    The sources will be available at model._cross_get_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    print('using', 'cross_get')

    model.__class__ = CrossGetTransformer 
    model.ratio = 1.0 
    model.r=0.0
    
    # model.compress_method = 'cross_get' 
    model._cross_get_info = {
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

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._cross_get_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, ResidualAttentionBlock):
            module.__class__ = CrossGetBlock
            module._cross_get_info = model._cross_get_info
            current_layer +=1
