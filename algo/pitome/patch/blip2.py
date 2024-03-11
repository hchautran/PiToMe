import torch
import torch.nn.functional as F 
import torch.utils.checkpoint as checkpoint
from lavis.models.eva_vit import VisionTransformer,Attention,Block
from ..merge import merge_source, pitome_vision, prune, merge_mean, merge_wavg

class PiToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def init_margin(self, margin=0.5):
        # self.margin = nn.Parameter(torch.tensor(margin)) 
        self.margin = margin
    
    def compress_x(self, metric, x):
        ratio = self._tome_info["ratio"]
        if ratio < 1.0:
            merge, isolated_score = pitome_vision(
                ratio=ratio,
                metric=metric,
                margin=self.margin,
                class_token=self._tome_info["class_token"]
            )

            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )

            # if isolated_score is not None and self._tome_info["size"] is not None:
            #     weight = self._tome_info["size"] + isolated_score
            #     x, self._tome_info["size"] = merge_wavg(merge, x, weight)
            # else:
            weight = self._tome_info["size"] 
            # print(x.shape)
            # print(weight.shape)
            x, self._tome_info["size"] = merge_wavg(merge, x, weight )
            # print(x.shape)
        return x


    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x_attn = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
            x = x + self.drop_path(x_attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_attn = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
            x = x + self.drop_path(x_attn)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = self.compress_x(x,x)
        return x


def make_pitome_class(transformer_class):
    class PiToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x) -> torch.Tensor:
      
            self._tome_info["r"] = [self.r]* len(self.blocks) 
            self._tome_info["ratio"] = self.ratio
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self.total_flop = 0

            x = super().forward(x)
            return x
                
        def forward_features(self, x):
            x = self.patch_embed(x)
            batch_size, seq_len, _ = x.size()

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            x = self.pos_drop(x)

            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, rel_pos_bias)
                else:
                    x = blk(x, rel_pos_bias)
                self.total_flop+= self.calculate_block_flop(x.shape)
            return x
 
        def calculate_block_flop(self, shape):
            flops = 0
            _, N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops

    return PiToMeVisionTransformer


def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, margin=0.9, use_k=False):
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
    model.r=0.0
    
    # model.compress_method = 'tome' 
    model._tome_info = {
        "ratio": model.ratio,
        "margin":  [],
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": False,
        "class_token": False,
        "distill_token": False,
    }
    current_layer = 0
    margin = margin 
    num_layers = len(model.blocks)
    margins = [.75 - 0.5*(i/num_layers) for i in range(num_layers)]

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            # module.__class__ = ToMeBlock if compress_method == 'tome' else PiToMeBlock 
            module.__class__ = PiToMeBlock
            module.init_margin(margins[current_layer])
            module._tome_info = model._tome_info
            current_layer +=1
        # elif isinstance(module, Attention):
        #     module.__class__ = PiToMeAttention
