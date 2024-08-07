import torch
from lavis.models.vit import VisionTransformer, Attention, Block
from ..merge import merge_source, pitome_vision, merge_wavg, pitome_vision_using_attn, unprotected_pitome_vision

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
        ratio = self._pitome_info["ratio"].pop()
        if ratio < 1.0:
            merge, isolated_score = pitome_vision(
                ratio=ratio,
                metric=metric,
                margin=self.margin,
                class_token=self._pitome_info["class_token"],
                alpha=self._pitome_info["alpha"]
            )
          
            if self._pitome_info["trace_source"]:
                self._pitome_info["source"] = merge_source(
                    merge, x, self._pitome_info["source"]
                )
                self._pitome_info["sources"].append(self._pitome_info["source"])

            if isolated_score is not None and self._pitome_info["size"] is not None:
                weight = self._pitome_info["size"] + isolated_score
                x, self._pitome_info["size"] = merge_wavg(merge, x, weight)
            else:
                weight = self._pitome_info["size"] 
                x, self._pitome_info["size"] = merge_wavg(merge, x, weight)
        return x

    def forward(self, x, register_hook=False):
        x = self.compress_x(x, x)
        x = x + self.drop_path(self.attn.forward_and_save_attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class PiToMeAttention(Attention):

    def forward_and_save_attn(self, x, register_hook=False):
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
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            # attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def make_pitome_class(transformer_class):
    class PiToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self,x, register_blk=-1):
            self._pitome_info["r"] = [self.r]* len(self.blocks) 
            self._pitome_info["ratio"] =[1.0] + [self.ratio] * (len(self.blocks)-1)
            self._pitome_info["size"] = None
            self._pitome_info["source"] = None
            self._pitome_info["attn"] = []
            self._pitome_info["sources"] = []
            self.total_flop = 0
            self.final_shape = 0
            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

            x = x + self.pos_embed[:, : x.size(1), :]
            x = self.pos_drop(x)

            for i, blk in enumerate(self.blocks):
                self.total_flop += self.calculate_block_flop(x.shape)
                x = blk(x)
            x = self.norm(x)
            self.final_shape = x.shape
            return x

        def forward_features(self, x, register_blk=-1) -> torch.Tensor:
      
            self._pitome_info["r"] = [self.r]* len(self.blocks) 
            self._pitome_info["ratio"] = [self.ratio] * len(self.blocks) 
            self._pitome_info["size"] = None
            self._pitome_info["source"] = None
            self.total_flop = 0
            self.final_shape= None 

            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

            x = x + self.pos_embed[:, : x.size(1), :]
            x = self.pos_drop(x)

            for i, blk in enumerate(self.blocks):
                self.total_flop += self.calculate_block_flop(x.shape)
                x = blk(x) 
            x = self.norm(x)
            self.final_shape = x.shape

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
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, margin=None, use_k=False, output_attn=False, alpha=1.0):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._pitome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    PiToMeVisionTransformer = make_pitome_class(model.__class__)
    print('using', 'pitome')

    model.__class__ = PiToMeVisionTransformer
    model.ratio = 1.0 
    model.r=0.0
    
    # model.compress_method = 'tome' 
    model._pitome_info = {
        "ratio": model.ratio,
        "margin": [],
        "size": None,
        "source": None,
        "output_attn": output_attn,
        "attn": [],
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
        "alpha": alpha,
    }
    current_layer = 0
    num_layers = len(model.blocks)
    # margins = [margin - margin*(i/num_layers) for i in range(num_layers)]
    if margin is None:
        margins = [0.9 - .9*(i/num_layers) for i in range(num_layers)]
    else:
        margins = [margin for i in range(num_layers)]

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._pitome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = PiToMeBlock
            module.init_margin(margins[current_layer])
            module._pitome_info = model._pitome_info
            current_layer +=1
        elif isinstance(module, Attention):
            module.__class__ = PiToMeAttention
