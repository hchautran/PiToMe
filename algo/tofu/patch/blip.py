import torch
from lavis.models.vit import VisionTransformer, Attention, Block
from ..merge import merge_source, bipartite_soft_matching, merge_wavg

class ToFuBlock(Block):
    """
    Modifications:
     - Apply ToFu between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def init_strategy(self, strategy='mean'):
        # self.margin = nn.Parameter(torch.tensor(margin)) 
        self.strategy = strategy 

    
    def compress_x(self, metric, x):
        ratio = self._tofu_info["ratio"].pop()
        if ratio < 1.0:
            merge, isolated_score = bipartite_soft_matching(
                ratio=ratio,
                metric=metric,
                class_token=self._tofu_info["class_token"]
            )

            if self._tofu_info["trace_source"]:
                self._tofu_info["source"] = merge_source(
                    merge, x, self._tofu_info["source"]
                )
            x = merge(x, mode=self.strategy)
        return x

    def forward(self, x, register_hook=False):
        # attn_size = self._tofu_info["size"] if self._tofu_info["prop_attn"] else None
        # x_attn, metric, attn = self.attn(self.norm1(x), register_hook=register_hook)
        # x = x + self.drop_path(x_attn)
        # x = self.compress_x(metric, x) 
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # return x
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.compress_x(x, x) 
        # print(x.shape)
        return x




class ToFuAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """


    def forward(self, x, register_hook=False):
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
        return x, k.mean(1), attn

def make_tofu_class(transformer_class):
    class ToFuVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
        def forward(self,x, register_blk=-1):
            self._tome_info["r"] = [self.r]* len(self.blocks) 
            self._tome_info["ratio"] = [self.ratio] * len(self.blocks) 
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self.total_flop = 0
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
                x = blk(x, register_blk == i)
            x = self.norm(x)
            return x

        def forward_features(self, x, register_blk=-1) -> torch.Tensor:
      
            self._tofu_info["r"] = [self.r]* len(self.blocks) 
            self._tofu_info["ratio"] = [self.ratio] * len(self.blocks) 
            self._tofu_info["size"] = None
            self._tofu_info["source"] = None
            self.total_flop = 0

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
                x = blk(x, register_blk == i)
            x = self.norm(x)
            return x


        def calculate_block_flop(self, shape):
            flops = 0
            _, N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops

    return ToFuVisionTransformer


def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, margin=0.9, use_k=False):
    """
    Applies ToFu to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tofu_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToFuVisionTransformer = make_tofu_class(model.__class__)
    print('using', 'tofu')

    model.__class__ = ToFuVisionTransformer
    model.ratio = 1.0 
    model.r=0.0
    
    # model.compress_method = 'tofu' 
    model._tofu_info = {
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
    num_layers = len(model.blocks)
    # margins = [margin - margin*(i/num_layers) for i in range(num_layers)]
    strategies = ['tofu' if i > num_layers //2 else 'prune' for i in range(num_layers)]

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tofu_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToFuBlock
            module._tofu_info = model._tofu_info
            module.init_strategy(strategies[current_layer])
            current_layer +=1
