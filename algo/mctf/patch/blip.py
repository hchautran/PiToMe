import torch
from lavis.models.vit import VisionTransformer, Attention, Block
from ..merge import merge_source, bipartite_soft_matching, merge_wavg

class MCTFBlock(Block):
    """
    Modifications:
     - Apply MCTF between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    
    def compress_x(self, metric, x):
        ratio = self._mctf_info["ratio"].pop()
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
            attn=self.attn.attention_map,
            size=self._mctf_info["size"]
        )
        return x

    def forward(self, x, register_hook=False):
        self.attn(self.norm1(x), register_hook=True)
        x = self.compress_x(x, x) 
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class MCTFAttention(Attention):
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
        return x

def make_mctf_class(transformer_class):
    class MCTFVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
        def forward(self,x, register_blk=-1):
            self._mctf_info["r"] = [self.r]* len(self.blocks) 
            self._mctf_info["ratio"] = [1.0] + [self.ratio] * (len(self.blocks)-1)
            self._mctf_info["size"] = None
            self._mctf_info["source"] = None
            self.total_flop = 0
            self.final_shape = None
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
                x = blk(x, self._mctf_info["output_attn"])
            x = self.norm(x)
            self.final_shape = x.shape 
            return x


        def forward_features(self, x, register_blk=-1) -> torch.Tensor:
      
            self._mctf_info["r"] = [self.r]* len(self.blocks) 
            self._mctf_info["ratio"] = [1.0] + [self.ratio] * (len(self.blocks) -1)
            self._mctf_info["size"] = None
            self._mctf_info["source"] = None
            self.total_flop = 0
            self.final_shape= 0

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
                x = blk(x, self._mctf_info["output_attn"])
            self.final_shape = x.shape 
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

    return MCTFVisionTransformer


def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, margin=0.9, use_k=False, output_attn=False):
    """
    Applies MCTF to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._mctf_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    MCTFVisionTransformer = make_mctf_class(model.__class__)
    print('using', 'mctf')

    model.__class__ = MCTFVisionTransformer
    model.ratio = 1.0 
    model.r=0.0
    default=[0.35, 0.15, 0, 1, 1, 1, 20, 40, 1, 1, 0]
    model._mctf_info = {
        "trace_source"   : False,
        "prop_attn"      : 1,
        "one_step_ahead" : 1,
        "tau_sim"        : 1,
        "tau_info"       : 20,
        "tau_size"       : 40,
        "bidirection"    : 1,
        "pooling_type"   : 0,
        "size": None,
        "class_token"  : model.cls_token is not None,
        "output_attn": output_attn,
    }


    current_layer = 0
    margin = margin 
    num_layers = len(model.blocks)
    # margins = [margin - margin*(i/num_layers) for i in range(num_layers)]

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._mctf_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            # module.__class__ = MCTFBlock if compress_method == 'mctf' else PiMCTFBlock 
            module.__class__ = MCTFBlock
            module._mctf_info = model._mctf_info
            current_layer +=1
        elif isinstance(module, Attention):
            module.__class__ = MCTFAttention 
