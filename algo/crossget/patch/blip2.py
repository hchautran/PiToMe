import torch
import torch.nn.functional as F 
import torch.utils.checkpoint as checkpoint
from lavis.models.eva_vit import VisionTransformer, Block, Attention
from ..merge import merge_source, crossget, merge_wavg

class CrossGetBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    
    def compress_x(self, metric, x, attn):
        ratio = self._info["ratio"]
        if ratio < 1.0:
            merge = crossget(
                ratio=ratio,
                metric=metric,
                class_token=self._info["class_token"]
            )

            if self._info["trace_source"]:
                self._info["source"] = merge_source(
                    merge, x, self._info["source"]
                )

    
            weight = self._info["size"] 
            x, self._info["size"] = merge_wavg(merge, x, weight )
        return x 


    def forward(self, x, rel_pos_bias=None):
        attn_size = self._info["size"] if self._info["prop_attn"] else None
        if self.gamma_1 is None:
            x_attn, metric, attn = self.attn(self.norm1(x), attn_size, rel_pos_bias=rel_pos_bias)
            x = x + self.drop_path(x_attn)
            # x, attn_size = self.compress_x(x,x, attn)
            x, _ = self.compress_x(x=x,metric=x, attn=attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_attn, metric, attn = self.attn(self.norm1(x), attn_size, rel_pos_bias=rel_pos_bias)
            x = x + self.drop_path(x_attn)
            x, _ = self.compress_x(x=x,metric=x, attn=attn)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CrossGetAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(self, x:torch.Tensor, isolation_score: torch.Tensor = None, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if isolation_score is not None:
            attn = attn +  isolation_score.log()[:, None, None, :, 0]

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k.mean(1), attn


def make_cross_get_class(transformer_class):
    class CrossGetVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x) -> torch.Tensor:
      
            self._info["ratio"] = self.ratio
            self._info["size"] = None
            self._info["source"] = None
            self.total_flop = 0
            self.final_shape = 0

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
                self.total_flop += self.calculate_block_flop(x.shape)
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

    return CrossGetVisionTransformer



def apply_patch(
   model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True):

    CrossGetVisionTransformer = make_cross_get_class(model.__class__)
    print('using', 'cross_get')

    model.__class__ = CrossGetVisionTransformer
    model.ratio = 1.0 
    
    # model.compress_method = 'tome' 
    model._info = {
        "ratio": model.ratio,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": True,
        "class_token": True,
        "distill_token": False,
    }
    current_layer = 0

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            # module.__class__ = ToMeBlock if compress_method == 'tome' else CrossGetBlock 
            module.__class__ = CrossGetBlock
            module._info = model._info
            current_layer +=1
        elif isinstance(module, Attention):
            module.__class__ = CrossGetAttention
