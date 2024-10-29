import torch
import torch.nn.functional as F 
import torch.utils.checkpoint as checkpoint
from lavis.models.eva_vit import VisionTransformer,Attention,Block
# import DiffRate.ddp as ddp
from ..ddp import DiffRate
from ..merge import get_merge_func


class DiffRateBlock(Block):
    """
    Modifications:
     - Apply DiffRate between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def introduce_diffrate(self,patch_number, prune_granularity, merge_granularity):
        self.prune_ddp = DiffRate(patch_number,prune_granularity)
        self.merge_ddp = DiffRate(patch_number,merge_granularity)


    def compress_x(self, x, attn, mask):
        B, _, _ = x.shape
        # importance metric
        cls_attn = attn[:, :, 0, 1:]
        cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
        _, idx = torch.sort(cls_attn, descending=True)
        cls_index = torch.zeros((B,1), device=idx.device).long()
        idx = torch.cat((cls_index, idx+1), dim=1)
        
        # sorting
        x = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        self._info["size"] = torch.gather(self._info["size"], dim=1, index=idx.unsqueeze(-1))
        mask = torch.gather( mask, dim=1, index=idx)
        if self._info["trace_source"]:
            self._info["source"] = torch.gather(self._info["source"], dim=1, index=idx.unsqueeze(-1).expand(-1, -1, self._info["source"].shape[-1]))

        if self.training:
        # pruning, pruning only needs to generate masks during training
            last_token_number = mask[0].sum().int()
            prune_kept_num = self.prune_ddp.update_kept_token_number()      # expected prune compression rate, has gradiet
            self._info["prune_kept_num"].append(prune_kept_num)
            if prune_kept_num < last_token_number:        # make sure the kept token number is a decreasing sequence
                prune_mask = self.prune_ddp.get_token_mask(last_token_number)
                mask = mask * prune_mask.expand(B, -1)

            mid_token_number = min(last_token_number, int(prune_kept_num)) # token number after pruning
                
            # merging
            merge_kept_num = self.merge_ddp.update_kept_token_number()
            self._info["merge_kept_num"].append(merge_kept_num)

            if merge_kept_num < mid_token_number:
                merge_mask = self.merge_ddp.get_token_mask(mid_token_number)
                x_compressed, size_compressed = x[:, mid_token_number:], self._info["size"][:,mid_token_number:]
                merge_func, node_max = get_merge_func(metric=x[:, :mid_token_number].detach(), kept_number=int(merge_kept_num))
                x = merge_func(x[:,:mid_token_number],  mode="mean", training=True)
                # optimize proportional attention in ToMe by considering similarity
                size = torch.cat((self._info["size"][:, :int(merge_kept_num)],self._info["size"][:, int(merge_kept_num):mid_token_number]*node_max[..., None]),dim=1)
                size = size.clamp(1)
                size = merge_func(size,  mode="sum", training=True)
                x = torch.cat([x, x_compressed], dim=1)
                self._info["size"] = torch.cat([size, size_compressed], dim=1)
                mask = mask * merge_mask

            self._info["mask"] = mask
        
        else:
            # pruning
            prune_kept_num = self.prune_ddp.kept_token_number
            x = x[:, :prune_kept_num]
            self._info["size"] = self._info["size"][:, :prune_kept_num]
            if self._info["trace_source"]:
                self._info["source"] = self._info["source"][:, :prune_kept_num]
                
            
            # merging
            merge_kept_num = self.merge_ddp.kept_token_number
            if merge_kept_num < prune_kept_num:
                merge,node_max = get_merge_func(x.detach(), kept_number=merge_kept_num)
                x = merge(x,mode='mean')
                # optimize proportional attention in ToMe by considering similarity, this is benefit to the accuracy of off-the-shelf model.
                self._info["size"] = torch.cat((self._info["size"][:, :merge_kept_num],self._info["size"][:, merge_kept_num:]*node_max[..., None] ),dim=1)
                self._info["size"] = merge(self._info["size"], mode='sum')
                if self._info["trace_source"]:
                    self._info["source"] = merge(self._info["source"], mode="amax")
        return x


    def forward(self, x, rel_pos_bias=None):
        size = self._info["size"]
        mask = self._info["mask"]
        if self.gamma_1 is None:
            x_attn, attn = self.attn(self.norm1(x), size=size, rel_pos_bias=rel_pos_bias, mask=mask)
            x = x + self.drop_path(x_attn)
            x = self.compress_x(x, attn, mask)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_attn, attn = self.attn(self.norm1(x),  size=size, rel_pos_bias=rel_pos_bias, mask=mask)
            x = x + self.drop_path(x_attn)
            x = self.compress_x(x, attn, mask)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class DiffRateAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """
    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x:torch.Tensor, size: torch.Tensor = None, rel_pos_bias=None, mask:torch.Tensor=None):
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
        if size is not None:
            attn = attn +  size.log()[:, None, None, :, 0]



        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        if self.training:
            attn = self.softmax_with_policy(attn, mask)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


def make_pidiffrate_class(transformer_class):
    class DiffRateVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, x) -> torch.Tensor:
      
            B = x.shape[0]
            self._info["size"] = torch.ones([B,self.patch_embed.num_patches+1,1], device=x.device)
            self._info["mask"] =  torch.ones((B,self.patch_embed.num_patches+1),device=x.device)
            self._info["prune_kept_num"] = []
            self._info["merge_kept_num"] = []
            if self._info["trace_source"]:
                self._info["source"] = torch.eye(self.patch_embed.num_patches+1, device=x.device)[None, ...].expand(B, self.patch_embed.num_patches+1, self.patch_embed.num_patches+1)
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
            self.final_shape = x.shape
            return x

        def parameters(self):
            # original network parameter
            params = []
            for n, m in self.named_parameters():
                if n.find('ddp') > -1:
                    continue
                params.append(m)
            return iter(params)     

        def arch_parameters(self):
            params = []
            for n, m in self.named_parameters():
                if n.find('ddp') > -1:
                    params.append(m)
            return iter(params)    

        def get_kept_num(self):
            prune_kept_num = []
            merge_kept_num = []
            for block in self.blocks:
                prune_kept_num.append(int(block.prune_ddp.kept_token_number))
                merge_kept_num.append(int(block.merge_ddp.kept_token_number))
            return prune_kept_num, merge_kept_num
                

        def set_kept_num(self, prune_kept_numbers, merge_kept_numbers):
            assert len(prune_kept_numbers) == len(self.blocks) and len(merge_kept_numbers) == len(self.blocks)
            for block, prune_kept_number, merge_kept_number in zip(self.blocks, prune_kept_numbers, merge_kept_numbers):
                block.prune_ddp.kept_token_number = prune_kept_number
                block.merge_ddp.kept_token_number = merge_kept_number
        
        def init_kept_num_using_ratio(self, ratio):
            import math
            N = self.patch_embed.num_patches
            for block in self.blocks:
                r = math.floor(N - N*ratio)
                block.prune_ddp.kept_token_number = N - 0 
                block.merge_ddp.kept_token_number = N - r
                N -= r
            
        def init_kept_num_using_r(self, r):
            N = self.patch_embed.num_patches
            for block in self.blocks:
                r = min(r, N // 2)
                block.prune_ddp.kept_token_number = N - 0 
                block.merge_ddp.kept_token_number = N - r
                N -= r
        
         
        def calculate_block_flop(self, shape):
            flops = 0
            _, N, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops
 
    return DiffRateVisionTransformer



def apply_patch(
    model: VisionTransformer, trace_source: bool = False,prune_granularity=1, merge_granularity=1
):
    """
    Applies DiffRate to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    DiffRateVisionTransformer = make_pidiffrate_class(model.__class__)
    print('using', 'diffrate')

 
    model.__class__ = DiffRateVisionTransformer
    model._info = {
        "size": None,
        "mask": None,           # only for training
        "source": None,
        "class_token": False, 
        "trace_source": trace_source,
    }

    block_index = 0
    non_compressed_block_index = [0]
    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = DiffRateBlock
            if block_index in non_compressed_block_index:
                module.introduce_diffrate(model.patch_embed.num_patches, model.patch_embed.num_patches+1, model.patch_embed.num_patches+1)
            else:
                module.introduce_diffrate(model.patch_embed.num_patches, prune_granularity, merge_granularity)
            block_index += 1
            module._info = model._info
        elif isinstance(module, Attention):
            module.__class__ = DiffRateAttention

 
