from lavis.models.clip_models.model import VisualTransformer, ResidualAttentionBlock
from typing import Optional, Callable
import torch.nn as nn
import torch
from ..ddp import DiffRate
from ..merge import get_merge_func


class DiffRateBlock(ResidualAttentionBlock):
    def introduce_diffrate(self,patch_number, prune_granularity, merge_granularity):
        self.prune_ddp = DiffRate(patch_number,prune_granularity)
        self.merge_ddp = DiffRate(patch_number,merge_granularity)


    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x_attn, attn = self.attn(x, x, x, need_weights=True, attn_mask=attn_mask)
        return x_attn, attn
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        N, B, C = x.shape
        size = self._diffrate_info["size"]
        mask = self._diffrate_info["mask"]
        attn_x, attn = self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + attn_x 
        x.transpose_(1,0)
        # importance metric
        cls_attn = attn[:, 0, 1:]
        _, idx = torch.sort(cls_attn, descending=True)
        cls_index = torch.zeros((B,1), device=idx.device).long()
        idx = torch.cat((cls_index, idx+1), dim=1)

        # sorting
        x = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        self._diffrate_info["size"] = torch.gather(self._diffrate_info["size"], dim=1, index=idx.unsqueeze(-1))
        mask = torch.gather( mask, dim=1, index=idx)
        if self._diffrate_info["trace_source"]:
            self._diffrate_info["source"] = torch.gather(self._diffrate_info["source"], dim=1, index=idx.unsqueeze(-1).expand(-1, -1, self._diffrate_info["source"].shape[-1]))

        if self.training:
            # pruning, pruning only needs to generate masks during training
            last_token_number = mask[0].sum().int()
            prune_kept_num = self.prune_ddp.update_kept_token_number()      # expected prune compression rate, has gradiet
            self._diffrate_info["prune_kept_num"].append(prune_kept_num)
            if prune_kept_num < last_token_number:        # make sure the kept token number is a decreasing sequence
                prune_mask = self.prune_ddp.get_token_mask(last_token_number)
                mask = mask * prune_mask.expand(B, -1)
            mid_token_number = min(last_token_number, int(prune_kept_num)) # token number after pruning
                
            # merging
            merge_kept_num = self.merge_ddp.update_kept_token_number()
            self._diffrate_info["merge_kept_num"].append(merge_kept_num)

            if merge_kept_num < mid_token_number:
                merge_mask = self.merge_ddp.get_token_mask(mid_token_number)
                x_compressed, size_compressed = x[:, mid_token_number:], self._diffrate_info["size"][:,mid_token_number:]
                merge_func, node_max = get_merge_func(metric=x[:, :mid_token_number].detach(), kept_number=int(merge_kept_num))
                x = merge_func(x[:,:mid_token_number],  mode="mean", training=True)
                # optimize proportional attention in ToMe by considering similarity
                size = torch.cat((self._diffrate_info["size"][:, :int(merge_kept_num)],self._diffrate_info["size"][:, int(merge_kept_num):mid_token_number]*node_max[..., None]),dim=1)
                size = size.clamp(1)
                size = merge_func(size,  mode="sum", training=True)
                x = torch.cat([x, x_compressed], dim=1)
                self._diffrate_info["size"] = torch.cat([size, size_compressed], dim=1)
                mask = mask * merge_mask
            self._diffrate_info["mask"] = mask
        else:
             # pruning
            prune_kept_num = self.prune_ddp.kept_token_number
            x = x[:, :prune_kept_num]
            self._diffrate_info["size"] = self._diffrate_info["size"][:, :prune_kept_num]
            if self._diffrate_info["trace_source"]:
                self._diffrate_info["source"] = self._diffrate_info["source"][:, :prune_kept_num]
            # merging
            merge_kept_num = self.merge_ddp.kept_token_number
            if merge_kept_num < prune_kept_num:
                merge,node_max = get_merge_func(x.detach(), kept_number=merge_kept_num)
                x = merge(x,mode='mean')
                self._diffrate_info["size"] = torch.cat((self._diffrate_info["size"][:, :merge_kept_num],self._diffrate_info["size"][:, merge_kept_num:]*node_max[..., None] ),dim=1)
                self._diffrate_info["size"] = merge(self._diffrate_info["size"], mode='sum')
                if self._diffrate_info["trace_source"]:
                    self._diffrate_info["source"] = merge(self._diffrate_info["source"], mode="amax")
        x.transpose_(1,0)
        x = x + self.mlp(self.ln_2(x))
        return x


class DiffRateTransformer(VisualTransformer):

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        N, _= self.positional_embedding.data.shape
        self._diffrate_info["size"] = torch.ones([B,N +1,1], device=x.device)
        self._diffrate_info["mask"] =  torch.ones((B,N+1),device=x.device)
        self._diffrate_info["prune_kept_num"] = []
        self._diffrate_info["merge_kept_num"] = []
        if self._diffrate_info["trace_source"]:
            self._diffrate_info["source"] = torch.eye(self.patch_embed.num_patches+1, device=x.device)[None, ...].expand(B, self.patch_embed.num_patches+1, self.patch_embed.num_patches+1)

        self.transformer.total_flop = 0
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for r in self.transformer.resblocks:
            self.transformer.total_flop += self.calculate_block_flop(x.shape)
            x = r(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


    def calculate_block_flop(self, shape):
            flops = 0
            N ,_, C = shape
            mhsa_flops = 4*N*C*C + 2*N*N*C
            flops += mhsa_flops
            ffn_flops = 8*N*C*C
            flops += ffn_flops
            return flops
    
    
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
        for block in self.transformer.resblocks:
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
        N, _= self.positional_embedding.data.shape
        for block in self.transformer.resblocks:
            r = math.floor(N - N*ratio)
            block.prune_ddp.kept_token_number = N - 0 
            block.merge_ddp.kept_token_number = N - r
            N -= r
        
    def init_kept_num_using_r(self, r):
        N, _= self.positional_embedding.data.shape
        for block in self.transformer.resblocks:
            r = min(r, N // 2)
            block.prune_ddp.kept_token_number = N - 0 
            block.merge_ddp.kept_token_number = N - r
            N -= r


def apply_patch(
    model: VisualTransformer, trace_source: bool = False,prune_granularity=1, merge_granularity=1):
    """
    Applies DiffRate to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._diffrate_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    print('using', 'diffrate')

    model.__class__ = DiffRateTransformer 
    model.ratio = 1.0 
    model.r=0.0
    
    # model.compress_method = 'diffrate' 
    model._diffrate_info = {
        "ratio": model.ratio,
        "margin":  [],
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": True,
        "class_token": True,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._diffrate_info["distill_token"] = True

    block_index = 0
    non_compressed_block_index = [0]
    for module in model.modules():
        if isinstance(module, ResidualAttentionBlock):
            # module.__class__ = DiffRateBlock if compress_method == 'diffrate' else DiffRateBlock 
            module.__class__ = DiffRateBlock
            shape = model.positional_embedding.data.shape
            if block_index in non_compressed_block_index:
                module.introduce_diffrate(shape[0], shape[0]+1, shape[0]+1)
            else:
                module.introduce_diffrate(shape[0], prune_granularity, merge_granularity)
            block_index += 1
            module._diffrate_info = model._diffrate_info
