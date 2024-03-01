import torch
import torch.nn as nn


class ThresholdMasker(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau
        self.threshold = nn.Parameter(torch.empty(1))

    def forward(self, values):
        soft_mask = torch.sigmoid((values - self.threshold) / self.tau)
        hard_mask = (soft_mask > 0.5).float()
        ret = soft_mask + (hard_mask - soft_mask).detach()
        return ret


class InferenceThresholdMasker(ThresholdMasker):
    def forward(self, values):
        return values > self.threshold


def softmax_with_mask(attn, mask, training, eps=1e-5):
    attn_mask = mask[:, None, None, :] * torch.ones_like(attn)
    max_att = torch.max(attn, dim=-1, keepdim=True)[0]
    attn = attn - max_att
    attn = attn.exp_() * attn_mask
    if training:
        # Add epsilon for numeric stability for training
        attn = (attn + eps / attn.shape[-1]) / (attn.sum(dim=-1, keepdim=True) + eps)
    else:
        attn = attn / attn.sum(dim=-1, keepdim=True)
    return attn


