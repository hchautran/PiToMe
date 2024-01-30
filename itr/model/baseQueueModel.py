from copy import deepcopy

import torch
import torch.nn.functional as F
from lavis.models.base_model import (
    MomentumDistilationMixin,
    SharedQueueMixin,
)
from lavis.models.blip_models.blip import BlipBase
from lavis.models import BlipRetrieval 
from torch import nn

class Text(object):
    pass


class BaseModelWithQueue(BlipBase, MomentumDistilationMixin, SharedQueueMixin):
    """
    BLIP retrieval model.

    Supported model types:
        - coco: fine-tuned BLIP base model on COCO dataset (Karpathy split).
        - flickr: fine-tuned BLIP base model on Flickr30k dataset.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_retrieval", "coco")
        >>> model = load_model("blip_retrieval", "flickr")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "coco": "configs/models/blip_retrieval_coco.yaml",
        "flickr": "configs/models/blip_retrieval_flickr.yaml",
    }

    def __init__(
        self,
        config,
    ):
        """ """
        super().__init__()
        self.config = config
        self.model_ckt = config.model_ckt
        self.clip_r = config.clip_radius
        self.queue_size = config.queue_size
        self.weight_i2t = config.weight_i2t
        class_weight = torch.tensor([1.0, 1.0])
        self.itm_criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')

        self.model_m = None 
        self.model = None 

        self.momentum = config.momentum
        self.logit_scale = nn.Parameter(torch.tensor(config.temp))
        self.eu_logit_scale = nn.Parameter(torch.tensor(config.temp))

        self.alpha = config.alpha
        self.max_txt_len = config.max_txt_len
    
    def _init_queue(self, config, ft_out):
        self.model_m= deepcopy(self.model) 
        self.model_pairs = [
            [self.model, self.model_m],
        ]
        self.copy_params()
          # create the queue
        self.register_buffer("image_queue", torch.randn(self.queue_size, ft_out).T)
        self.register_buffer("text_queue", torch.randn(self.queue_size, ft_out).T)
        self.image_queue = nn.functional.normalize(self.image_queue.T, dim=-1).T
        self.text_queue = nn.functional.normalize(self.text_queue.T, dim=-1).T
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (num_iters_per_epoch))
    
    
    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params
    


    def dist_func(self, x:torch.Tensor, y:torch.Tensor): 
        x = F.normalize(x,p=2, dim=-1) 
        y = F.normalize(y,p=2, dim=-1) 
        eu_dis = torch.matmul(x, y.T) 
        return  eu_dis 



    def forward(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        image_id: torch.FloatTensor,
        epoch: int=None,
        iters: int=None,
        num_iters_per_epoch:int=None,
    ):
        idx = image_id
    
        
        text_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        image_output = self.model(
            pixel_values=pixel_values,
        )
        text_feat = text_output[1] 
        image_feat = image_output[1] 
   

        bsize = text_feat.shape[0]

        # Image-text Contrastive Learning
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        with torch.no_grad():
            self.logit_scale.clamp_(0.001, 0.5)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.model_m(
                pixel_values=pixel_values, 
            )

            text_embeds_m = self.model_m(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            image_feat_m = image_embeds_m[1]
            text_feat_m = text_embeds_m[1]

            image_feat_m_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()], dim=1
            )

            text_feat_m_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()], dim=1
            )

        sim_i2t = self.dist_func(image_feat, text_feat_m_all.T) 
        sim_t2i = self.dist_func(text_feat, image_feat_m_all.T)


        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t / self.logit_scale, dim=1) * sim_targets, dim=-1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i / self.logit_scale, dim=1) * sim_targets, dim=-1
        ).mean()      

        loss_itc = self.config.weight_i2t * (loss_i2t) + (1-self.config.weight_i2t) * (loss_t2i)
      

        sims = self.dist_func(image_feat, text_feat)
     
        

        in_batch_target = torch.arange(bsize).to(self.device)
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/itc_loss": loss_itc.item(),
            "logits/min": sims.min().item(),
            "logits/mean": sims.mean().item(),
            "logits/max": sims.max().item(),
            "logits/acc": (sims.argmax(-1) == in_batch_target).float().mean().item(),
            "logits/saved_memory": image_output[4],
        }

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
        loss = loss_itc 
        return  loss, stats

    def reset_queue_ptr(self):
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

    def get_text_features(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
    ):
        text_output = self.model(
           input_ids=input_ids, 
           attention_mask=attention_mask, 
        )
        text_feat = F.normalize(text_output[1], p=2, dim=-1)
        return text_feat, text_output[0]

    def get_vision_features(self, pixel_values: torch.Tensor, return_source:bool=False):
        image_output = self.model.get_vision_features(pixel_values=pixel_values, return_source=return_source)
        image_feat = F.normalize(image_output[1], dim=-1, p=2)
        return image_feat, image_output[0], image_output[3], image_output[4], image_output[5]
    
