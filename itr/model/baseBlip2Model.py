import torch
import torch.nn as nn
# from model.manifolds.lorentz import Lorentz 
from typing import  Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPOutput
import torch.nn.functional as F


EUCLID = 'euclidean'
POINCARE = 'poincare'
LORENTZ = 'lorentz'


class BaseModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model_ckt = config.model_ckt
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.temp = nn.Parameter(torch.tensor(config.temp))
        self.logit_scale = nn.Parameter(torch.tensor(config.temp))
        self.weight_i2t = self.config.weight_i2t 

        self.model = None 

    def num_parameters(self, only_trainable=True):
        num_params = 0
        if only_trainable:
            num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in self.parameters())
        return num_params

        

    def itm_loss(self, input_ids, attention_mask, vit_embeds, sim_t2i, sim_i2t):
        text_input_ids_world = input_ids
        bs = input_ids.size(0)
        text_attention_mask_world = attention_mask
        image_embeds_world = vit_embeds
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            mask = (torch.eye(bs) > 0).to(self.device)
            weights_i2t.masked_fill_(mask, 0)
            weights_t2i.masked_fill_(mask, 0) 

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [input_ids, input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [attention_mask, attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.model.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            vit_embeds.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [vit_embeds, image_embeds_neg, vit_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            vit_embeds.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(vit_embeds.device)
        loss_itm = F.cross_entropy(logits, itm_labels)
        return loss_itm
        
    def itc_loss(self, image_embeds , text_embeds):

        sim_q2t = torch.matmul(
            image_embeds.unsqueeze(1), text_embeds.unsqueeze(-1)
        ).squeeze()
        sim_i2t, _ = sim_q2t.max(-1)

        # text-query similarity: [batch_size, batch_size, num_query_tokens]
        sim_t2q = torch.matmul(
            text_embeds.unsqueeze(1).unsqueeze(1), image_embeds.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)

        bs = image_embeds.size(0)
        targets = torch.arange(bs).to(self.device)

        itc_loss= (
            F.cross_entropy(sim_i2t/self.temp, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i/ self.temp , targets, label_smoothing=0.1)
        ) / 2
        stats = {
            "logits/weight_t2i": 1.0 - self.weight_i2t,
            "logits/itc_loss": itc_loss.item(),
            "logits/min": sim_i2t.min().item(),
            "logits/mean": sim_i2t.mean().item(),
            "logits/max": sim_i2t.max().item(),
            "logits/acc": (sim_i2t.argmax(-1) == targets).float().mean().item(),
        }
        return itc_loss, stats, sim_i2t
    
    def postprocess_embeds(self, embed):
        return F.normalize(embed, p=2, dim=-1) 

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_id: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CLIPOutput]:

        vision_outputs = self.model(
            pixel_values=pixel_values,
        )

        text_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]
        image_embeds = F.normalize(image_embeds,p=2, dim=-1) 
        text_embeds = F.normalize(text_embeds,p=2, dim=-1) 
        itc_loss, stats, sims_i2t = self.itc_loss(image_embeds, text_embeds)
        # itm_loss = self.itm_loss(input_ids=input_ids, attention_mask=attention_mask, vit_embeds=vision_outputs[0], sim_i2t=sims_i2t, sim_t2i=sims_i2t.T)
        # stats["logits/itm_loss"] = itm_loss.item() 
        # loss = itm_loss + itc_loss 
        return itc_loss, stats

    def get_text_features(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
    ):
        text_output = self.model(
           input_ids=input_ids, 
           attention_mask=attention_mask, 
        )
        text_feat = F.normalize(text_output[1], dim=-1, p=2)
        return text_feat, text_output[0]

    def get_vision_features(self, pixel_values: torch.Tensor, return_source=False):
        image_output = self.model.get_vision_features(pixel_values=pixel_values, return_source=return_source)
        image_feat = F.normalize(image_output[1], dim=-1, p=2)
        return image_feat, image_output[0], image_output[3], image_output[4], image_output[5]

