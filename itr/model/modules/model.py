
import torch
import torch.nn as nn
from typing import Optional
from transformers import CLIPModel

class CLIPEncoder(nn.Module): 
    def __init__(self, config ,body, head, manifold_mapper=None) -> None:
        super().__init__()
        self.body = body
        self.head = head 
        self.manifold_mapper = manifold_mapper
      
        self.config = config
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
    
        if pixel_values is not None:
            outputs = self.body(
                pixel_values=pixel_values,
                return_dict=True,
            )

        else:
            outputs = self.body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )

        pooled_output = outputs[1]
        last_hidden_state = outputs[0]
        pooled_output = self.head(pooled_output)
        if self.manifold_mapper is not None:
            pooled_output = self.manifold_mapper(pooled_output, use_normalized=True)

        return last_hidden_state, pooled_output 



class BLIPEncoder(nn.Module): 
    def __init__(self, config ,body, head, manifold_mapper=None) -> None:
        super().__init__()
        self.body = body
        self.head = head 
        self.manifold_mapper = manifold_mapper
      
        self.config = config
        
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
    
        if pixel_values is not None:
            outputs = self.body(
                pixel_values=pixel_values,
                return_dict=True,
            )

        else:
            outputs = self.body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:,0,:]
        pooled_output = self.head(pooled_output)
        if self.manifold_mapper is not None:
            pooled_output = self.manifold_mapper(pooled_output, use_normalized=(pixel_values is None))

        return last_hidden_state, pooled_output 
