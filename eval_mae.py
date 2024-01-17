import torch
import torchvision.transforms as transforms
import sys
sys.path.append('./mae')

import models_mae
from datasets import load_dataset
import timm
import timm.models.vision_transformer
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import tome
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader, device, max_step:int=None):
    model.eval()
    correct = 0
    total = 0
    step = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if max_step is not None and step >= max_step: break
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            print(outputs.shape)
            _, predicted = torch.max(outputs.softmax(dim=1), -1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            step+=1

    accuracy = correct / total
    print(f'Accuracy on the test set: {100 * accuracy:.2f}%')




def get_processor(model):
    input_size = model.default_cfg["input_size"][1]
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return transforms

def process_image(batch, transform):
    images = []
    labels = []
    for item in batch:
        try:
            images.append(transform(item['image']).unsqueeze(0))
            labels.append(item['label'])
        except:
            pass
    images_tensor = torch.cat(images)
    labels_tensor = torch.tensor(labels)

    return {'image': images_tensor, 'label': labels_tensor}

VIT_B_16="vit_base_patch16_224.mae"
VIT_L_16="vit_large_patch16_224.mae"
VIT_H_14="vit_huge_patch14_224.mae"
TOME = 'tome'
PITOME = 'pitome'


if __name__ == '__main__':

    model_ckt = VIT_B_16 
    model = timm.create_model(model_ckt, pretrained=True).to(device)
    tome.patch.timm(model, TOME)
    # tome.patch.mae(model, PITOME)
    # model.r=7
    model.r=0.90
    processor = get_processor(model)
    dataset = load_dataset("imagenet-1k", split='validation', cache_dir="/mnt/data/mount_4TBSSD/nmduy/imagenet/")
    val_dataloader = DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=lambda batch: process_image(batch, processor))


    evaluate(model, val_dataloader, device, max_step=50)
    