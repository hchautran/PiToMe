import torch
import torchvision.transforms as transforms
from torchvision import datasets

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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            step+=1

    accuracy = correct / total
    print(f'Accuracy on the test set: {100 * accuracy:.2f}%')




def get_processor(model):
    input_size = model.default_cfg["input_size"][1]
    return transforms.Compose([
        transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
    ])

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

VIT_S_16="vit_small_patch16_224"
VIT_B_16="vit_base_patch16_224"
VIT_L_16="vit_large_patch16_224"
VIT_L_16_384="vit_large_patch16_384"
DEIT_T_16="deit_tiny_patch16_224"
DEIT_S_16="deit_small_patch16_224"
DEIT_B_16="deit_base_patch16_224"
TOME = 'tome'
PITOME = 'pitome'


if __name__ == '__main__':

    model_ckt = VIT_L_16_384
    model = timm.create_model(model_ckt, pretrained=True).to(device)
    # tome.patch.timm(model, TOME)
    tome.patch.timm(model, PITOME)
    # model.r=7
    model.r=0.924
    processor = get_processor(model)

    dataset = load_dataset("imagenet-1k", split='validation', cache_dir="/mnt/data/mount_4TBSSD/nmduy/imagenet/")
    val_dataloader = DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=lambda batch: process_image(batch, processor))
    evaluate(model, val_dataloader, device, max_step=50)
    