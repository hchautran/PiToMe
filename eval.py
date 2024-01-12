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

# Set the device (GPU if available, otherwise use CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model_name = "vit_huge_patch16_224"
model_name = "vit_base_patch16_224"
# model_name = "vit_large_patch16_384"


# Load a pretrained model
model = timm.create_model(model_name, pretrained=True).to(device)
TOME = 'tome'
PITOME = 'pitome'
tome.patch.timm(model, TOME)
# tome.patch.timm(model, PITOME)
model.r=0.95
# model.compress_method='pitome'
input_size = model.default_cfg["input_size"][1]
# Define the transformation for the input images
transform = transforms.Compose([
    transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
])

def process_image(batch):
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


dataset = load_dataset("imagenet-1k", split='validation', cache_dir="/mnt/data/mount_4TBSSD/nmduy/imagenet/")
val_dataloader = DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=process_image)

evaluate(model, val_dataloader, device, max_step=None)
