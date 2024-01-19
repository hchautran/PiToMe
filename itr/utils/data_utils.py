import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm
import numpy as np
from transformers import CLIPProcessor, BlipProcessor, AutoTokenizer, AutoImageProcessor 
from datasets import dataset_dict 
from datasets import load_dataset
from lavis.datasets.builders import load_dataset as lavis_dataset
from itertools import chain


def parse_int(sample):
    sample['img_id'] = int(sample['img_id'])
    return sample

class EvalDataset(Dataset):
    def __init__(self, dataset, vis_processor, tokenizer, txt_processor=None, use_tokenizer=False):  
        self.dataset = dataset
        self.txt_processor = txt_processor
        self.vis_processor = vis_processor 
        self.tokenizer = tokenizer 
        self.img2txt = dataset.img2txt
        self.txt2img = dataset.txt2img
        self.text = dataset.text
        self.image = dataset.image
        self.use_tokenizer = use_tokenizer

    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, index, use_tokenizer=False): 
        data =  self.dataset[index]
        cap_indexes = self.dataset.img2txt[index]
        captions = []
        output = {}
        for i in cap_indexes:
            if self.txt_processor is not None:
                captions.append(self.txt_processor(self.text[i]))
            else:
                captions.append(self.text[i])

        if self.use_tokenizer:
            if isinstance(self.tokenizer, CLIPProcessor) or isinstance(self.tokenizer, BlipProcessor):
                text_inputs = self.tokenizer(text=captions, max_length=35, truncation=True, padding=True ,return_tensors='pt') 
                output['pixel_values'] = self.vis_processor(images=data['image'], return_tensors='pt')['pixel_values']
            else:
                text_inputs = self.tokenizer(captions, max_length=35, truncation=True, padding=True ,return_tensors='pt') 
                output['pixel_values'] = self.vis_processor(data['image']).unsqueeze_(0)
            output['input_ids'] = text_inputs['input_ids']
            output['attention_mask'] = text_inputs['attention_mask']
        else:
            output['text_input'] = captions
            output['image'] = data['image']
        output['image_id'] = data['index']
        return output

class FuseEvalDataset(Dataset):
    def __init__(self, dataset, vis_processors, tokenizers, txt_processors=None):  
        self.dataset = dataset
        self.txt_processors = txt_processors
        self.vis_processors = vis_processors
        self.tokenizers = tokenizers 
        self.img2txt = dataset.img2txt
        self.txt2img = dataset.txt2img
        self.text = dataset.text
        self.image = dataset.image

    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, index): 
        data =  self.dataset[index]
        cap_indexes = self.dataset.img2txt[index]
        output = {}
        for i in range(len(self.txt_processors)):
            captions = []
            for j in cap_indexes:
                if self.txt_processors[i] is not None:
                    captions.append(self.txt_processors[i](self.text[j]))
                else:
                    captions.append(self.text[j])

            if isinstance(self.tokenizers[i], CLIPProcessor) or isinstance(self.tokenizers[i], BlipProcessor):
                text_inputs = self.tokenizers[i](text=captions, max_length=35, truncation=True, padding=True ,return_tensors='pt') 
                output[f'pixel_values_{i}'] = self.vis_processors[i](images=data['image'], return_tensors='pt')['pixel_values']
            else:
                text_inputs = self.tokenizers[i](captions, max_length=35, truncation=True, padding=True ,return_tensors='pt') 
                output[f'pixel_values_{i}'] = self.vis_processors[i](data['image']).unsqueeze_(0)
            output[f'input_ids_{i}'] = text_inputs['input_ids']
            output[f'attention_mask_{i}'] = text_inputs['attention_mask']
        output[f'img_id'] = data['index']
        return output


class UniqueClassSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.classes= {}
        self.remain_sample = 0

        progress = tqdm(range(len(data_source.dataset['img_id'])))

        for index, img_id in enumerate(data_source.dataset['img_id']):
            self.classes[img_id] = [data_source.cap_per_img * index + i for i in range(data_source.cap_per_img)]
            self.remain_sample += 5
            progress.update(1)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        while self.remain_sample >= self.batch_size:
            if len(self.classes.keys()) >= self.batch_size:
              batch_classes = np.random.choice(list(self.classes.keys()), self.batch_size, replace=False)
            else:
              batch_classes = list(self.classes.keys())
            for img_id in batch_classes:
                indices = self.classes[img_id] 
                selected_index = np.random.choice(indices)
                yield selected_index
                self.classes[img_id] = np.setdiff1d(self.classes[img_id], selected_index)
                self.remain_sample -=1
                if len(self.classes[img_id]) == 0:
                    self.classes.pop(img_id)


def collate_func(batch, processor):
    # print(batch)
    df = pd.DataFrame(batch)
    data = processor(
        text=list(df['caption']), 
        padding=True, 
        return_tensors='pt',
        truncation=True
    )
    data['pixel_values'] = torch.from_numpy(np.concatenate(list(df['pixel_values']), 0))
    data['img_id'] = torch.tensor(list(df['img_id'])) 
    return data

def get_fused_dataloader(dataset,  vis_processors, tokenizers, txt_processors=None, mode='train', batch_size=1):
    def coco_collate_func(batch):
        df = pd.DataFrame(batch)
        images = list(df['image']) 
        texts = list(df['text_input'])
        data = {}
        for i, vis_processor in enumerate(vis_processors):
            text_inputs = []
            if isinstance(vis_processor, (BlipProcessor, CLIPProcessor)): 
                pixel_values = vis_processor(images=images, return_tensors='pt')['pixel_values']
            else:
                pixel_values = []
                for image in list(images):
                    pixel_values.append(vis_processor(image).unsqueeze_(0))
                pixel_values = torch.cat(pixel_values, dim=0)
            for text in texts: 
                if txt_processors[i] is not None:
                    text_inputs.append(txt_processors[i](text))
                else:
                    text_inputs.append(text)
            if isinstance(tokenizers[i], (BlipProcessor, CLIPProcessor)):
                text_inputs = tokenizers[i](text=text_inputs, max_length=35, truncation=True, padding=True ,return_tensors='pt') 
            else:
                text_inputs = tokenizers[i](text_inputs, max_length=35, truncation=True, padding=True ,return_tensors='pt') 
            # data[f'pixel_values_{i}'] = torch.cat(pixel_values, dim=0) 
            data[f'pixel_values_{i}'] = pixel_values 
            data[f'attention_mask_{i}'] = text_inputs['attention_mask'] 
            data[f'input_ids_{i}'] = text_inputs['input_ids'] 

        data['img_id'] = torch.tensor(list(df['image_id']))
        return data

    if mode == 'train':
        return DataLoader(
            dataset[mode], 
            batch_size=batch_size, 
            collate_fn=coco_collate_func,
            shuffle=True
        ) 
    else:
        cur_dataset = FuseEvalDataset(
            dataset[mode], 
            vis_processors=vis_processors,
            txt_processors=txt_processors,
            tokenizers=tokenizers
        )
        return cur_dataset

def coco_eval_collate_func(batch, vis_processor, tokenizer):
    df = pd.DataFrame(batch)
    data = {}
    pixel_values = []

    for i, image in enumerate(list(df['image'])):
        if isinstance(vis_processor, CLIPProcessor) or isinstance(vis_processor, BlipProcessor):
            pixel_values.append(vis_processor(images=image, return_tensors='pt')['pixel_values'])
        else:
            pixel_values.append(vis_processor(image).unsqueeze_(0))

    if isinstance(tokenizer, CLIPProcessor) or isinstance(tokenizer, BlipProcessor):
        text_inputs = tokenizer(text=list(chain(*df['text_input'])), max_length=35, truncation=True, padding=True ,return_tensors='pt') 
    else:
        text_inputs = tokenizer(list(chain(*df['text_input'])), max_length=35, truncation=True, padding=True ,return_tensors='pt') 

    data['pixel_values'] = torch.cat(pixel_values, dim=0)
    data['img_id'] = torch.tensor(list(df['image_id']))
    data['input_ids'] = text_inputs['input_ids']
    data['attention_mask'] = text_inputs['attention_mask']

    return data

def get_dataloader(dataset,  vis_processor, tokenizer, txt_processor=None, mode='train', batch_size=1):
    def coco_collate_func(batch):
        df = pd.DataFrame(batch)
        data = {}
        pixel_values = []
        texts = []

        for i, image in enumerate(list(df['image'])):
            if isinstance(vis_processor, CLIPProcessor) or isinstance(vis_processor, BlipProcessor):
                pixel_values.append(vis_processor(images=image, return_tensors='pt')['pixel_values'])
            else:
                pixel_values.append(vis_processor(image).unsqueeze_(0))
            if txt_processor is not None:
                texts.append(txt_processor(list(df['text_input'])[i]))
            else:
                texts.append(list(df['text_input'])[i])
        if isinstance(tokenizer, CLIPProcessor) or isinstance(tokenizer, BlipProcessor):
            text_inputs = tokenizer(text=texts, max_length=35, truncation=True, padding=True ,return_tensors='pt') 
        else:
            text_inputs = tokenizer(texts, max_length=35, truncation=True, padding=True ,return_tensors='pt') 

        data['pixel_values'] = torch.cat(pixel_values, dim=0)
        data['img_id'] = torch.tensor(list(df['image_id']))
        data['input_ids'] = text_inputs['input_ids']
        data['attention_mask'] = text_inputs['attention_mask']

        return data

    if mode == 'train':
        return DataLoader(
            dataset[mode], 
            batch_size=batch_size, 
            collate_fn=coco_collate_func,
            shuffle=True
        ) 
    else:
        cur_dataset = EvalDataset(
            dataset[mode], 
            vis_processor=vis_processor,
            txt_processor=txt_processor,
            tokenizer=tokenizer
        )
        return  DataLoader(
            cur_dataset,
            batch_size=batch_size, 
            collate_fn=lambda batch: coco_eval_collate_func(batch, vis_processor=vis_processor, tokenizer=tokenizer),
            shuffle=False
        ), cur_dataset.img2txt, cur_dataset.txt2img 

def get_loaders(batch_size, dataset, vis_processor, tokenizer, txt_processor=None, eval_batch_size = 20):
    train_loader  = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        vis_processor=vis_processor,
        txt_processor=txt_processor,
        tokenizer=tokenizer,
        mode='train',
    ) 
    if eval_batch_size == 1:
        test_loader = EvalDataset(dataset['test'], vis_processor=vis_processor,txt_processor=txt_processor, tokenizer=tokenizer, use_tokenizer=True)
        val_loader = EvalDataset(dataset['val'], vis_processor=vis_processor,txt_processor=txt_processor, tokenizer=tokenizer, use_tokenizer=True)
        return  train_loader, val_loader, test_loader, test_loader.img2txt, test_loader.txt2img, val_loader.img2txt, val_loader.txt2img
    else:
        test_loader, test_img2txt, test_txt2img  = get_dataloader(
            dataset=dataset,
            batch_size=eval_batch_size,
            vis_processor=vis_processor,
            txt_processor=txt_processor,
            tokenizer=tokenizer,
            mode='test',
        ) 
        val_loader, val_img2txt, val_txt2img  = get_dataloader(
            dataset=dataset,
            batch_size=eval_batch_size,
            vis_processor=vis_processor,
            txt_processor=txt_processor,
            tokenizer=tokenizer,
            mode='val',
        ) 
    return train_loader, val_loader, test_loader, test_img2txt, test_txt2img, val_img2txt, val_txt2img


def get_fused_loaders(dataset, vis_processors, tokenizers, txt_processors=None):
    train_loader = get_fused_dataloader(dataset, vis_processors=vis_processors, tokenizers=tokenizers, txt_processors=txt_processors, batch_size=40, mode='train') 
    test_loader= get_fused_dataloader(dataset, vis_processors=vis_processors, tokenizers=tokenizers, txt_processors=txt_processors, batch_size=1, mode='test') 
    val_loader= get_fused_dataloader(dataset, vis_processors=vis_processors, tokenizers=tokenizers, txt_processors=txt_processors, batch_size=1, mode='val') 
 
    return train_loader, val_loader, test_loader