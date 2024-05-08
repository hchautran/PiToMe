import numpy as np
import pandas as pd
import pickle
from functools import reduce
import torch
from glob import glob
from itertools import cycle
from datasets import load_dataset
import os

# Access the environment variable
# DATA_PATH =  '/mnt/data/mount_4TBSSD/nmduy/pitome'
DATA_PATH = '/media/caduser/MyBook/chau'


class ImdbDataset:
    def __init__(self, config, split='train'):       
        paths = {'train': f"{DATA_PATH}/aclImdb/train", 'eval': f"{DATA_PATH}/aclImdb/test"}
        split_path = paths[split]
        neg_path = split_path + "/neg"
        pos_path = split_path + "/pos"
        neg_inputs = zip(glob(neg_path+"/*.txt"), cycle([0]))
        pos_inputs = zip(glob(pos_path+"/*.txt"), cycle([1]))
        self.data = np.random.permutation(list(neg_inputs) + list(pos_inputs))
        
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    def __getitem__(self, i):
        data = self.data[i]
        with open(data[0], 'r') as fo:
            source = fo.read()
        target = int(data[1])
        return source, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)

        
class RottenTomatoes: 
    def __init__(self, config ,split='train'):
        cache_dir = f'{DATA_PATH}/.cache' 
        self.split = 'train' if split == 'train' else 'test'
        self.data = load_dataset('rotten_tomatoes', cache_dir=cache_dir)[self.split]
        
    def __getitem__(self, i):
        sample = self.data[i]
        return sample['text'], torch.LongTensor([sample['label']])
    
    def __len__(self):
        return len(self.data)

        
class SST2Dataset: 
    def __init__(self, config ,split='train'):
        cache_dir = f'{DATA_PATH}/.cache' 
        self.split = 'train' if split == 'train' else 'validation'
        self.data = load_dataset('stanfordnlp/sst2', cache_dir=cache_dir)[self.split]
        
    def __getitem__(self, i):
        sample = self.data[i]
        return sample['sentence'], torch.LongTensor([sample['label']])
    
    def __len__(self):
        return len(self.data)


class BBCDataset: 
    def __init__(self, config ,split='train'):
        cache_dir = f'{DATA_PATH}/.cache' 
        self.split = 'train' if split == 'train' else 'test'
        self.data = load_dataset('SetFit/bbc-news', cache_dir=cache_dir)[self.split]
        
    def __getitem__(self, i):
        sample = self.data[i]
        return sample['text'], torch.LongTensor([sample['label']])
    
    def __len__(self):
        return len(self.data)

class ListOpsDataset:
    def __init__(self, config, split='train'):
        paths = {'train': f"{DATA_PATH}/lra_release/listops-1000/basic_train.tsv",
                      'eval': f"{DATA_PATH}/lra_release/listops-1000/basic_val.tsv"}
        self.data = pd.read_csv(paths[split], delimiter='\t')
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    def __getitem__(self, i):
        data = self.data.iloc[i]
        source = data.Source
        inputs = self.tokenizer(source, max_length=self.max_length) #return_tensors='pt', truncation=True, padding='max_length'
        target = data.Target
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)


class Cifar10Dataset:
    def __init__(self, config, split='train'):
        paths = {'train': [f"{DATA_PATH}/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)],
                      'eval': [f"{DATA_PATH}/cifar-10-batches-py/test_batch"]
                     }
        print("loading cifar-10 data...")
        data_dicts = [Cifar10Dataset.unpickle(path) for path in paths[split]]
        print("assembling cifar-10 files..")
        self.data = reduce((lambda x, y: {b'data': np.concatenate([x[b'data'], y[b'data']], axis=0), 
                                         b'labels': np.concatenate([x[b'labels'], y[b'labels']], axis=0)}), 
                           data_dicts)
        # TODO CHECK: i think this is the right shape 
        # see: https://www.cs.toronto.edu/~kriz/cifar.html 
        #      section "Dataset layouts" discusses the memory layout of the array
        self.data[b'data'] = self.data[b'data'].reshape((-1, 3, 1024)) 
       
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
    
    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d
    
    def __getitem__(self, i):
        r, g, b = self.data[b'data'][i]
        # grayscale image (assume pixels in [0, 255])
        source = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(int)
        inputs = self.tokenizer(source, max_length=self.max_length)
        target = self.data[b'labels'][i]
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data[b'data'])
