import numpy as np
import pandas as pd
import pickle
from functools import reduce
import torch
from glob import glob
from itertools import cycle
from datasets import load_dataset
import os
import subprocess
from .config import DATA_PATH


class ImdbDataset:

    def __init__(self, config, split='train'):       

        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH, exist_ok=True)
            os.chdir(DATA_PATH)
            print('Saving data to', DATA_PATH)
            imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            imdb_tar = "aclImdb_v1.tar.gz"
            subprocess.run(["wget", imdb_url])
            subprocess.run(["tar", "-xvf", imdb_tar])
            os.remove(imdb_tar)

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





