import json
import time
from itertools import cycle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AlbertForSequenceClassification,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
)
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from ml_collections import ConfigDict
from tc.lra_config import (
    get_listops_config, 
    get_cifar10_config, 
    get_text_classification_config
)
from tc.lra_datasets import (SST2Dataset, ImdbDataset, RottenTomatoes)
from argparse import ArgumentParser
from accelerate import Accelerator
from algo import (
    pitome, 
    tome,
    tofu,
    dct,
    mctf,
    crossget,
    PITOME,
    TOME,
    TOFU,
    DCT,
    NONE,
    DIFFRATE,
    CROSSGET,
    MCTF,
)
import os
import wandb
from .get_data import (
    DATA_PATH
)
from copy import deepcopy

def transformers_collator(batch, tokenizer):
    input_list, target_list = zip(*batch)
    inputs = tokenizer(input_list, truncation=True,max_length=512, padding=True, return_tensors='pt')
    return inputs, torch.cat(target_list)

def accuracy_score(outp, target):
    assert len(outp.shape) == 2, "accuracy score must receive 2d output tensor"
    assert len(target.shape) == 1, "accuracy score must receive 1d target tensor"
    return (torch.argmax(outp, dim=-1) == target).sum().item() / len(target)


TASKS = {
    'imdb': ConfigDict(dict(dataset_fn=ImdbDataset, config_getter=get_text_classification_config)),
    'rotten': ConfigDict(dict(dataset_fn=RottenTomatoes, config_getter=get_text_classification_config)),
    'sst2': ConfigDict(dict(dataset_fn=SST2Dataset, config_getter=get_text_classification_config)),
}
batch_sizes = {
    'imdb': 24, 
    'rotten': 256,
    'sst2': 256,
}
BERT_BASE = 'bert-base-uncased'
DISTILBERT_BASE = 'distilbert-base-uncased'
BERT_LARGE= 'bert-large-uncased'
ALBERT= 'albert'

model_imdb_dict = {
    BERT_BASE: 'JiaqiLee/imdb-finetuned-bert-base-uncased',
    DISTILBERT_BASE: 'lvwerra/distilbert-imdb',
    BERT_LARGE:BERT_LARGE,
    ALBERT:'JeffreyJIANG/albert-imdb'
}
model_rotten_dict = {
    BERT_BASE: 'zebans/bert-base-cased-finetuned-rotten-tomatoes-epochs-2',
    DISTILBERT_BASE: 'pig4431/rtm_DistilBERT_5E',
    BERT_LARGE:BERT_LARGE
}
model_sst2_dict = {
    BERT_BASE: 'gchhablani/bert-base-cased-finetuned-sst2',
    DISTILBERT_BASE: 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
    BERT_LARGE:'assemblyai/bert-large-uncased-sst2',
    ALBERT:'Alireza1044/albert-base-v2-sst2'
}
model_bbc_dict = {
    BERT_LARGE:'AyoubChLin/BERT-Large_BBC_news',
    BERT_BASE: 'AyoubChLin/ESG-bert-BBC_news',
}

model_dict  = {
    BERT_BASE: BERT_BASE,
    DISTILBERT_BASE: DISTILBERT_BASE,
    BERT_LARGE: BERT_LARGE,
}

BERT_PATCHES = {
    PITOME: pitome.patch.bert, 
    TOME: tome.patch.bert, 
    TOFU: tofu.patch.bert, 
    DCT: dct.patch.bert, 
    MCTF: mctf.patch.bert, 
    CROSSGET: crossget.patch.bert, 
}
DISTILBERT_PATCHES = {
    PITOME: pitome.patch.distilbert, 
    TOME: tome.patch.distilbert, 
    TOFU: tofu.patch.distilbert, 
    DCT: dct.patch.distilbert, 
    MCTF: mctf.patch.distilbert, 
    CROSSGET: pitome.patch.distilbert, 
}
class Engine:

    def __init__(self, task_name, model_ckt, ratio=1.0, algo=NONE, batch_size=None, enable_log=False, trained=False, margin=None, alpha=1.0):

        self.accelerator = Accelerator(
            mixed_precision='fp16',
            gradient_accumulation_steps=1
        )

        task = TASKS[task_name]
        if task_name == 'imdb' and algo==NONE: self.batch_size = 12
        else: self.batch_size = batch_sizes[task_name] if batch_size is None else batch_size
        self.ratio = ratio
        self.margin = margin
        self.alpha = alpha
        self.config, self.model_config = task.config_getter()    
        self.train_dataset = task.dataset_fn(self.config, split='train')
        self.eval_dataset = task.dataset_fn(self.config, split='eval')    
        self.max_train_steps = int(np.ceil(self.config.total_train_samples / self.batch_size))
        self.enable_log = enable_log
        self.algo = algo
        self.ori_model = None
        self.model_ckt = model_ckt
        self.task_name = task_name
        if trained:
            if task_name == 'imdb':
                self.model_dict = model_imdb_dict 
            elif task_name == 'rotten':
                self.model_dict = model_rotten_dict
            elif task_name == 'bbc':
                self.model_dict = model_bbc_dict
            else :
                self.model_dict = model_sst2_dict 
        else:
            self.model_dict = model_dict 
        self.prepare_model(self.model_ckt, self.algo)

        self.config.tokenizer = self.tokenizer

        self.train_loader = self.accelerator.prepare(DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            collate_fn=lambda batch: transformers_collator(batch, self.tokenizer),
            shuffle=True
        ))
        self.eval_loader = self.accelerator.prepare(DataLoader(
            self.eval_dataset, 
            batch_size=self.batch_size, 
            collate_fn=lambda batch: transformers_collator(batch, self.tokenizer),
            shuffle=False
        ))

    def prepare_model(self, model_ckt, algo=None):
        self.algo = algo
        
        print(self.model_dict[model_ckt])
        if model_ckt == BERT_BASE or model_ckt == BERT_LARGE:
            self._prepare_bert_model(self.model_dict[model_ckt],algo=algo)
        elif model_ckt == DISTILBERT_BASE:
            self._prepare_distil_model(self.model_dict[model_ckt],algo=algo)
        else:
            self.model = AlbertForSequenceClassification.from_pretrained(self.model_dict[model_ckt], cache_dir=f'{DATA_PATH}/.cache')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dict[model_ckt], cache_dir=f'{DATA_PATH}/.cache')
            self.model = self.accelerator.prepare(self.model)

    def log(self, stats):
        if self.enable_log:
            wandb.log(stats)

    def _prepare_bert_model(self, model_ckt, algo=None):
        self.model= BertForSequenceClassification.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')

        self.algo = algo


        if self.algo is not None:
            BERT_PATCHES[self.algo](self.model.bert.encoder)
        else:
            pitome.patch.bert(self.model.bert.encoder)
            self.set_ratio(1.0)

        self.model.bert.encoder.ratio = self.ratio 
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
        self.model = self.accelerator.prepare(self.model)
    


    def _prepare_distil_model(self, model_ckt, algo=None):
        self.model = DistilBertForSequenceClassification.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
        self.algo = algo

        if self.algo == PITOME:
            DISTILBERT_PATCHES[self.algo](self.model.distilbert.transformer)
        else:
            tome.patch.distilbert(self.model.distilbert.transformer)
            self.set_ratio(1.0)

        self.model.distilbert.transformer.ratio = self.ratio 
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
        self.model = self.accelerator.prepare(self.model)
    

    def set_ratio(self, ratio):
        self.ratio = ratio
        if self.model_ckt == BERT_BASE or self.model_ckt == BERT_LARGE:
            self.model.bert.encoder.ratio = self.ratio 
        else:
            self.model.distilbert.transformer.ratio = self.ratio 

        self.model = self.accelerator.prepare(self.model)

    def init_logger(self):
        if self.enable_log:
            wandb.init(
                name=f'{self.algo}_{self.model_ckt}',
                project=f'tc_train_{self.task_name}',
                config={
                    'algo': self.algo, 
                    'model': self.model_ckt, 
                    },
                reinit=True
            )


    def train(self, num_epochs:int):

        lr = self.config.learning_rate
        wd = self.config.weight_decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, min_lr=1e-8, mode='max')
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  eta_min=1e-8, T_max=3)
        optimizer, scheduler= self.accelerator.prepare(optimizer, scheduler)
        best_acc = 0
        start = time.time()
        for i in range(num_epochs):
            self.train_one_epoch(optimizer, scheduler)
            eval_stats = self.evaluate()
            scheduler.step(eval_stats['acc'])
            # scheduler.step(i)
                
            eval_stats['epoch'] = i + 1 
            print(eval_stats)
            if best_acc < eval_stats['acc']:
                best_acc = eval_stats['acc']
                print('best acc:', best_acc)
            self.log(eval_stats)
        train_time = time.time() - start
        eval_stats['acc'] = best_acc
        eval_stats['train time'] = train_time 
        return eval_stats
            



    def train_one_epoch(self, optimizer, scheduler):
        self.model.train()
        pbar = tqdm(self.train_loader, total=len(self.train_loader))
        for i, (inputs, target) in enumerate(pbar):
            outputs = self.model(**inputs, return_dict=False)
            loss = F.cross_entropy(outputs[0], target)
            self.accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            cur_acc = accuracy_score(outputs[0], target)
            pbar.set_postfix_str(f"loss: {loss.item():.4f} accuracy: {cur_acc*100:.2f} gflops: {outputs[3]/1e9:.2f}")
            self.log({'logits/loss': loss.item(), 'logits/acc':cur_acc, 'gflops': outputs[3]/1e9})
        self.accelerator.clear()



    def evaluate(self):
        self.model.eval()
        start = time.time()
        eval_running_loss = 0.
        eval_running_acc = 0.
        gflops = 0.
        eval_pbar = tqdm(self.eval_loader, total=len(self.eval_loader))
        for j, (inputs, target) in enumerate(eval_pbar):
            outputs = self.model(**inputs, return_dict=False)
            loss = F.cross_entropy(outputs[0], target)
            eval_running_loss += loss.item()
            eval_running_acc += accuracy_score(outputs[0], target)
            gflops += outputs[3]/1e9 
            eval_pbar.set_postfix_str(
                f"eval loss: {100*eval_running_loss/(j+1):.2f} "
                f"eval accuracy: {100*eval_running_acc/(j+1):.2f} "
                # f"gflops: {gflops/(j+1):.2f}"
            )
        eval_time = time.time() - start
        if isinstance(self.model, BertForSequenceClassification):

            return {'acc': 100*eval_running_acc/len(self.eval_loader), 'ratio':self.model.bert.encoder.ratio, 'gflops': gflops/len(self.eval_loader), 'eval time': eval_time, 'train time': 0}
        else:
            return {'acc': 100*eval_running_acc/len(self.eval_loader), 'ratio':self.model.distilbert.transformer.ratio, 'gflops': gflops/len(self.eval_loader), 'eval time':eval_time, 'train time': 0}
