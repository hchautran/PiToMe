import json
from itertools import cycle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
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
from tc.lra_datasets import (ListOpsDataset, Cifar10Dataset, ImdbDataset)
from argparse import ArgumentParser
from accelerate import Accelerator
from dotenv import load_dotenv
from algo import (
    pitome, 
    tome,
    tofu,
    PITOME,
    TOME,
    TOFU,
    NONE
)
import os
import wandb
from consts import (
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
    'listops': ConfigDict(dict(dataset_fn=ListOpsDataset, config_getter=get_listops_config)),
    'cifar10': ConfigDict(dict(dataset_fn=Cifar10Dataset, config_getter=get_cifar10_config)),
    'imdb': ConfigDict(dict(dataset_fn=ImdbDataset, config_getter=get_text_classification_config)),
}
BERT_BASE = 'bert-base-uncased'
DISTILBERT_BASE = 'distilbert-base-uncased'
model_ft_dict = {
    BERT_BASE: 'JiaqiLee/imdb-finetuned-bert-base-uncased',
    DISTILBERT_BASE: 'lvwerra/distilbert-imdb',
}
model_dict  = {
    BERT_BASE: BERT_BASE,
    DISTILBERT_BASE: DISTILBERT_BASE,
}

class Engine:

    def __init__(self, task_name, model_ckt, ratio=1.0, algo=NONE, batch_size=32, enable_log=False):

        self.accelerator = Accelerator(
            mixed_precision='fp16',
            gradient_accumulation_steps=1
        )

        task = TASKS[task_name]
        self.batch_size = batch_size
        self.ratio = ratio
        self.config, self.model_config = task.config_getter()    
        train_dataset = task.dataset_fn(self.config, split='train')
        eval_dataset = task.dataset_fn(self.config, split='eval')    
        self.max_train_steps = int(np.ceil(self.config.total_train_samples / self.batch_size))
        self.enable_log = enable_log
        self.algo = algo
        self.ori_model = None
        self.model_ckt = model_ckt
        self.prepare_model(self.model_ckt, self.algo)

        self.config.tokenizer = self.tokenizer

        self.train_loader = self.accelerator.prepare(DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            collate_fn=lambda batch: transformers_collator(batch, self.tokenizer),
            shuffle=True
        ))
        self.eval_loader = self.accelerator.prepare(DataLoader(
            eval_dataset, 
            batch_size=self.batch_size, 
            collate_fn=lambda batch: transformers_collator(batch, self.tokenizer),
            shuffle=False
        ))

    def prepare_model(self, model_ckt, algo=None):
        self.algo = algo
        if model_ckt == BERT_BASE:
            self._prepare_bert_model(model_ft_dict[model_ckt],algo=algo)
        else:
            self._prepare_distil_model(model_ft_dict[model_ckt],algo=algo)

    def log(self, stats):
        if self.enable_log:
            wandb.log(stats)

    def _prepare_bert_model(self, model_ckt, algo=None):
        if self.ori_model is None:
            self.ori_model = BertForSequenceClassification.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')

        self.model = deepcopy(self.ori_model) 
        if algo is not None:
            self.algo = algo
        if self.algo == PITOME:
            pitome.patch.bert(self.model.bert.encoder)
        elif self.algo == TOME:
            tome.patch.bert(self.model.bert.encoder)
        elif self.algo == TOFU:
            tome.patch.bert(self.model.bert.encoder)

        self.model.bert.encoder.ratio = self.ratio 
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
        self.model = self.accelerator.prepare(self.model)
    


    def _prepare_distil_model(self, model_ckt, algo=None):
        if self.ori_model is None:
            self.ori_model = DistilBertForSequenceClassification.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')

        self.model = deepcopy(self.ori_model) 
        if self.algo is not None:
            self.algo = algo
        if self.algo == PITOME:
            pitome.patch.distilbert(self.model.distilbert.transformer)
        elif self.algo == TOME:
            tome.patch.distilbert(self.model.distilbert.transformer)
        elif self.algo == TOFU:
            tome.patch.distilbert(self.model.distilbert.transformer)

        self.model.distilbert.transformer.ratio = self.ratio 
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
        self.model = self.accelerator.prepare(self.model)
    

    def set_ratio(self, ratio):
        self.ratio = ratio
        if self.model_ckt == BERT_BASE:
            self.model.bert.encoder.ratio = self.ratio 
        else:
            self.model.distilbert.transformer.ratio = self.ratio 

    def init_logger(self):
        if self.enable_log:
            wandb.init(
                name=f'{self.algo}_{self.model_ckt}',
                project='tc_off_the_shell',
                config={
                    'algo': self.algo, 
                    'model': self.model_ckt, 
                    },
                reinit=True
            )


    def train(self, num_epochs:int):

        lr = self.config.learning_rate
        wd = self.config.weight_decay
        scheduler_fn = self.config.lr_scheduler
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        scheduler = scheduler_fn(optimizer)
        optimizer, scheduler= self.accelerator.prepare(optimizer, scheduler)
        for i in range(num_epochs):
            self.train_one_epoch(optimizer, scheduler)
            eval_stats = self.evaluate()
            eval_stats['epoch'] = i + 1 
            self.log(eval_stats)
            



    def train_one_epoch(self, optimizer, scheduler):
        self.model.train()
        pbar = tqdm(cycle(self.train_loader), total=self.max_train_steps)
        for i, (inputs, target) in enumerate(pbar):
            self.accelerator.free_memory()
            optimizer.zero_grad()
            outputs = self.model(**inputs, return_dict=False)
            loss = F.cross_entropy(outputs[0], target)
            self.accelerator.backward(loss)
            optimizer.step()
            cur_acc = accuracy_score(outputs[0], target)
            pbar.set_postfix_str(f"loss: {loss.item():.4f} accuracy: {cur_acc:.2f} gflops: {outputs[3]/1e9:.2f}")
            self.log({'logits/loss': loss.item(), 'logits/acc':cur_acc, 'gflops': outputs[3]/1e9})



    def evaluate(self):
        self.model.eval()
        eval_running_loss = 0.
        eval_running_acc = 0.
        eval_pbar = tqdm(self.eval_loader, total=len(self.eval_loader))
        for j, (inputs, target) in enumerate(eval_pbar):
            self.accelerator.free_memory()
            outputs = self.model(**inputs, return_dict=False)
            loss = F.cross_entropy(outputs[0], target)
            eval_running_loss += loss.item()
            eval_running_acc += accuracy_score(outputs[0], target)
            eval_pbar.set_postfix_str(
                f"eval loss: {100*eval_running_loss/(j+1):.2f} "
                f"eval accuracy: {100*eval_running_acc/(j+1):.2f} "
                f"gflops: {outputs[3]/1e9:.2f}"
            )
        if isinstance(self.model, BertForSequenceClassification):
            return {'eval acc': 100*eval_running_acc/len(self.eval_loader), 'ratio':self.model.bert.encoder.ratio, 'gflops': outputs[3]/1e9}
        else:
            return {'eval acc': 100*eval_running_acc/len(self.eval_loader), 'ratio':self.model.distilbert.transformer.ratio, 'gflops': outputs[3]/1e9}
