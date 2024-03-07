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

class Engine:

    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1
    )

    def __init__(self, task_name, model_ckt, ratio=1.0, algo=NONE, batch_size=32, enable_log=False):

        task = TASKS[task_name]
        self.batch_size = batch_size
        self.config, self.model_config = task.config_getter()    
        train_dataset = task.dataset_fn(self.config, split='train')
        eval_dataset = task.dataset_fn(self.config, split='eval')    
        self.max_train_steps = int(np.ceil(self.config.total_train_samples / self.batch_size))
        self.enable_log = enable_log

        if self.enable_log:
            wandb.init(
                name=f'{method}_{model_ckt}',
                project='tc_off_the_shell',
                config={
                    'algo': method, 
                    'model': model_ckt, 
                    },
                reinit=True
            )


        if model_ckt == BERT_BASE:
            self.model, self.tokenizer = self._prepare_bert_model(
                model_ft_dict[model_ckt], 
                compress_method=method,
                ratio=ratio
            )
        else:
            self.model, self.tokenizer = self._prepare_distil_model(
                model_ft_dict[model_ckt], 
                compress_method=method,
                ratio=ratio
            )
        config.tokenizer = self.tokenizer

        self.config = config
        self.model_ckt = model_ckt
        self.train_loader = accelerator.prepare(DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            collate_fn=lambda batch: transformers_collator(batch, self.tokenizer),
            shuffle=True
        ))
        self.eval_loader = accelerator.prepare(DataLoader(
            eval_dataset, 
            batch_size=self.batch_size, 
            collate_fn=lambda batch: transformers_collator(batch, self.tokenizer),
            shuffle=False
        ))
        self.model = accelerator.prepare(self.model)


    def log(self, stats):
        if self.enable_log:
            wandb.log(stats)

    def _prepare_bert_model(self):
        model = BertForSequenceClassification.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
        if compress_method == PITOME:
            pitome.patch.bert(model.bert.encoder)
        elif compress_method == TOME:
            tome.patch.bert(model.bert.encoder)
        elif compress_method == TOFU:
            tome.patch.bert(model.bert.encoder)
        model.bert.encoder.ratio = self.ratio 
        tokenizer = AutoTokenizer.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
        model = accelerator.prepare(model)
        return model, tokenizer


    def _prepare_distil_model(self):
        model = DistilBertForSequenceClassification.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
        if compress_method == PITOME:
            pitome.patch.distilbert(model.distilbert.transformer)
        elif compress_method == TOME:
            tome.patch.distilbert(model.distilbert.transformer)
        elif compress_method == TOFU:
            tome.patch.bert(model.bert.encoder)
        model.distilbert.transformer.ratio = ratio 
        tokenizer = AutoTokenizer.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
        model = accelerator.prepare(model)
        return model, tokenizer



    def train(self, num_epochs:int):
        lr = self.config.learning_rate
        wd = self.config.weight_decay
        scheduler_fn = self.config.lr_scheduler

        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        scheduler = scheduler_fn(optimizer)

        optimizer, scheduler= accelerator.prepare(optimizer, scheduler)
        avg_loss = None
        avg_acc = None
        for i in range(num_epochs):
            avg_loss, avg_acc = self.train_one_epoch(optimizer, scheduler)
            eval_stats = self.evaluate()
            eval_stats['train loss'] = avg_loss
            eval_stats['train acc'] = avg_acc
            eval_stats['epoch'] = i + 1 
            self.log(eval_stats)
            



    def train_one_epoch(self, optimizer, scheduler):
        self.model.train()
        pbar = tqdm(cycle(self.train_loader), total=self.max_train_steps)
        for i, (inputs, target) in enumerate(pbar):
            accelerator.free_memory()
            optimizer.zero_grad()
            outputs = self.model(**inputs, return_dict=False)
            loss = F.cross_entropy(outputs[0], target)
            accelerator.backward(loss)
            optimizer.step()
            cur_loss = loss.item()
            cur_acc = accuracy_score(outputs[0], target)
            avg_loss = cur_loss if avg_loss is None else avg_factor * avg_loss + (1-avg_factor) * cur_loss  
            avg_acc = cur_acc if avg_acc is None else avg_factor * avg_acc + (1-avg_factor) * cur_acc
            pbar.set_postfix_str(f"loss: {avg_loss:.2f} accuracy: {avg_acc:.2f} gflops: {outputs[3]/1e9:.2f}")
            self.log({'logits/loss': avg_loss, 'logits/acc':avg_acc, 'gflops': outputs[3]/1e9})

        return avg_loss, avg_acc



    def evaluate(self):
        self.model.eval()
        eval_running_loss = 0.
        eval_running_acc = 0.
        eval_pbar = tqdm(eval_dataloader, total=len(self.eval_dataloader))
        for j, (inputs, target) in enumerate(eval_pbar):
            accelerator.free_memory()
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
            return {'eval acc': 100*eval_running_acc/len(eval_dataloader), 'ratio':self.model.bert.encoder.ratio, 'gflops': outputs[3]/1e9}
        else:
            return {'eval acc': 100*eval_running_acc/len(eval_dataloader), 'ratio':self.model.distilbert.transformer.ratio, 'gflops': outputs[3]/1e9}
