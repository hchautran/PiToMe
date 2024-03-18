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
    PITOME,
    TOME,
    TOFU,
    DCT,
    NONE
)
import os
import wandb
from consts import (
    DATA_PATH
)

accelerator = Accelerator(
    mixed_precision='fp16',
    gradient_accumulation_steps=4
)
from tc.engine import Engine


def transformers_collator(batch, tokenizer):
    input_list, target_list = zip(*batch)
    inputs = tokenizer(input_list, truncation=True,max_length=512, padding=True, return_tensors='pt')
    return inputs, torch.cat(target_list)


def accuracy_score(outp, target):
    assert len(outp.shape) == 2, "accuracy score must receive 2d output tensor"
    assert len(target.shape) == 1, "accuracy score must receive 1d target tensor"
    return (torch.argmax(outp, dim=-1) == target).sum().item() / len(target)


# consts
OUTPUT_DIR = "output_dir/"
deepspeed_json = "ds_config.json"

TASKS = {
    'listops': ConfigDict(dict(dataset_fn=ListOpsDataset, config_getter=get_listops_config)),
    'cifar10': ConfigDict(dict(dataset_fn=Cifar10Dataset, config_getter=get_cifar10_config)),
    'imdb': ConfigDict(dict(dataset_fn=ImdbDataset, config_getter=get_text_classification_config)),
}


def prepare_bert_model(model_ckt, compress_method='none', ratio=1.0):
    model = BertForSequenceClassification.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
    if compress_method == PITOME:
        pitome.patch.bert(model.bert.encoder)
    elif compress_method == TOME:
        tome.patch.bert(model.bert.encoder)

    model.bert.encoder.ratio = ratio 

    tokenizer = AutoTokenizer.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
    model = accelerator.prepare(model)

    return model, tokenizer

def prepare_distil_model(model_ckt, compress_method='none', ratio=1.0):
    model = DistilBertForSequenceClassification.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
    if compress_method == PITOME:
        pitome.patch.distilbert(model.distilbert.transformer)
    elif compress_method == TOME:
        tome.patch.distilbert(model.distilbert.transformer)

    model.distilbert.transformer.ratio = ratio 

    tokenizer = AutoTokenizer.from_pretrained(model_ckt, cache_dir=f'{DATA_PATH}/.cache')
    model = accelerator.prepare(model)

    return model, tokenizer




def eval(model, eval_dataset, tokenizer,batch_size=4):
    model.eval()
    eval_running_loss = 0.
    eval_running_acc = 0.
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        collate_fn=lambda batch: transformers_collator(batch, tokenizer),
        shuffle=False
    )
    
    eval_dataloader = accelerator.prepare(eval_dataloader)
    eval_pbar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for j, (inputs, target) in enumerate(eval_pbar):
        accelerator.free_memory()
        outputs = model(**inputs, return_dict=False)
        loss = F.cross_entropy(outputs[0], target)
        eval_running_loss += loss.item()
        eval_running_acc += accuracy_score(outputs[0], target)
        eval_pbar.set_postfix_str(
            f"eval loss: {100*eval_running_loss/(j+1):.2f} "
            f"eval accuracy: {100*eval_running_acc/(j+1):.2f} "
            f"gflops: {outputs[3]/1e9:.2f}"
        )
    if isinstance(model, BertForSequenceClassification):
        return {'acc': 100*eval_running_acc/len(eval_dataloader), 'ratio':model.bert.encoder.ratio, 'gflops': outputs[3]/1e9}
    else:
        return {'acc': 100*eval_running_acc/len(eval_dataloader), 'ratio':model.distilbert.transformer.ratio, 'gflops': outputs[3]/1e9}

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

if __name__ == "__main__":
    import pathlib
    parser = ArgumentParser()
    parser.add_argument("--task", default="imdb", choices=TASKS.keys(),
                        help="choose an LRA dataset from available options")
    parser.add_argument("--algo", default=PITOME, choices=[PITOME, TOME, NONE, TOFU, DCT],
                        help="choose an LRA dataset from available options")
    parser.add_argument("--ratio", default=0.55, help="remain ratio")
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    args = parser.parse_args()
    batch_size = 4 
    avg_factor = 0.95
    task_name = args.task
    algo = args.algo




    for model_ckt in [
        BERT_BASE,
        DISTILBERT_BASE, 
    ]:
        engine = Engine(
            task_name=task_name,
            model_ckt=model_ckt,
            ratio=float(args.ratio),
            algo=args.algo,
            batch_size=64,
            enable_log=False,
            trained=True
        )
        if args.eval:
            metrics = engine.evaluate()
        else:
            metrics = engine.train(num_epochs=2)
                
        abs_path ='/home/caduser/HDD/vit_token_compress/PiToMe'
        file_name = 'test_tc.csv'
        path = f'{abs_path}/{file_name}'
        if not pathlib.Path(path).is_file():
            head = "model, algo, gflops, ratio ,acc\n"
            with open(file_name, "a") as myfile:
                myfile.write(head)

        if metrics is not None:
            row = f'{BERT_BASE}, {args.algo}, {metrics["gflops"]}, {metrics["ratio"]}, {metrics["acc"]}\n'
            with open(file_name, "a") as myfile:
                myfile.write(row)
                        
                