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
    NONE
)
import os
import wandb

# Load environment variables from .env file
load_dotenv()

# Access the environment variable
DATA_PATH = os.environ.get('DATA_PATH')
# DATA_PATH = '/mnt/data/mount_4TBSSD/nmduy/pitome'
DATA_PATH = '/media/caduser/MyBook/chau'



accelerator = Accelerator(
    mixed_precision='fp16',
    gradient_accumulation_steps=4
)


device = torch.device(
    f"cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


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



def train(model, config, dataset ,use_deepspeed):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=transformers_collator)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = config.learning_rate
    wd = config.weight_decay
 
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler_fn = config.lr_scheduler
    scheduler = scheduler_fn(optimizer)
    
    if use_deepspeed:
            optimizer, dataloader, scheduler= accelerator.prepare(optimizer, dataloader, scheduler)
    
    # train model
    model.to(device)
    model.train()
    avg_loss = None
    avg_acc = None
    pbar = tqdm(cycle(dataloader), total=max_train_steps)
    for i, (inputs, target) in enumerate(pbar):
        accelerator.free_memory()
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, target)
        accelerator.backward(loss)
        optimizer.step()
        cur_loss = loss.item()
        cur_acc = accuracy_score(outputs.logits, target)
        avg_loss = cur_loss if avg_loss is None else avg_factor * avg_loss + (1-avg_factor) * cur_loss  
        avg_acc = cur_acc if avg_acc is None else avg_factor * avg_acc + (1-avg_factor) * cur_acc
        pbar.set_postfix_str(f"loss: {avg_loss:.2f} accuracy: {avg_acc:.2f}")


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
    return {'acc': 100*eval_running_acc/len(eval_dataloader), 'ratio':model.distilbert.transformer, 'gflops': outputs[3]/1e9}
    # return {'acc': 100*eval_running_acc/len(eval_dataloader), 'ratio':model.bert.encoder.ratio}


# main
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", default="imdb", choices=TASKS.keys(),
                        help="choose an LRA dataset from available options")
    parser.add_argument("--cm", default="dct", choices=TASKS.keys(),
                        help="choose an LRA dataset from available options")
    parser.add_argument("--deepspeed", action="store_true",
                        help="use deepspeed optimization for better performance")
    args = parser.parse_args()
    batch_size = 4 
    avg_factor = 0.95
    task_name = args.task
    # model_ckt = 'JiaqiLee/imdb-finetuned-bert-base-uncased'
    model_ckt = 'lvwerra/distilbert-imdb'

    # model_ckt = 'bert-base-uncased'
    # model_ckt = 'bert-large-uncased'

    # compress_methjod='none' 
    # compress_method='dct'
    for method in [
        PITOME,
        TOME, 
        # 'dct', 
        # NONE,
    ]:
        # wandb.init(
        #     name=f'{method}_bert-base',
        #     project='tc_off_the_shell',
        #     config={
        #        'algo': method, 
        #        'model': 'bert-base', 
        #     },
        #     reinit=True
        # )
        model, tokenizer = prepare_distil_model(
            model_ckt, 
            compress_method=method,
            ratio=.505
        )

        task = TASKS[task_name]
        config, model_config = task.config_getter()    
        config.tokenizer = tokenizer

        dataset = task.dataset_fn(config, split='train')
        eval_dataset = task.dataset_fn(config, split='eval')    
        max_train_steps = int(np.ceil(config.total_train_samples / batch_size))

        res = eval(model, eval_dataset, tokenizer ,batch_size=128)
        # wandb.log(stats)
