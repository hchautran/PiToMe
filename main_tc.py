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
    DIFFRATE,
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
from tc.engine import Engine, BERT_BASE, DISTILBERT_BASE, BERT_LARGE


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

TASKS = [
    'sst2',
    'imdb',
    'rotten',
    'bbc',
]

def eval(model, eval_dataset, tokenizer,batch_size=4):
    model.eval()
    eval_running_loss = 0.
    eval_running_acc = 0.
    gflops = 0.
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
        gflops += outputs[3]/1e9 
        eval_pbar.set_postfix_str(
            f"eval loss: {100*eval_running_loss/(j+1):.2f} "
            f"eval accuracy: {100*eval_running_acc/(j+1):.2f} "
            f"gflops: {100 * gflops/(j+1):.2f}"
        )
    if isinstance(model, BertForSequenceClassification):
        return {'acc': 100*eval_running_acc/len(eval_dataloader), 'ratio':model.bert.encoder.ratio, 'gflops': outputs[3]/1e9}
    else:
        return {'acc': 100*eval_running_acc/len(eval_dataloader), 'ratio':model.distilbert.transformer.ratio, 'gflops': outputs[3]/1e9}


if __name__ == "__main__":
    import pathlib
    parser = ArgumentParser()
    parser.add_argument("--task", default="imdb", choices=TASKS,
                        help="choose an LRA dataset from available options")
    parser.add_argument("--algo", default=PITOME, choices=[PITOME, TOME, NONE, TOFU, DCT, DIFFRATE],
                        help="choose an LRA dataset from available options")
    parser.add_argument("--model", default=BERT_BASE, choices=[BERT_BASE, DISTILBERT_BASE, BERT_LARGE],
                        help="choose an LRA dataset from available options")
    parser.add_argument("--ratio", default=0.55, help="remain ratio")
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--batch_size', default=8, help='Perform evaluation only')
    args = parser.parse_args()
    batch_size = 4 
    avg_factor = 0.95
    task_name = args.task
    algo = args.algo




    file_name = f'train_tc_{args.model}_{task_name}.csv' if not args.eval else f'eval_tc_{args.model}_{task_name}.csv'
    print(file_name)
    engine = Engine(
        task_name=task_name,
        model_ckt=args.model,
        ratio=float(args.ratio),
        algo=args.algo,
        enable_log=not args.eval,
        trained=args.eval
    )
    engine.init_logger()
    if args.eval:
        metrics = engine.evaluate()
    else:
        metrics = engine.train(num_epochs=10)
            
    abs_path ='/home/caduser/HDD/vit_token_compress/PiToMe'
    path = f'{abs_path}/{file_name}'
    if not pathlib.Path(path).is_file():
        head = "dataset, model, algo, gflops, ratio, acc\n"
        with open(file_name, "a") as myfile:
            myfile.write(head)

    if metrics is not None:
        row = f'{args.task}, {args.model}, {args.algo}, {metrics["gflops"]}, {metrics["ratio"]}, {metrics["acc"]}\n'
        with open(file_name, "a") as myfile:
            myfile.write(row)
                    
                