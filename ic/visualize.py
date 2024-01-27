import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import load_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler
import utils
import shutil
import warnings
from utils import MultiEpochsDataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler

from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import torch
from dotenv import load_dotenv
from utils import build_transform, DATA_PATH
from main_pitome import process_image
import tome
import pitome
import os
import DiffRate
from main_pitome import get_args_parser
import models_mae
import wandb

def get_tome_model(model, args):
    if 'deit' in model_ckt:
        tome.patch.deit(model,use_r=args.use_r)
        model.ratio=float(args.ratio)
        model.r=int(args.r)
    elif 'mae' in args.model:
        tome.patch.mae(model,use_r=args.use_r)
        model.ratio=float(args.ratio)
        model.r=int(args.r)
    else:
        raise ValueError("only support deit, mae and caformer in this codebase")
    

def get_pitome_model(model, args):
    if 'deit' in args.model:
        pitome.patch.deit(model,use_r=args.use_r)
        model.ratio=float(args.ratio)
        model.r=int(args.r)
    elif 'mae' in args.model:
        pitome.patch.mae(model,use_r=args.use_r)
        model.ratio=float(args.ratio)
        model.r=int(args.r)
    else:
        raise ValueError("only support deit, mae and caformer in this codebase")



def get_diffrate_model(model, args):
    if 'deit' in args.model:
        DiffRate.patch.deit(model, prune_granularity=args.granularity, merge_granularity=args.granularity)
    elif 'mae' in args.model:
        DiffRate.patch.mae(model, prune_granularity=args.granularity, merge_granularity=args.granularity)
    else:
        raise ValueError("only support deit, mae and caformer in this codebase")

    if args.use_r:
        model.init_kept_num_using_r(args.r)
    else:
        model.init_kept_num_using_ratio(args.ratio)
    
            

def main(args, model ,logger):
    # utils.setup_default_logging()

    
    model_name_dict = {
        'vit_deit_tiny_patch16_224':'ViT-T-DeiT',
        'vit_deit_small_patch16_224':'ViT-S-DeiT',
        'vit_deit_base_patch16_224': 'ViT-B-DeiT',
        'vit_base_patch16_mae': 'ViT-B-MAE',
        'vit_large_patch16_mae': 'ViT-L-MAE',
        'vit_huge_patch14_mae': 'ViT-H-MAE',
    }
    
            
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed
        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr


    test_stats = evaluate(data_loader_val, model, device,logger)
    logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    return test_stats


model_name_dict = {
    'deit_tiny_patch16_224':'ViT-T-DeiT',
    'deit_small_patch16_224':'ViT-S-DeiT',
    'deit_base_patch16_224': 'ViT-B-DeiT',
    'vit_base_patch16_mae': 'ViT-B-MAE',
    'vit_large_patch16_mae': 'ViT-L-MAE',
    'vit_huge_patch14_mae': 'ViT-H-MAE',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)

    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir,dist_rank=utils.get_rank())
    wandb = utils.Wandb()
    logger.info(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    dataset = load_dataset("imagenet-1k", cache_dir=f"{DATA_PATH}/imagenet/")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    def filter_out_grayscale(example):
        img_tensor = transform(example['image'])
        # Check if the image has only one channel (grayscale)
        if img_tensor.shape[0] == 3:
            return True
        return False


    dataset_val = dataset['validation']
    dataset_val = dataset_val.filter(filter_out_grayscale, num_proc=10)


    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
  
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # leveraging MultiEpochsDataLoader for faster data loading

    args.batch_size = 100

    data_loader_val = MultiEpochsDataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=lambda batch: process_image(batch, build_transform(is_train=False, args=args)),
        drop_last=False
    )
    
    for model_ckt in [
        # 'deit_tiny_patch16_224',
        # 'deit_small_patch16_224',
        # 'deit_base_patch16_224',
        # 'vit_base_patch16_mae',
        'vit_large_patch16_mae',
        'vit_huge_patch14_mae',
    ]:
        for algo in [
            'PiToMe',
            'DiffRate',
            'ToMe',
            'Baseline',
        ]:
            wandb.init(
                name=f'{algo}_{model_name_dict[model_ckt]}',
                project='ic_off_the_shell',
                config={
                   'algo': algo, 
                   'model': model_name_dict[model_ckt], 
                },
                reinit=True
            )
            args.model = model_ckt
            logger.info(f"Creating model: {args.model}")
            ratios = [0.975, 0.95, 0.925, 0.90, 0.875, 0.85] if algo != 'Baseline' else [1.0]

            for ratio in ratios:
            # for ratio in [0.975, 0.95, 0.925, 0.90, 0.875, 0.85]:
                model = create_model(
                    args.model,
                    pretrained=True,
                    num_classes=1000,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                )
                args.use_r = False 
                args.ratio = ratio 
                args.r = 13
                if algo == 'ToMe':
                    get_tome_model(model, args)
                elif algo == 'PiToMe':
                    get_pitome_model(model, args)
                elif algo == 'DiffRate':
                    get_diffrate_model(model, args)
                else:
                    args.ratio = 1.0 
                    get_tome_model(model, args)
                stats = main(args, model,logger)
                wandb.log(stats)
                
                
        
                
