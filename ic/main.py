# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
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
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import load_dataset
import models_mae
from accelerated_engine import train_one_epoch, evaluate
from samplers import RASampler
import utils
import shutil
import warnings
from timm.scheduler.cosine_lr import CosineLRScheduler

from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import torch
import pitome
import tome
from dotenv import load_dotenv
from utils import build_transform, DATA_PATH
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
import wandb


torch.hub.set_dir(f'{DATA_PATH}/.vision_ckts')

warnings.filterwarnings('ignore')


def process_image(batch, transform):
    # start = time.time()
    images = []
    labels = []
    for item in batch:
        images.append(transform(item['image']).unsqueeze(0))
        labels.append(item['label'])
    images_tensor = torch.cat(images)
    labels_tensor = torch.tensor(labels)
    # print(time.time() - start)

    return images_tensor, labels_tensor



def get_args_parser():
    parser = argparse.ArgumentParser('Diffrate training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--ratio', default=0.9125, type=float)
    parser.add_argument('--reduced_token', default=8, type=int)
    parser.add_argument('--use_r', default=False, type=bool)

    # Model parameters
    parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--multi-reso', default=False, action='store_true',help='')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.00,
                        help='weight decay (default: 0.00)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./log/temp',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--autoresume', action='store_true', help='auto resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--port', default="15662", type=str,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--target_flops', type=float, default=3.0)
    parser.add_argument('--granularity', type=int, default=4, help='the token number gap between each compression rate candidate')
    parser.add_argument('--load_compression_rate', action='store_true', help='eval by exiting compression rate in compression_rate.json')
    parser.add_argument('--warmup_compression_rate', action='store_true', default=False, help='inactive computational constraint in first epoch')
    return parser

gray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
def filter_out_grayscale(example):
    img_tensor = gray_transform(example['image'])
    # Check if the image has only one channel (grayscale)
    if img_tensor.shape[0] == 3:
        return True
    return False



def prepare_model(args, use_r=False):
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    args.use_r=use_r
    if 'deit'  in args.model:
        pitome.patch.deit(model,use_r=args.use_r)
        model.ratio=float(args.ratio)
        model.r=int(args.reduced_token)
    elif 'mae' in args.model:
        pitome.patch.mae(model,use_r=args.use_r)
        model.ratio=float(args.ratio)
        model.r=int(args.reduced_token)
    elif 'vit' in args.model:
        pitome.patch.aug(model)
        model.ratio=float(args.ratio)
        model.r =int(args.reduced_token)
    else:
        raise ValueError("only support deit, mae and caformer in this codebase")
    return model


def main(args):
    accelerator = Accelerator() 
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir,dist_rank=utils.get_rank())
    logger.info(args)


    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    dataset = load_dataset("imagenet-1k", cache_dir=f"{DATA_PATH}/imagenet/")

    dataset_train = dataset['train']
    dataset_val = dataset['validation']
    dataset_train = dataset_train.filter(filter_out_grayscale, num_proc=10)
    dataset_val = dataset_val.filter(filter_out_grayscale, num_proc=10)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    train_transform =  build_transform(is_train=True, args=args) 
    eval_transform =  build_transform(is_train=False, args=args) 
    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        collate_fn=lambda batch: process_image(batch, train_transform),
        # shuffle=False,
        drop_last=True,
        num_workers = 16,
        pin_memory=True,
        persistent_workers=True

        # sampler=sampler_train
    )

    data_loader_val = DataLoader(
        dataset_val, 
        batch_size=args.batch_size,
        collate_fn=lambda batch: process_image(batch, eval_transform),
        # shuffle=False,
        drop_last=False,
        sampler=sampler_val,
        num_workers = 8, prefetch_factor = 8, pin_memory=True, 
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=1000)
    

    accelerator.print(f"Creating model: {args.model}")
    model = prepare_model(args, use_r=False)

            
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cuda:5', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cuda:5')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                accelerator.print(f"Removing key {k} from pretrained checkpoint")
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

    model = accelerator.prepare(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=0)
    lr_scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=args.min_lr, decay_rate=args.decay_rate )
    optimizer, lr_scheduler, data_loader_train, data_loader_val = accelerator.prepare(optimizer, lr_scheduler, data_loader_train, data_loader_val)


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print(f'number of params: {n_parameters}')

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0

    args.lr = linear_scaled_lr


    if args.eval:
        test_stats = evaluate(data_loader_val, model, accelerator)
        accelerator.print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    else:
        pass
        # if accelerator.is_main_process:
        #     wandb.init(
        #         name=args.model,
        #         project='ic',
        #         config={
        #             'compress_method': 'pitome'
        #         }
        #     )
    

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    
    if args.autoresume and os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1


    accelerator.print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
            # data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, 
            criterion=criterion, 
            data_loader=data_loader_train,
            optimizer=optimizer, 
            accelerator=accelerator,
            epoch=epoch, 
            logger=logger,
            mixup_fn=mixup_fn,
        )
        # if accelerator.is_main_process:
            # wandb.log(train_stats)

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, accelerator)
        accelerator.print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if accelerator.is_main_process and max_accuracy < test_stats['acc1'] :
            shutil.copyfile(checkpoint_path, f'{args.output_dir}/model_best.pth')
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            # wandb.log({'acc': f'{test_stats["acc1"]}%'})
            # wandb.log({'max acc': f'{max_accuracy:.2f}%'})
        accelerator.print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # if accelerator.is_main_process():
            # wandb.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    accelerator.print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
