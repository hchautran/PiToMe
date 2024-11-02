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
import tasks.ic.models_mae
from tasks.ic.engine import train_one_epoch, evaluate
from tasks.ic.samplers import RASampler
import tasks.ic.utils as utils
import shutil
import warnings
from timm.scheduler.cosine_lr import CosineLRScheduler 
from tasks.ic.utils import build_transform
import torch
from algo import (
    PITOME,
    TOME,
    DCT,
    TOFU,
    DIFFRATE,
    NONE,
    pitome,
    dct,
    tome,
    tofu,
    DiffRate
)
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
import wandb
from torchvision import transforms
from skimage import color
from tasks.ic.utils import DATA_PATH

ALGO = {
    PITOME: pitome,
    TOME: tome,
    DCT: dct,
    TOFU: tofu,
    NONE: tome,
}


torch.hub.set_dir(f'{DATA_PATH}/.vision_ckts')
warnings.filterwarnings('ignore')


def process_image(example, transform):
    example['pixel_values'] = transform(example['image'])
    return example

gray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def process_grayscale(example):
    image = example['image']
    img_tensor = gray_transform(image)
    # Check if the image has only one channel (grayscale)
    if img_tensor.shape[0] == 2:
        example['image'] = color.gray2rgb(image)
    return example

def filter_out_grayscale(example):
    img_tensor = gray_transform(example['image'])
    # Check if the image has only one channel (grayscale)
    if img_tensor.shape[0] == 3:
        return True
    return False
 
def process_image(batch, transform):
    images_tensor = torch.stack([transform(item['image']) for item in batch])
    labels_tensor = torch.tensor([item['label'] for item in batch])
    return images_tensor, labels_tensor


model_dict = {
    'DEIT-T-224':'deit_tiny_patch16_224',
    'DEIT-S-224':'deit_small_patch16_224',
    'DEIT-B-224':'deit_base_patch16_224',
    'DEIT-T-384':'deit_tiny_patch16_384',
    'DEIT-S-384':'deit_small_patch16_384',
    'DEIT-B-384':'deit_base_patch16_384',
    'MAE-B-224':'vit_base_patch16_mae',
    'MAE-L-224':'vit_large_patch16_mae',
    'MAE-H-224':'vit_huge_patch14_mae',
}


def get_args_parser():
    parser = argparse.ArgumentParser('ic training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--ratio', default=0.9125, type=float)
    parser.add_argument('--reduced_token', default=8, type=int)
    parser.add_argument('--algo', default=PITOME) 

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
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-7, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-8, metavar='LR',
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
    parser.add_argument('--ThreeAugment', action='store_true') #3augment

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

def get_model(model, args):
    if 'deit' in model_dict[args.model]:
        ALGO[args.algo].patch.deit(model)
        model.ratio=float(args.ratio)
    elif 'mae' in model_dict[args.model]:
        ALGO[args.algo].patch.mae(model)
        model.ratio=float(args.ratio)
    else:
        raise ValueError("Only support DEIT and MAE models in this codebase")



def get_diffrate_model(model, args):
    if 'deit' in model_dict[args.model]:
        DiffRate.patch.deit(model, prune_granularity=args.granularity, merge_granularity=args.granularity)
    elif 'mae' in model_dict[args.model]:
        DiffRate.patch.mae(model, prune_granularity=args.granularity, merge_granularity=args.granularity)
    else:
        raise ValueError("only support deit, mae and caformer in this codebase")
    model.init_kept_num_using_ratio(args.ratio)
            

def main(args):
    accelerator = Accelerator(mixed_precision='no') 
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir,dist_rank=utils.get_rank())
    logger.info(args)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    args.data_path = DATA_PATH + '/.cache/'
    # args.data_set  = 'CIFAR'
    dataset_train, args.nb_classes = utils.build_dataset(is_train=True, args=args)
    dataset_val, _ = utils.build_dataset(is_train=False, args=args)
    dataset_train = dataset_train.filter(filter_out_grayscale, num_proc=10)
    dataset_val = dataset_val.filter(filter_out_grayscale, num_proc=10)

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

    train_transform =  build_transform(is_train=True, args=args) 
    eval_transform =  build_transform(is_train=False, args=args) 
    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda batch: process_image(batch, train_transform),
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1 * args.batch_size),
        num_workers=10,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda batch: process_image(batch, eval_transform),
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, 
            cutmix_alpha=args.cutmix, 
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, 
            switch_prob=args.mixup_switch_prob, 
            mode=args.mixup_mode,
            label_smoothing=args.smoothing, 
            num_classes=args.nb_classes
        )
    
    logger.info(f"Creating model: {args.model}")
    model = create_model(
        model_dict[args.model],
        pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    if args.algo == DIFFRATE: 
        get_diffrate_model(model, args)
    elif args.algo != NONE:
        get_model(model, args)
    else:
        args.ratio = 1.0
        get_model(model, args)


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

    
    model = accelerator.prepare(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=args.min_lr) 
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, min_lr=args.min_lr)
    loss_scaler = tasks.ic.utils.NativeScalerWithGradNormCount()
    optimizer, lr_scheduler, data_loader_train, data_loader_val = accelerator.prepare(optimizer, lr_scheduler, data_loader_train, data_loader_val)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f'number of params: {n_parameters}')
    linear_scaled_lr = args.lr * args.batch_size * tasks.ic.utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    if args.eval:
        test_stats = evaluate(data_loader_val, model, accelerator)
        accelerator.print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        test_stats['best acc'] = test_stats['acc1']
        return test_stats
    else:
        # pass
        if accelerator.is_main_process:
            wandb.init(
                name=f'{args.model}-{args.algo}',
                project=f'ic-{args.model}',
                config={
                    'compress_method': args.algo,
                    'model': args.model,
                    'ratio': args.ratio
                }
            )
 
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
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    accelerator.print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):

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
        if accelerator.is_main_process:
            wandb.log(train_stats)

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                tasks.ic.utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, accelerator)
        # lr_scheduler.step(test_stats['acc1'])
        accelerator.print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        if accelerator.is_main_process and max_accuracy < test_stats['acc1'] :
            shutil.copyfile(checkpoint_path, f'{args.output_dir}/model_best.pth')
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            wandb.log({'acc': f'{test_stats["acc1"]}'})
            wandb.log({'max acc': max_accuracy})
        accelerator.print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if accelerator.is_main_process:
            wandb.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    accelerator.print('Training time {}'.format(total_time_str))
    test_stats['best acc'] = f'{max_accuracy:.2f}'
    return test_stats


if __name__ == '__main__':
    import pathlib
   
    parser = argparse.ArgumentParser('Image classification training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    abs_path = f'{os.getcwd()}/outputs/ic_outputs'
    Path(abs_path).mkdir(parents=True, exist_ok=True)
    file_name = f'{"eval" if args.eval else "train"}-{args.model}.csv'
    path = f'{abs_path}/{file_name}'
    if not pathlib.Path(path).is_file():
        head = "model, algo, gflops, ratio ,acc_1\n"
        if utils.is_main_process():
            with open(path, "a") as myfile:
                myfile.write(head)
        
    metrics = main(args)
    if metrics is not None:
        row = f'{args.model}, {args.algo}, {metrics["flops"]}, {args.ratio}, {metrics["best acc"]}\n'
        if utils.is_main_process():
            with open(path, "a") as myfile:
                myfile.write(row)
