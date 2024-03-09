"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *
from algo import (
    PITOME,
    TOME,
    DIFFRATE,
    DCT,
    TOFU,
    LTMP,
    NONE, 
    pitome,
    tome,
    DiffRate,
    tofu,
    # ltmp
)

def get_tome_model(model, args):
    if 'blip2' in args.model:
        tome.patch.blip2(model.visual_encoder,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.r=int(args.reduced_token)
    elif 'blip' in args.model:
        tome.patch.blip(model.visual_encoder,use_k=args.use_k)
        tome.patch.blip(model.visual_encoder_m,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.visual_encoder_m.ratio=float(args.ratio)
        model.visual_encoder.r=int(args.reduced_token)
        model.visual_encoder_m.r=int(args.reduced_token)
    else:
        raise ValueError("only support deit, mae and caformer in this codebase")




def get_pitome_model(model, args):
    if 'blip2' in args.model:
        pitome.patch.blip2(model.visual_encoder,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.r=int(args.reduced_token)
    elif 'blip' in args.model:
        pitome.patch.blip(model.visual_encoder,use_k=args.use_k)
        pitome.patch.blip(model.visual_encoder_m,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.visual_encoder_m.ratio=float(args.ratio)
        model.visual_encoder.r=int(args.reduced_token)
        model.visual_encoder_m.r=int(args.reduced_token)
    else:
        raise ValueError("only support deit, mae and caformer in this codebase")



def get_diffrate_model(model, args):
    if 'blip2' in args.model:
        DiffRate.patch.blip2(model.visual_encoder, prune_granularity=args.granularity, merge_granularity=args.granularity)
    elif 'blip' in args.model:
        DiffRate.patch.blip(model.visual_encoder, prune_granularity=args.granularity, merge_granularity=args.granularity)
    else:
        raise ValueError("only support deit, mae and caformer in this codebase")
    if args.use_k:
        model.init_kept_num_using_r(args.reduced_token)
    else:
        model.init_kept_num_using_ratio(args.ratio)
    

def get_tofu_model(model, args):
    if 'blip2' in args.model:
        tofu.patch.blip2(model.visual_encoder, prune_granularity=args.granularity, merge_granularity=args.granularity)
    elif 'blip' in args.model:
        tofu.patch.blip(model.visual_encoder, prune_granularity=args.granularity, merge_granularity=args.granularity)
    else:
        raise ValueError("only support deit, mae and caformer in this codebase")
    if args.use_k:
        model.init_kept_num_using_r(args.reduced_token)
    else:
        model.init_kept_num_using_ratio(args.ratio)
    
            


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--algo", default=PITOME, required=True, help="compress method")
    parser.add_argument("--model", default='blip', required=True, help="model_type")
    parser.add_argument("--use_k", default=False)
    parser.add_argument("--ratio", default=0.9, type=float)
    parser.add_argument("--reduced_token", default=12, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    args = parse_args()
    cfg = Config(args)


    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    if args.algo == TOME:
        get_tome_model(model, args)
    elif args.algo == PITOME:
        get_pitome_model(model, args)
    elif args.algo == DIFFRATE:
        get_diffrate_model(model, args)
    elif args.algo == TOFU:
        get_tofu_model(model, args)
    else:
        get_tome_model(model, args)


    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    # metrics = runner.evaluate(skip_reload=True)['test']
    if args.eval:
        metrics = runner.evaluate(skip_reload=True)['test']
        if metrics is not None:
            print('r_sum', metrics['txt_r10'] + metrics['txt_r5'] + metrics['txt_r1'] + metrics['img_r10'] + metrics['img_r5'] + metrics['img_r1'])
    else:
        runner.train()
    print('gflops', model.visual_encoder.total_flop/1e9)


if __name__ == "__main__":
    main()