"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import numpy as np
import argparse
import random
import torch
import time

import lavis.tasks as tasks
import torch.backends.cudnn as cudnn
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now
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
    MCTF,
    NONE, 
    pitome,
    tome,
    DiffRate,
    tofu,
    dct, 
    mctf,
    # ltmp
)

def get_tome_model(model, args):
    if 'clip' in args.model:
        tome.patch.clip(model.visual.transformer,use_k=args.use_k)
        model.visual.transformer.ratio=float(args.ratio)
        model.visual.transformer.r=float(args.reduced_token)
    elif 'blip2' in args.model:
        tome.patch.blip2(model.visual_encoder,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.r=int(args.reduced_token)
    elif 'blip' or 'albef' in args.model:
        tome.patch.blip(model.visual_encoder,use_k=args.use_k)
        tome.patch.blip(model.visual_encoder_m,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.visual_encoder_m.ratio=float(args.ratio)
        model.visual_encoder.r=int(args.reduced_token)
        model.visual_encoder_m.r=int(args.reduced_token)
    else:
        raise ValueError("only support clip, blip and blip2 for image-text retrieval task")

def get_pitome_model(model, args):
    if 'clip' in args.model:
        pitome.patch.clip(model.visual.transformer, use_k=args.use_k)
        model.visual.transformer.ratio=float(args.ratio)
        model.visual.transformer.r=int(args.reduced_token)
    elif 'blip2' in args.model:
        pitome.patch.blip2(model.visual_encoder,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.visual_encoder.r=int(args.reduced_token)
    elif 'blip' or 'albef' in args.model:
        pitome.patch.blip(model.visual_encoder,use_k=args.use_k)
        pitome.patch.blip(model.visual_encoder_m,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.visual_encoder_m.ratio=float(args.ratio)
        model.visual_encoder.r=int(args.reduced_token)
        model.visual_encoder_m.r=int(args.reduced_token)
    else:
        raise ValueError("only support clip, blip, albef and blip2 in this codebase")


def get_mctf_model(model, args):
    if 'clip' in args.model:
        mctf.patch.clip(model.visual.transformer, use_k=args.use_k)
        model.visual.transformer.ratio=float(args.ratio)
        model.visual.transformer.r=int(args.reduced_token)
    elif 'blip2' in args.model:
        mctf.patch.blip2(model.visual_encoder,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.visual_encoder.r=int(args.reduced_token)
    elif 'blip' or 'albef' in args.model:
        mctf.patch.blip(model.visual_encoder,use_k=args.use_k)
        mctf.patch.blip(model.visual_encoder_m,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.visual_encoder_m.ratio=float(args.ratio)
        model.visual_encoder.r=int(args.reduced_token)
        model.visual_encoder_m.r=int(args.reduced_token)
    else:
        raise ValueError("only support clip, blip, albef and blip2 in this codebase")



def get_diffrate_model(model, args):
    if 'clip' in args.model:
        DiffRate.patch.clip(model.visual, prune_granularity=args.granularity, merge_granularity=args.granularity)
        if not args.use_k:
            model.visual.init_kept_num_using_r(args.reduced_token)
        else:
            model.visual.init_kept_num_using_ratio(args.ratio)
    elif 'blip2' in args.model:
        DiffRate.patch.blip2(model.visual_encoder, prune_granularity=args.granularity, merge_granularity=args.granularity)
        if not args.use_k:
            model.visual_encoder.init_kept_num_using_r(args.reduced_token)
        else:
            model.visual_encoder.init_kept_num_using_ratio(args.ratio)
    elif 'blip' or 'albef' in args.model:
        DiffRate.patch.blip(model.visual_encoder, prune_granularity=args.granularity, merge_granularity=args.granularity)
        DiffRate.patch.blip(model.visual_encoder_m, prune_granularity=args.granularity, merge_granularity=args.granularity)
        if not args.use_k:
            model.visual_encoder.init_kept_num_using_r(args.reduced_token)
            model.visual_encoder_m.init_kept_num_using_r(args.reduced_token)
        else:
            model.visual_encoder.init_kept_num_using_ratio(args.ratio)
            model.visual_encoder_m.init_kept_num_using_ratio(args.ratio)
    else:
        raise ValueError("only support deit, mae and caformer in this codebase")

def get_tofu_model(model, args):
    if 'clip' in args.model:
        tofu.patch.clip(model.visual.transformer, use_k=args.use_k)
        model.visual.transformer.ratio=float(args.ratio)
        model.visual.transformer.r=int(args.reduced_token)
    elif 'blip2' in args.model:
        tofu.patch.blip2(model.visual_encoder,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.visual_encoder.r=int(args.reduced_token)
    elif 'blip' or 'albef' in args.model:
        tofu.patch.blip(model.visual_encoder,use_k=args.use_k)
        tofu.patch.blip(model.visual_encoder_m,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.visual_encoder_m.ratio=float(args.ratio)
        model.visual_encoder.r=int(args.reduced_token)
        model.visual_encoder_m.r=int(args.reduced_token)
    else:
        raise ValueError("only support clip, blip and blip2 in this codebase")

def get_dct_model(model, args):
    if 'clip' in args.model:
        dct.patch.clip(model.visual.transformer, use_k=args.use_k)
        model.visual.transformer.ratio=float(args.ratio)
        model.visual.transformer.r=int(args.reduced_token)
    elif 'blip2' in args.model:
        dct.patch.blip2(model.visual_encoder,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.visual_encoder.r=int(args.reduced_token)
    elif 'blip' or 'albef' in args.model:
        dct.patch.blip(model.visual_encoder,use_k=args.use_k)
        dct.patch.blip(model.visual_encoder_m,use_k=args.use_k)
        model.visual_encoder.ratio=float(args.ratio)
        model.visual_encoder_m.ratio=float(args.ratio)
        model.visual_encoder.r=int(args.reduced_token)
        model.visual_encoder_m.r=int(args.reduced_token)
    else:
        raise ValueError("only support clip, blip and blip2 in this codebase")


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--algo", default=PITOME, required=True, help="compress method")
    parser.add_argument("--model", default='blip', required=True, help="model_type")
    parser.add_argument("--use_k", default=False)
    parser.add_argument("--ratio", default=0.9, type=float)
    parser.add_argument("--reduced_token", default=12, type=int)
    parser.add_argument('--granularity', type=int, default=4, help='the token number gap between each compression rate candidate')
    parser.add_argument('--dataset', default='flickr', help='dataset')
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

def calculate_cross_flops(dataset, model, final_shape):
    average_sentence_length = {
       'flickr': 13.4, 
       'coco': 10.5, 
    }
    num_layer = { 
        'albef': 12, 
        'blip': 12, 
        'blip2': 48, 
    }
    _, N_i, C = final_shape 
    print(final_shape)
    N_t = average_sentence_length[dataset]
    num_layers = num_layer[model]
    if model != 'blip2':
        flops = 0
        mhsa_flops = 4*N_t*C*C + 2*N_t*N_t*C
        flops += num_layers*mhsa_flops
        ffn_flops = 8*N_t*C*C
        flops += num_layers*ffn_flops

        mhsa_flops = 2*N_i*C*C + 2*N_t*C*C + 2*N_i*N_t*C
        flops += num_layers*mhsa_flops
        ffn_flops = 8*N_t*C*C
        flops += num_layers*ffn_flops 
        return flops
    else:
        N_t=N_t + 32
        flops = 0
        mhsa_flops = 4*N_t*C*C + 2*N_t*N_t*C
        ffn_flops = 8*N_t*C*C
        flops += num_layers*mhsa_flops
        flops += num_layers*ffn_flops

        mhsa_flops = 2*N_i*C*C + 2*N_t*C*C + 2*N_i*N_t*C
        ffn_flops = 8*N_t*C*C
        flops += num_layers*mhsa_flops
        flops += num_layers*ffn_flops 
        return flops
    
    

def get_gflops(args, model):
    if 'clip' in args.model:
        return model.visual.transformer.total_flop/1e9
    elif 'blip2' in args.model:
        flops = model.visual_encoder.total_flop  + calculate_cross_flops(args.dataset, args.model, model.visual_encoder.final_shape)
        return flops/1e9
    else:
        flops = model.visual_encoder.total_flop  + calculate_cross_flops(args.dataset, args.model, model.visual_encoder.final_shape)
        return flops/1e9
    

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
    elif args.algo == DCT:
        get_dct_model(model, args)
    elif args.algo == MCTF:
        get_mctf_model(model, args)
    elif args.algo == NONE:
        args.ratio = 1.0
        get_tome_model(model, args)
    else:
        raise ValueError("only support pitome, tome, tofu, dct, diffrate for image retrieval task")


    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    # metrics = runner.evaluate(skip_reload=True)['test']
    train_time = 0
    eval_time = 0
    if args.eval:
        start = time.time()
        metrics = runner.evaluate(skip_reload=True)['test']
        eval_time = time.time() - start
        if metrics is not None:
            print('r_sum', metrics['txt_r10'] + metrics['txt_r5'] + metrics['txt_r1'] + metrics['img_r10'] + metrics['img_r5'] + metrics['img_r1'])
    else:
        start = time.time()
        runner.train()
        train_time = time.time() - start
        start = time.time()
        metrics = runner.evaluate(skip_reload=False)
        if metrics is not None: 
            metrics = metrics['test']
        eval_time = time.time() - start
    gflops = get_gflops(args, model)
    if metrics is not None:
        metrics['gflops'] = gflops
    return metrics, args, train_time, eval_time 


if __name__ == "__main__":
    import pathlib
    import time
    model_dict = {
        'clip': 'CLIP',
        'blip': 'BLIP',
        'blip2': 'BLIP2',
        'albef': 'ALBEF'
    }
    abs_path ='/home/caduser/HDD/vit_token_compress/PiToMe'
    metrics, args, train_time, eval_time = main()
    file_name = f'{"eval" if args.eval else "train"}_itr_{model_dict[args.model]}.csv'
    # file_name = f'ablation_study_wo_step.csv'
    path = f'{abs_path}/{file_name}'
    if not pathlib.Path(path).is_file():
        head = "dataset,model,algo,gflops,ratio,txt_r1,txt_r5,txt_r10,img_r1,img_r5,img_r10,r_sum,train time,eval time,use attn\n"
        with open(file_name, "a") as myfile:
            myfile.write(head)

    if metrics is not None:
        sum = metrics["txt_r1"] + metrics["txt_r5"] + metrics["txt_r10"] + metrics["img_r1"] + metrics["img_r5"] + metrics["img_r10"]
        row = f'{args.dataset},{model_dict[args.model]},{args.algo},{metrics["gflops"]},{args.ratio},{metrics["txt_r1"]},{metrics["txt_r5"]},{metrics["txt_r10"]},{metrics["img_r1"]},{metrics["img_r5"]},{metrics["img_r10"]},{sum},{train_time},{eval_time},{"false"}\n'
        with open(file_name, "a") as myfile:
            myfile.write(row)
