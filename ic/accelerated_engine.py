# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from .utils import *
from timm.utils import accuracy, ModelEma
import time
import wandb
from tqdm.auto import tqdm
from accelerate import Accelerator



def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, accelerator:Accelerator, logger, mixup_fn: Optional[Mixup] = None,
                    
    ):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    logger.info_freq = 10

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, logger.info_freq, header,logger)):
        # print('got here')
        with accelerator.autocast():
            optimizer.zero_grad()
            # print(samples.shape)

            # if mixup_fn is not None:
            #     samples, targets = mixup_fn(samples, targets)

            # outputs, flops = model(samples)
            # loss = criterion(outputs, targets)
            # # if utils.is_main_process():
            #     # wandb.log({'current loss': loss_cls_value})
            
            # accelerator.backward(loss)
            # if accelerator.sync_gradients:
            #     accelerator.clip_grad_norm_(model.parameters(), 1.0)
            # optimizer.step() 
            
            # metric_logger.update(loss_cls=loss.item())
            # metric_logger.update(flops=flops/1e9)
            metric_logger.update(loss_cls=0.0)
            metric_logger.update(flops=0.0)


    metric_logger.synchronize_between_processes()
    accelerator.print(f"Averaged stats:{metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, accelerator=None):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    
    for images, targets  in tqdm(data_loader):
        # start = time.time()

        output, flops = model(images)
        loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(flops=flops/1e9)
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # print('eval time:', time.time() - start)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    accelerator.print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} flops {flops.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, flops=metric_logger.flops))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}