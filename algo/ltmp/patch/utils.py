import csv
from collections import OrderedDict
from functools import partial

import torch
import torch.utils.data
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.dataset import IterableImageDataset
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
from timm.data.loader import MultiEpochsDataLoader, PrefetchLoader, _worker_init, fast_collate
from timm.data.transforms_factory import create_transform
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from timm.models.vision_transformer import checkpoint_filter_fn

import wandb


def create_vision_transformer(transformer_class, variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError("features_only not implemented for Vision Transformer models.")

    if "flexi" in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation="bilinear", antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop("pretrained_cfg", None))
    model = build_model_with_cfg(
        transformer_class,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=False,
        **kwargs
    )
    return model


def create_loader(
    dataset,
    input_size,
    batch_size,
    is_training=False,
    shuffle=None,
    use_prefetcher=True,
    no_aug=False,
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_split=False,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    num_aug_repeats=0,
    num_aug_splits=0,
    interpolation="bilinear",
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    num_workers=1,
    distributed=False,
    crop_pct=None,
    collate_fn=None,
    pin_memory=False,
    fp16=False,  # deprecated, use img_dtype
    img_dtype=torch.float32,
    device=torch.device("cuda"),
    tf_preprocessing=False,
    use_multi_epochs_loader=False,
    persistent_workers=True,
    worker_seeding="all",
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )
    if hasattr(dataset, "dataset"):
        dataset.dataset.transform = transform
    else:
        dataset.transform = transform

    if isinstance(dataset, IterableImageDataset):
        # give Iterable datasets early knowledge of num_workers so that sample estimates
        # are correct before worker processes are launched
        dataset.set_loader_cfg(num_workers=num_workers)

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate

    loader_class = torch.utils.data.DataLoader
    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    if shuffle is None:
        shuffle = is_training
    shuffle = not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and shuffle

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers,
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop("persistent_workers")  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)
    if use_prefetcher:
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.0
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_size[0],
            device=device,
            fp16=fp16,  # deprecated, use img_dtype
            img_dtype=img_dtype,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
        )

    return loader


def update_summary(
    epoch,
    train_metrics,
    minival_metrics,
    eval_metrics,
    filename,
    lr=None,
    write_header=False,
    log_wandb=False,
):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([("train_" + k, v) for k, v in train_metrics.items()])
    rowd.update([("minival_" + k, v) for k, v in minival_metrics.items()])
    rowd.update([("eval_" + k, v) for k, v in eval_metrics.items()])
    if lr is not None:
        rowd["lr"] = lr
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode="a") as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)



## Register all vision transformer variants to which you want to apply ltpm here.
