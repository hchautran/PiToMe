#!/usr/bin/env bash

FAIRSEQ_PATH=/your/path
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/bart_dct
BART_PATH=${FAIRSEQ_PATH}/bart.large/model.pt

TENSORBOARD_LOG=${FAIRSEQ_PATH}/tb_logs/
CHECKPOINTS_DIR=${FAIRSEQ_PATH}/checkpoints/
CUDA_VISIBLE_DEVICES=4,5,6,7 python custom_code/train.py /path/to/pile-bin \
    --user-dir $FAIRSEQ_USER_DIR \
    --seed 926 \
    --fp16 \
    --mask 0.3 --tokens-per-sample 512 \
    --total-num-update 5000 --max-update 5000 --warmup-updates 200 \
    --task denoising --save-interval 1 \
    --arch bart_large_dct \
    --optimizer adam --lr-scheduler polynomial_decay --lr 0.00005 --dropout 0.1 \
    --criterion cross_entropy --max-tokens 2048 --weight-decay 0.01 --attention-dropout 0.1 \
    --share-all-embeddings --clip-norm 0.1 \
    --log-format tqdm --log-interval 1 --save-interval-updates 500 --update-freq 400 \
    --validate-interval-updates 500 \
    --no-epoch-checkpoints --mask-length span-poisson --replace-length 1 \
    --encoder-learned-pos --decoder-learned-pos --rotate 0.0 \
    --mask-random 0.1 --permute-sentences 1 --insert 0.0 --poisson-lambda 3.5 --dataset-impl mmap \
    --bpe gpt2 --num-workers 4 \
    --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir $TENSORBOARD_LOG \
    --save-dir $CHECKPOINTS_DIR \
    --dct-lowpass \
    --dct-layers 2 \
    --dct-percentage 0.5 \
    --global-residual
