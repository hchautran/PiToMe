#!/usr/bin/env bash

FAIRSEQ_PATH=/home/caduser/HDD/vit_token_compress/PiToMe/sumarization
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/bart_dct

TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=5e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=/media/caduser/MyBook/chau/model.pt

TENSORBOARD_LOG=${FAIRSEQ_PATH}/tb_logs/
CHECKPOINTS_DIR=${FAIRSEQ_PATH}/checkpoints/

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train.py cnn_dm-bin \
    --seed 666 \
    --restore-file $BART_PATH \
    --user-dir $FAIRSEQ_USER_DIR \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --reset-optimizer --reset-dataloader --reset-meters \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --required-batch-size-multiple 1 \
    --arch bart_large_dct \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES \
    --max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir $TENSORBOARD_LOG \
    --save-dir $CHECKPOINTS_DIR \
    --dct-lowpass \
    --dct-layers 2 \
    --dct-percentage 0.5 \
    --global-residual