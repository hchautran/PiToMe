#!/usr/bin/env bash


FAIRSEQ_PATH=/your/path
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/bart_dct
BART_PATH=${FAIRSEQ_PATH}/bart.large

dct_percentage=0.3
dct_layers=2

echo $dct_percentage
echo $dct_layers


TENSORBOARD_LOG=${FAIRSEQ_PATH}/tb_logs/bart_large_${dct_percentage}_${dct_layers}
CHECKPOINTS_DIR=${FAIRSEQ_PATH}/saved_checkpoints/checkpoint_bart/bart_large_${dct_percentage}_${dct_layers}

CUDA_VISIBLE_DEVICES=7 python train.py /path/to/data \
	--seed 926 \
	--restore-file $BART_PATH \
	--user-dir $FAIRSEQ_USER_DIR \
	--max-tokens 4096 \
	--task translation \
	--truncate-source \
	--reset-optimizer --reset-dataloader --reset-meters \
	--arch bart_large_dct \
	--criterion cross_entropy \
	--dropout 0.1 --attention-dropout 0.1 \
	--weight-decay 0.01 --optimizer adam \
	--clip-norm 0.1 \
	--lr-scheduler polynomial_decay --lr 0.00006 --total-num-update 50000 \
	--max-update 50000 --warmup-updates 10000 \
	--fp16 --update-freq 4 \
	--skip-invalid-size-inputs-valid-test \
	--tensorboard-logdir $TENSORBOARD_LOG \
	--save-dir $CHECKPOINTS_DIR \
	--dct-lowpass \
	--dct-layers ${dct_layers} \
	--dct-percentage ${dct_percentage} \
	--global-residual
	
	
	
