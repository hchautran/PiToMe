MODEL=$1
ALGO=$2
RATIO=$3
TASK=$4

CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch main_tc.py \
   --model $MODEL \
   --algo $ALGO \
   --ratio $RATIO \
   --task $TASK 