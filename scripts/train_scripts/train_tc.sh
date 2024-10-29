MODEL=$1
ALGO=$1
RATIO=$1
TASK=$1

CUDA_VISIBLE_DEVICES=$5 python -m accelerate.commands.launch main_tc.py \
   --model $MODEL \
   --algo $ALGO \
   --ratio $RATIO \
   --task $TASK 