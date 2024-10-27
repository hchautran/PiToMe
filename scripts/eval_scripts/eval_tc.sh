ALGO=$1
RATIO=$2
TASK=$3
MODEL=$4

python main_tc.py \
   --algo $ALGO \
   --ratio $RATIO \
   --task $TASK \
   --model $MODEL \
   --eval 