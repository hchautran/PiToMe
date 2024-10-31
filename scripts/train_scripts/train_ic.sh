
# export CKT_SIZE='vit_base_patch16_mae'
# export CKT_SIZE='vit_large_patch16_mae'
# export CKT_SIZE='vit_huge_patch14_mae'

# export CKT_SIZE='deit_tiny_patch16_224'
# export CKT_SIZE='deit_small_patch16_224'
# export CKT_SIZE='deit_base_patch16_224'

# export CKT_SIZE='vit_small_patch16_224'
# export CKT_SIZE='vit_base_patch16_224'
# export CKT_SIZE='vit_large_patch16_224'
export ARCH=$1
export MODEL_SIZE=$2
export INPUT_SIZE=$3
export RATIO=$4
export ALGO=$5
export BATCH_SIZE=64
export EPOCH=30

CUDA_VISIBLE_DEVICES=$6 python -m accelerate.commands.launch --main_process_port 2950$6 main_ic.py \
   --batch-size $BATCH_SIZE \
   --model ${ARCH}-${MODEL_SIZE}-${INPUT_SIZE}  \
   --algo ${ALGO} \
   --ratio ${RATIO} \
   --epoch $EPOCH  \

