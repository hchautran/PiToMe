
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
export SIZE=$2
export INPUT_SIZE=$3
export RATIO=$4
export ALGO=$5
export BATCH_SIZE=128
export EPOCH=2

CUDA_VISIBLE_DEVICES=$6 python -m accelerate.commands.launch --main_process_port 2950$6 main_ic.py --batch-size $BATCH_SIZE --model ${ARCH}_${SIZE}_patch16_${INPUT_SIZE}  --algo ${ALGO} --use_k False --ratio ${RATIO} --input-size ${INPUT_SIZE} --epoch $EPOCH --lr 0.00001 
