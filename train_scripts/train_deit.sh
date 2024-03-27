
# export CKT_SIZE='vit_base_patch16_mae'
# export CKT_SIZE='vit_large_patch16_mae'
# export CKT_SIZE='vit_huge_patch14_mae'
export INPUT_SIZE=$2
export ARCH=$3
export SIZE=$4

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
export BATCH_SIZE=8
export EPOCH=120

python -m accelerate.commands.launch --config_file accelerate.yml main_ic.py --batch-size $BATCH_SIZE --model ${ARCH}_${SIZE}_patch16_${INPUT_SIZE}  --algo ${ALGO} --use_k False --ratio ${RATIO} --input-size ${INPUT_SIZE} --epoch $EPOCH  --lr 0.0001
