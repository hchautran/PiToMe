
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

python -m accelerate.commands.launch --config_file accelerate.yml main_ic.py --epoch 30 --batch-size 200 --model ${ARCH}_${SIZE}_patch16_${INPUT_SIZE}  --algo $1 --use_k False --ratio 0.925 --input_size ${INPUT_SIZE} 
