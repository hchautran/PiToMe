
# export CKT_SIZE='vit_base_patch16_mae'
# export CKT_SIZE='vit_large_patch16_mae'
# export CKT_SIZE='vit_huge_patch14_mae'

# export CKT_SIZE='deit_tiny_patch16_224'
# export CKT_SIZE='deit_small_patch16_224'
# export CKT_SIZE='deit_base_patch16_224'

# export CKT_SIZE='vit_small_patch16_224'
# export CKT_SIZE='vit_base_patch16_224'
# export CKT_SIZE='vit_base_patch16_384'
# export CKT_SIZE='vit_large_patch16_224'
export ARCH=$1
export SIZE=$2
export INPUT_SIZE=$3
export RATIO=$4
export ALGO=$5


# python main_ic.py --eval --load_compression_rate ---algo $1 --use_k False --ratio 0.925 --input_size ${INPUT_SIZE} 

python main_ic.py --eval --batch-size 200 --model ${ARCH}_${SIZE}_patch16_${INPUT_SIZE}  --algo ${ALGO} --use_k False --ratio ${RATIO} --input-size ${INPUT_SIZE} 
