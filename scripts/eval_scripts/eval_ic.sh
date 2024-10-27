
# export CKT_SIZE='vit_base_patch16_mae'
# export CKT_SIZE='vit_large_patch16_mae'
# export CKT_SIZE='vit_huge_patch14_mae'

# export CKT_SIZE='deit_tiny_patch16_224'
# export CKT_SIZE='deit_small_patch16_224'
# export CKT_SIZE='deit_base_patch16_224'

export ARCH=$1 # DEIT, MAE
export MODEL_SIZE=$2 # T:tiny S:small , B:base , L:large, H:huge
export INPUT_SIZE=$3 # 224, 384
export RATIO=$4 # 0.9 0.925 0.95 0.975
export ALGO=$5 # pitome, tome, tofu, mctf, crossget, diffrate


# python main_ic.py --eval --load_compression_rate ---algo $1 --use_k False --ratio 0.925 --input_size ${INPUT_SIZE} 

python main_ic.py \
   --batch-size 256 \
   --model ${ARCH}-${MODEL_SIZE}-${INPUT_SIZE}  \
   --algo ${ALGO} \
   --ratio ${RATIO} \
   --eval
