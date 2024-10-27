
# export CKT_SIZE='vit_base_patch16_mae'
# export CKT_SIZE='vit_large_patch16_mae'
# export CKT_SIZE='vit_huge_patch14_mae'

# export CKT_SIZE='deit_tiny_patch16_224'
# export CKT_SIZE='deit_small_patch16_224'
# export CKT_SIZE='deit_base_patch16_224'

# export CKT_SIZE='vit_small_patch17_224'
# export CKT_SIZE='vit_base_patch16_224'
# export CKT_SIZE='vit_base_patch16_384'
# export CKT_SIZE='vit_large_patch16_224'
export MODEL=$1
export RATIO=$2
export ALGO=$3



python main_ic.py \
   --batch-size 100 \
   --model ${MODEL} \
   --algo ${ALGO} \
   --ratio ${RATIO} \
   --eval
