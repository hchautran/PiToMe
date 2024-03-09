
# export CKT_SIZE='vit_base_patch16_mae'
# export CKT_SIZE='vit_large_patch16_mae'
# export CKT_SIZE='vit_huge_patch14_mae'

# export CKT_SIZE='deit_tiny_patch16_224'
export CKT_SIZE='deit_small_patch16_224'
# export CKT_SIZE='deit_base_patch16_224'

# export CKT_SIZE='vit_small_patch16_224'
# export CKT_SIZE='vit_base_patch16_224'
# export CKT_SIZE='vit_large_patch16_224'

python -m accelerate.commands.launch \
--config_file accelerate.yml main_ic.py \
--epoch 30 --batch-size 200 \
--model ${CKT_SIZE} \
--use_k False \
--ratio 0.925 \
--algo $1 
