
# export CKT_SIZE='vit_base_patch16_mae'
export CKT_SIZE='vit_large_patch16_mae'
# export CKT_SIZE='vit_huge_patch14_mae'

# export CKT_SIZE='deit_tiny_patch16_224'
# export CKT_SIZE='deit_small_patch16_224'
# export CKT_SIZE='deit_base_patch16_224'

# export CKT_SIZE='vit_small_patch16_224'
# export CKT_SIZE='vit_base_patch16_224'
# export CKT_SIZE='vit_large_patch16_224'

python -m torch.distributed.launch \
--nproc_per_node=4 --use_env  \
--master_port 29513 ic/main_pitome.py \
--epoch 10 --batch-size 64 \
--model ${CKT_SIZE} \
--ratio 0.94
