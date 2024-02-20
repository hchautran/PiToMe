
# export CKT_SIZE='vit_base_patch16_mae'
# export CKT_SIZE='vit_large_patch16_mae'
# export CKT_SIZE='vit_huge_patch14_mae'

# export CKT_SIZE='deit_tiny_patch16_224'
export CKT_SIZE='deit_small_patch16_224'
# export CKT_SIZE='deit_base_patch16_224'

# export CKT_SIZE='vit_small_patch16_224'
# export CKT_SIZE='vit_base_patch16_224'
# export CKT_SIZE='vit_large_patch16_224'

# export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1




python -m accelerate.commands.launch \
--config_file accelerate.yml --main_process_port 29513 ic/main.py \
--epoch 20 --batch-size 256 \
--model ${CKT_SIZE} \
--ratio 0.9125 --use_r False --reduced_token 13 

# accelerate launch --config_file --main_process_port 63333  --rdzv_backend 'static' accelerate.yml main.py
# --nproc_per_node=5 --use_env  \
# --epoch 20 --batch-size 256 \
# --model ${CKT_SIZE} \
# --ratio 0.9125 --use_r False --reduced_token 13 