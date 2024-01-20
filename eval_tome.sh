
export CKT_SIZE='vit_large_patch16_mae'

# export CKT_SIZE='vit_deit_base_patch16_224'

python ic/main_tome.py --eval --load_compression_rate --data-path $path_to_imagenet$ --model ${CKT_SIZE} --target_flops 42.3 
