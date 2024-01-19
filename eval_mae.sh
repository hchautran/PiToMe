
export CKT_SIZE='huge'
export PATCH='14'
export ALGO=$1
export R='8'
export RATIO='0.95'

python mae/main_finetune.py --eval --resume mae_finetuned_vit_${CKT_SIZE}.pth --model vit_${CKT_SIZE}_patch${PATCH} --batch_size 100 --ratio ${RATIO} --compress_method ${ALGO}  --r ${R}
