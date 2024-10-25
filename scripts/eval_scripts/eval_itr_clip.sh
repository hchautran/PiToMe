
DATASET=$1 # coco or flickr
python -m torch.distributed.run --nproc_per_node=1 main_itr.py --cfg-path scripts/eval_scripts/clip_itr_${DATASET}.yml --algo $2 --dataset $1 --use_k False --ratio $3 --model clip --eval