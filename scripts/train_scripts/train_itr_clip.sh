DATASET=$1 # coco or flickr
python -m torch.distributed.run --nproc_per_node=4 main_itr.py --cfg-path train_scripts/clip_itr_${DATASET}.yml --algo $2 --use_k False --ratio $3 --model clip 