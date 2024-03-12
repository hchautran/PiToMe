# LIB_PATH='/media/caduser/MyBook/chau/miniconda3/envs/PiToMe/lib/python3.11/site-packages'
DATASET=$1 # coco or flickr
python -m torch.distributed.run --nproc_per_node=5 main_vl.py --cfg-path eval_scripts/blip2_itr_${DATASET}.yaml --algo $2 --use_k False --ratio $3 --model blip2 --eval