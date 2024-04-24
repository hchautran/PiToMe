# LIB_PATH='/media/caduser/MyBook/chau/miniconda3/envs/PiToMe/lib/python3.11/site-packages'
# DATASET = 'coco'
DATASET=$1 # coco or flickr
python -m torch.distributed.run --nproc_per_node=3 main_vl.py --cfg-path train_scripts/blip_itr_${DATASET}.yml --algo $2 --use_k False --ratio $3 --model blip 