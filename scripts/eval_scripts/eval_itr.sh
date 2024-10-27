# LIB_PATH='/media/caduser/MyBook/chau/miniconda3/envs/PiToMe/lib/python3.11/site-packages'
DATASET=$1 # coco or flickr
MODEL=$2 # albef clip blip blip2 
python -m torch.distributed.run --nproc_per_node=5 main_itr.py \
   --cfg-path scripts/eval_scripts/${MODEL}_itr_${DATASET}.yml \
   --algo $2 \
   --ratio $3 \
   --model $MODEL \
   --dataset $1 \
   --eval