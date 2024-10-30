# LIB_PATH='/media/caduser/MyBook/chau/miniconda3/envs/PiToMe/lib/python3.11/site-packages'
DATASET=$1 # coco or flickr
MODEL=$2 # albef blip blip2 
ALGO=$3 # pitome tome tofu mctf crossget 
RATIO=$4 # 0.9 0.925 0.95 0.975 

python -m torch.distributed.run --nproc_per_node=1 main_itr.py \
   --cfg-path scripts/eval_scripts/${MODEL}_itr_${DATASET}.yml \
   --dataset $DATASET \
   --model $MODEL \
   --algo $ALGO \
   --ratio $RATIO \
   --eval