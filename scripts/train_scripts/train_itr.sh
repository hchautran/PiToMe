DATASET=$1 # coco or flickr
MODEL=$2 # blip clip albef blip2
python -m torch.distributed.run \
   --nproc_per_node=1 main_itr.py \
   --cfg-path scripts/train_scripts/${MODEL}_itr_${DATASET}.yml \
   --algo $3 \
   --ratio $4 \
   --model $MODEL 