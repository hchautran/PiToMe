DATASET=$1 # coco or flickr
MODEL=$2 # blip clip albef blip2
python -m torch.distributed.run \
   --nproc_per_node=5 main_itr.py \
   --cfg-path train_scripts/${MODEL}_itr_${DATASET}.yml \
   --algo $2 \
   --ratio $3 \
   --model $MODEL 