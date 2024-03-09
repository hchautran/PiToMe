LIB_PATH='/media/caduser/MyBook/chau/miniconda3/envs/PiToMe/lib/python3.11/site-packages'
# DATASET = 'coco'
DATASET='flickr'
MODEL=$1
python -m torch.distributed.run --nproc_per_node=5 eval_vl.py --cfg-path ${LIB_PATH}/lavis/projects/$1/eval/ret_${DATASET}_eval.yaml --algo $2