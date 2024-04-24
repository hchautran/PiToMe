# LIB_PATH='/media/caduser/MyBook/chau/miniconda3/envs/PiToMe/lib/python3.11/site-packages'
# python -m torch.distributed.run --nproc_per_node=5 main_vl.py --cfg-path ${LIB_PATH}/lavis/projects/blip2/train/retrieval_coco_ft.yaml --algo $1 --use_k False --ratio $2 --model blip2
export DATASET=$1
python -m torch.distributed.run --nproc_per_node=5 main_vl.py --cfg-path train_scripts/blip2_itr_$1.yml --algo $2 --use_k False --ratio $3 --model blip2