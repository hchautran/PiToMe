
python -m torch.distributed.run --nproc_per_node=5 main_vl.py --cfg-path train_scripts/clip_itr_${DATASET}.yaml --algo $2 --use_k False --ratio $3 --model clip 