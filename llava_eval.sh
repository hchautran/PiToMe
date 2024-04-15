export DATASET_PATH='/media/caduser/MyBook/chau/.cache'
export HF_DATASETS_CACHE='/media/caduser/MyBook/chau/.cache'
export TASK='coco_cap'
export TASK='scienceqa_full'
export TASK='vqav2'
# export TASK='flickr30k'
accelerate launch  --main_process_port 29501 --num_processes=6 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b"   --tasks $TASK --log_samples --log_samples_suffix llava_v1.5_$TASK --output_path ./logs/ --batch_size 1 
