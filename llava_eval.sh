export DATASET_PATH='/media/caduser/MyBook/chau/.cache'
export HF_DATASETS_CACHE='/media/caduser/MyBook/chau/.cache'
accelerate launch  --main_process_port 29501 --num_processes=6 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b"   --tasks flickr30k   --batch_size 1 --log_samples --log_samples_suffix llava_v1.5_vqav2 --output_path ./logs/ --batch_size 16 
