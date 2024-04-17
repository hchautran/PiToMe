export DATASET_PATH='/media/caduser/MyBook/chau/.cache'
export HF_DATASETS_CACHE='/media/caduser/MyBook/chau/.cache'
# for tasks in 'coco_cap' 'scienceqa_full' 'vqav2' 'flickr30k'
export model='llava-v1.5-7b'
for task in 'vqav2' 'flickr30k' 'coco_cap' 
do 
    for ratio in '0.95' '0.925' '0.9'
    do 
        for algo in 'tome' 'pitome' 'tofu' 'diffrate' 'dct'
        do 
            accelerate launch  --main_process_port 29501 --num_processes=5 -m lmms_eval \
                --model llava   \
                --model_args pretrained="liuhaotian/$model" \
                --tasks $task --log_samples \
                --log_samples_suffix $model_$task \
                --output_path ./logs_$algo/ \
                --batch_size 1 \
                --algo $algo \
                --ratio $ratio \
                --wandb_args project=$model-$task,name=$algo-$ratio
        done
    done
    accelerate launch  --main_process_port 29501 --num_processes=5 -m lmms_eval \
        --model llava   \
        --model_args pretrained="liuhaotian/$model" \
        --tasks $task --log_samples \
        --log_samples_suffix $model_$task \
        --output_path ./logs_baseline/ \
        --batch_size 1 \
        --algo baseline \
        --ratio 1.0 \
        --wandb_args project=$model-$task,name=baseline
done
