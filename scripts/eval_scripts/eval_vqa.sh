# export DATASET_PATH='/media/caduser/MyBook/chau/.cache'
# export HF_DATASETS_CACHE='/media/caduser/MyBook/chau/.cache'
# for tasks in 'coco_cap' 'scienceqa_full' 'vqav2' 'flickr30k'
export model='llava-v1.5-7b'
# for task in  'vqav2' 'vizwiz_vqa_val' 'mmbench' 'mme'
for i in 1 2 3 4 5
do
    for task in   'scienceqa_img' 'mme' 
    do 

        # python -m torch.distributed.run --nproc_per_node=5 lmms-eval/lmms_eval/__main__.py \
        #     --model llava   \
        #     --model_args pretrained="liuhaotian/$model" \
        #     --tasks $task --log_samples \
        #     --log_samples_suffix $model_$task \
        #     --output_path ./logs_baseline/ \
        #     --batch_size 1 \
        #     --algo baseline \
        #     --ratio 1.0 
            # --wandb_args project=$model,name=baseline
        for algo in 'tome' 'pitome' 
        do 
            for ratio in  '0.85' '0.875' '0.9' '0.925' '0.95' '0.975'  
            do 
                python -m torch.distributed.run --nproc_per_node=1 main_vqa.py \
                    --model llava   \
                    --model_args pretrained="liuhaotian/$model" \
                    --tasks $task --log_samples \
                    --log_samples_suffix $model_$task \
                    --output_path ./logs_$algo/ \
                    --batch_size 1 \
                    --algo $algo \
                    --ratio $ratio \
                    # --wandb_args project=$model,name=$algo-$ratio 
            done
        done
    done
done

