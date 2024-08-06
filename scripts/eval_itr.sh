
# Loop through each element in the array
# for dataset in 'coco' 
# for dataset in 'flickr' 'coco' 
for dataset in 'flickr' 
do
    # for model in  'blip' 'blip2' 'albef'
    for model in  'blip' 
    do
        # for algo in 'diffrate' 
        # sh scripts/eval_scripts/eval_itr_$model.sh $dataset none 1.0 
        for algo in 'pitome' 
        # for algo in  'diffrate'
        do
            for ratio in '0.85' '0.875' '0.9' '0.925' 
            do
                echo "running $model $size $algo $ratio."
                sh scripts/eval_scripts/eval_itr_$model.sh $dataset $algo $ratio
            done
        done
    done
done