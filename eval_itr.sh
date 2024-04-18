
# Loop through each element in the array
for dataset in 'coco' 
# for dataset in 'flickr' 'coco' 
do
    # for model in  'blip' 'blip2' 'albef'
    for model in  'blip' 
    do
        # for algo in 'diffrate' 
        # sh eval_scripts/eval_itr_$model.sh $dataset none 1.0 
        # for algo in  'pitome' 'tome' 'tofu' 'dct' 'diffrate'
        for algo in  'diffrate'
        do
            # for ratio in '0.925' 
            # for ratio in '0.875' '0.9' '0.925' '0.95' '0.975'
            for ratio in  '0.925' '0.95' '0.9625' '0.975'
            do
            echo "running $model $size $algo $ratio."
            sh eval_scripts/eval_itr_$model.sh $dataset $algo $ratio
            done
        done
    done
done