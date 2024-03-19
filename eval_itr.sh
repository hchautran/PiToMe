
# Loop through each element in the array
# for dataset in 'flickr' 
for dataset in 'coco' 
do
    # for model in 'clip' 'blip' 
    for model in 'blip2'
    do
        for algo in 'pitome' 'tome' 'dct' 'tofu' 
        do
            # for ratio in '0.875' '0.9' '0.925' '0.95' '0.975'
            for ratio in '0.925' '0.95' '0.9625' '0.975'
            do
            echo "running $model $size $algo $ratio."
            sh eval_scripts/eval_itr_$model.sh $dataset $algo $ratio
            done
        done
        sh eval_scripts/eval_itr_$model.sh $dataset none 1.0 
    done
done