
# Loop through each element in the array
for dataset in 'flickr' 'coco'
    do
    for model in 'clip' 'blip' 'blip2'
    do
        for algo in 'pitome' 'tome' 'dct' 'tofu' 
        do
            for ratio in '0.875' '0.9' '0.925' '0.95' '0.975'
            do
            # Indented for loop body:
            echo "running $model $size $algo $ratio."
            sh eval_scripts/eval_itr_$model.sh $dataset $algo $ratio
            done
        done
        sh eval_scripts/eval_ic.sh deit $size 224 1.0 none  # Execute eval_ic.sh with arguments
    done
done