
# Loop through each element in the array
# for dataset in 'coco' 
for dataset in 'flickr' 'coco'
do
    # for model in  'clip' 'blip' 'albef' 'blip2'
    for model in  'blip2' 
    # for model in 'blip2' 
    do
        sh train_scripts/train_itr_$model.sh $dataset none 1.0 
        for algo in 'pitome' 'tome' 'tofu' 'dct' 
        do
            # for ratio in '0.925' 
            for ratio in '0.925' 
            do
                echo "running $model $size $algo $ratio."
                sh train_scripts/train_itr_$model.sh $dataset $algo $ratio
            done
        done
    done
done