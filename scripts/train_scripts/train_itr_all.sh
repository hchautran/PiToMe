
# Loop through each element in the array
# for dataset in 'coco' 
for dataset in 'flickr' 'coco'
do
    for model in  'clip' 'blip' 'albef' 'blip2'
    do
        for algo in 'pitome' 'tome' 'tofu' 'dct' 
        do
            # for ratio in '0.925' 
            for ratio in '0.925' 
            do
                echo "running $model $size $algo $ratio."
                sh scripts/train_scripts/train_itr.sh  $dataset $model $algo $ratio
            done
        done
        # sh train_scripts/train_itr_$model.sh $dataset none 1.0 
    done
done