
# Loop through each element in the array
# for dataset in 'coco' 
for dataset in 'flickr' 'coco'
do
    # for model in  'clip' 'blip' 'albef' 'blip2'
    for model in   'clip' 'albef' 
    # for model in 'blip2' 
    do
        for algo in 'pitome' 'tome' 'tofu' 'dct' 
        do
            # for ratio in '0.925' 
            for ratio in '0.925' 
            do
                echo "running $model $size $algo $ratio."
                if [ "$model" = "blip" || "$model" = "albef" ]; then
                    sh train_scripts/train_itr_blip.sh $dataset $algo $ratio
                else  
                    sh train_scripts/train_itr_$model.sh $dataset $algo $ratio
                fi
            done
        done
        if [ "$model" = "blip" || "$model" = "albef" ]; then
            sh train_scripts/train_itr_blip.sh $dataset none 1.0 
        else  
            sh train_scripts/train_itr_$model.sh $dataset none 1.0 
        fi
    done
done