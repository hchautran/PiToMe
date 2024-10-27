
# Loop through each element in the array
# for dataset in 'coco' 
for dataset in 'flickr' 'coco' 
do
    for model in 'blip2' 'blip' 'albef' 
    do
        for algo in  pitome tome 
        do
            for ratio in  '0.9' '0.925' '0.95' '0.975' 
            do
                echo "running $model $size $algo $ratio."
                sh scripts/eval_scripts/eval_itr.sh $dataset $model $algo $ratio
            done
        done
    done
done
