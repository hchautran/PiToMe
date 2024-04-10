
# Loop through each element in the array
for dataset in 'sst2' 'imdb' 
do
    for model in  'distilbert-base-uncased' 'bert-base-uncased'
    do 
        for algo in 'dct' 'pitome' 'tome' 'diffrate' 'tofu'
        do
            for ratio in '0.85' '0.9' '0.95'
            do
            echo "running $size $algo $ratio."
            sh eval_scripts/eval_tc.sh $algo $ratio $dataset $model
            done
        done
        # sh eval_scripts/eval_tc.sh none 1.0 $dataset $model
    done 
done