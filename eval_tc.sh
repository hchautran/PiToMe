
# Loop through each element in the array
for dataset in  'imdb' 
do
    # for model in  'distilbert-base-uncased'
    for model in  'bert-base-uncased'
    do 
        for algo in 'pitome' 'tofu' 'tome' 'dct'
        do
            for ratio in '0.525' '0.55' '0.6' '0.65' '0.7' '.75'
            do
            echo "running $size $algo $ratio."
            sh eval_scripts/eval_tc.sh $algo $ratio $dataset $model
            done
        done
    done
    sh eval_scripts/eval_tc.sh none 1.0 $dataset $model
done