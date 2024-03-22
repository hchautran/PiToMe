
# Loop through each element in the array
for dataset in 'imdb' 
do
    for model in 'bert-base-uncased' 'distilbert-base-uncased' 'bert-large-uncased'
    do 
        for algo in 'tome' 'pitome' 'tofu' 'diffrate' 'dct'
        do
            for ratio in '0.70' '0.6' '0.55'
            do
            echo "running $model $size $algo $ratio."
            sh train_scripts/train_tc.sh $model $algo $ratio
            done
        done
    done
    # sh eval_scripts/eval_tc.sh none 1.0 $dataset
done