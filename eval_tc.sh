
# Loop through each element in the array
for dataset in 'sst2' 'imdb' 
do
    for model in  'bert-base-uncased' 'distilbert-base-uncased'
    do 
        # for algo in 'dct' 'pitome' 'tome' 'diffrate' 'tofu'
        # for algo in 'dct' 'pitome' 'tome' 'diffrate' 'tofu'
        # for algo in  'pitome' 'tome' 'tofu' 'dct' 'diffrate' 
        # do
        #     for ratio in '0.55' '0.6' '0.7' '0.8' '0.85' '0.9'
        #     do
        #     echo "running $size $algo $ratio."
        #     sh eval_scripts/eval_tc.sh $algo $ratio $dataset $model
        #     done
        # done
        sh eval_scripts/eval_tc.sh none 1.0 $dataset $model
    done 
done