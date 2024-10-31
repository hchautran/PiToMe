
# Loop through each element in the array
for dataset in 'imdb' 'sst2' 'rotten' 
do
    for model in   'bert-base-uncased' 'distilbert-base-uncased'
    do 
        for algo in  'mctf' 'tome' 'tofu' 'pitome' 'crossget' 'dct'
        # for algo in  'tofu'  'crossget'
        do
            for ratio in '0.505' '0.525' '0.55' '0.6' '0.625' '0.7' '0.75' '0.8' '0.85' '1.0'
            do
            echo "running $size $algo $ratio."
            sh scripts/eval_scripts/eval_tc.sh $algo $ratio $dataset $model
            done
        done
    done 
done