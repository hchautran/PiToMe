
# Loop through each element in the array
for dataset in 'sst2' 
do
    for model in  'bert-base-uncased'  
    do 
        # for algo in 'dct' 'pitome' 'tome' 'diffrate' 'tofu'
        for algo in 'pitome'  'tome' 
        # for algo in  'pitome' 'tome' 'mctf' '
        do
            for ratio in '0.505' '0.525' '0.55' '0.6' '0.625' '0.7' '0.75' '0.8' '0.85'
            do
            echo "running $size $algo $ratio."
            sh scripts/eval_scripts/eval_tc.sh $algo $ratio $dataset $model
            done
        done
        sh scripts/eval_scripts/eval_tc.sh none 1.0 $dataset $model
    done 
done