
# Loop through each element in the array
for dataset in 'imdb' 
do
    for algo in 'pitome' 'tome' 'dct' 'tofu' 
    do
        for ratio in '0.525' '0.55' '0.6' '0.65' '0.7'
        do
        echo "running $size $algo $ratio."
        sh eval_scripts/eval_tc.sh $algo $ratio $dataset
        done
    done
    sh eval_scripts/eval_tc.sh none 1.0 $dataset
done