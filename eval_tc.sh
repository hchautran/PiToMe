
# Loop through each element in the array
for dataset in 'imdb' 
do
    for algo in 'pitome' 'tome' 'dct' 'tofu' 
    do
        for ratio in '0.875' '0.9' '0.925' '0.95' '0.975'
        do
        echo "running $size $algo $ratio."
        sh eval_scripts/eval_tc.sh $dataset $algo $ratio
        done
    done
    sh eval_scripts/eval_tc.sh $dataset none 1.0 
done