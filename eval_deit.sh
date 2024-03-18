
# Loop through each element in the array
for size in 'tiny' 'small' 'base'
do
    for algo in 'pitome' 'tome' 'dct' 'tofu' 
    do
        for ratio in '0.875' '0.9' '0.9125' '0.925' '0.95' 
        do
        # Indented for loop body:
        echo "running deit $size $algo."
        sh eval_scripts/eval_ic.sh deit $size 224 $ratio $algo  # Execute eval_ic.sh with arguments
        done
    done
    sh eval_scripts/eval_ic.sh deit $size 224 1.0 none  # Execute eval_ic.sh with arguments
done