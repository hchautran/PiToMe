
# Loop through each element in the array
for size in 'tiny' 'small' 'base'
do
    # for algo in 'diffrate' 
    for algo in 'tome' 'pitome' 'dct' 'tofu' 'pitome' 
    do
        for ratio in '0.9' '0.925' '0.95' '0.975' 
        do
        # Indented for loop body:
        echo "running vit $size $algo."
        sh scripts/eval_scripts/eval_ic.sh deit $size 384 $ratio $algo  # Execute eval_ic.sh with arguments
        done
    done
    sh scripts/eval_scripts/eval_ic.sh deit $size 384 1.0 none  # Execute eval_ic.sh with arguments
done