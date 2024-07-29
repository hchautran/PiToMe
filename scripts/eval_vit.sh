
# Loop through each element in the array
for size in  'large'
do
    # for algo in 'diffrate' 
    # for algo in 'tome' 'pitome' 'dct' 'tofu' 
    for algo in 'pitome' 
    do
        for ratio in '0.925'  
        do
        # Indented for loop body:
        echo "running vit $size $algo."
        sh eval_scripts/eval_deit.sh vit $size 384 $ratio $algo  # Execute eval_ic.sh with arguments
        done
    done
    sh eval_scripts/eval_deit.sh vit $size 384 1.0 none  # Execute eval_ic.sh with arguments
done