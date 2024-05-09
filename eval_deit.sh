
# Loop through each element in the array
for size in  'tiny'
do
    # for algo in 'diffrate' 
    for algo in 'diffrate' 
    # for algo in 'diffrate' 
    do
        for ratio in '0.925'  
        do
        # Indented for loop body:
        echo "running deit $size $algo."
        sh eval_scripts/eval_deit.sh deit $size 224 $ratio $algo  # Execute eval_ic.sh with arguments
        done
    done
    sh eval_scripts/eval_deit.sh deit $size 224 1.0 none  # Execute eval_ic.sh with arguments
done