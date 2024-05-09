
# export CKT_SIZE='vit_base_patch16_mae'
# export CKT_SIZE=''
# export CKT_SIZE='vit_huge_patch14_mae'

# Loop through each element in the array
for model in 'vit_huge_patch14_mae' 
do
    # for algo in 'diffrate' 
    # for algo in 'tome' 'tofu' 'dct' 'diffrate' 'pitome' 
    for algo in 'tome' 'tofu' 'diffrate' 'pitome' 
    do
        for ratio in '0.875' '0.9' '0.9125' '0.925' '0.95' 
        # for ratio in '0.9685'
        do
        # Indented for loop body:
        echo "running $model $algo $ratio."
        sh eval_scripts/eval_mae.sh $model $ratio $algo  # Execute eval_ic.sh with arguments
        done
    done
    sh eval_scripts/eval_mae.sh $model 1.0 none # Execute eval_ic.sh with arguments
done