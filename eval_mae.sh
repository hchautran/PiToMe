
# export CKT_SIZE='vit_base_patch16_mae'
# export CKT_SIZE=''
# export CKT_SIZE='vit_huge_patch14_mae'

# Loop through each element in the array
# for model in 'vit_huge_patch14_mae' 'vit_large_patch16_mae' 'vit_base_patch16_base'
for model in 'vit_base_patch16_mae'
do
    # for algo in 'diffrate' 
    # for algo in 'tome' 'tofu' 'dct' 'diffrate' 'pitome' 
    sh eval_scripts/eval_mae.sh $model 1.0 baseline # Execute eval_ic.sh with arguments
    for algo in 'tome' 'tofu' 'diffrate' 'pitome' 
    do
        for ratio in '0.9' '0.925' '0.95' '0.975' 
        # for ratio in '0.9685'
        do
        # Indented for loop body:
        echo "running $model $algo $ratio."
        sh eval_scripts/eval_mae.sh $model $ratio $algo  # Execute eval_ic.sh with arguments
        done
    done
done