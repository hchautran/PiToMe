for size in 'tiny' 'small' 'base'
do
    for algo in  'pitome' 'tome' 'dct' 'tofu' 'none' 
    do
        sh train_scripts/train_deit.sh deit $size 224 0.9125 $algo
    done
done