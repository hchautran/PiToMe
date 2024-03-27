for size in 'base' 'large' 'huge'
do
    for algo in  'pitome' 'tome' 'dct' 'tofu' 'none' 
    do
        sh train_scripts/train_ic.sh mae $size 224 0.9125 $algo 
    done
done