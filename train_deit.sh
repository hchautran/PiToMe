# for size in 'small' 'base' 'tiny'
# do
    # for algo in  'pitome' 'tome' 'dct' 'tofu' 'none' 
    # do
# sh train_scripts/train_deit.sh deit $1 224 0.9 tome 2 &  
# sh train_scripts/train_deit.sh deit $1 224 0.9 tofu 3 &  
# sh train_scripts/train_deit.sh deit $1 224 0.9 dct 4 &  
# sh train_scripts/train_deit.sh deit $1 224 0.9 none 5 &  
sh train_scripts/train_deit.sh deit $1 224 0.9 pitome 1 
    # done
# done