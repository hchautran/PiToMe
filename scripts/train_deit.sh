# for size in 'small' 'base' 'tiny'
# do
    # for algo in  'pitome' 'tome' 'dct' 'tofu' 'none' 
#     # do
# sh train_scripts/train_deit.sh deit $1 224 1.0 none 5 &  
# sh train_scripts/train_deit.sh deit $1 224 0.925 dct 2 &  
# sh train_scripts/train_deit.sh deit $1 224 0.925 tofu 3 &  
# sh train_scripts/train_deit.sh deit $1 224 0.925 tome 4 &  
sh train_scripts/train_deit.sh deit $1 224 0.925 pitome 1 
# sh train_scripts/train_deit.sh deit $1 224 0.925 diffrate 0 
    # done
# done