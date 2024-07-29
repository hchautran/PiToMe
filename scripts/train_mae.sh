# for size in 'small' 'base' 'tiny'
# do
    # for algo in  'pitome' 'tome' 'dct' 'tofu' 'none' 
    # do
sh train_scripts/train_mae.sh mae $1 224 0.95 tome 2 &  
sh train_scripts/train_mae.sh mae $1 224 0.95 tofu 3 &  
sh train_scripts/train_mae.sh mae $1 224 0.95 dct 4 &  
# sh train_scripts/train_mae.sh mae $1 224 0.95 none 5 &  
sh train_scripts/train_mae.sh mae $1 224 0.95 pitome 1   
    # done
# done