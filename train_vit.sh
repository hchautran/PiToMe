# for size in 'small' 'base' 'tiny'
# do
    # for algo in  'pitome' 'tome' 'dct' 'tofu' 'none' 
    # do
sh train_scripts/train_vit.sh vit $1 384 0.925 tome 2 &  
sh train_scripts/train_vit.sh vit $1 384 0.925 tofu 3 &  
sh train_scripts/train_vit.sh vit $1 384 0.925 dct 4 &  
sh train_scripts/train_vit.sh vit $1 384 0.925 none 5 &  
sh train_scripts/train_vit.sh vit $1 384 0.925 pitome 1   
    # done
# done