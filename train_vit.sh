# for size in 'small' 'base' 'tiny'
# do
    # for algo in  'pitome' 'tome' 'dct' 'tofu' 'none' 
#     # do
# sh train_scripts/train_vit.sh vit $1 384 0.75 tome 0 &  
# sh train_scripts/train_vit.sh vit $1 384 0.75 tofu 1 &  
# sh train_scripts/train_vit.sh vit $1 384 0.75 dct 2 &  
# sh train_scripts/train_vit.sh vit $1 384 0.75 none 3   
sh train_scripts/train_vit.sh vit $1 384 0.75 pitome 4   
    # done
# done