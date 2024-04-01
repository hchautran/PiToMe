# for size in 'small' 'base' 'tiny'
# do
    # for algo in  'pitome' 'tome' 'dct' 'tofu' 'none' 
#     # do
sh train_scripts/train_vit.sh vit $1 $2 0.8 tome 0 &  
sh train_scripts/train_vit.sh vit $1 $2 0.8 tofu 1 &  
sh train_scripts/train_vit.sh vit $1 $2 0.8 dct 2 &  
sh train_scripts/train_vit.sh vit $1 $2 0.8 pitome 4   
# sh train_scripts/train_vit.sh vit $1 224 0.8 none 3   
    # done
# done