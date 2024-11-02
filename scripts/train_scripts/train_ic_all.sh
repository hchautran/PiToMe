SIZE=S

# sh scripts/train_scripts/train_ic.sh DEIT $SIZE 224 1.0 none 5 &  
# sh scripts/train_scripts/train_ic.sh DEIT $SIZE  224 0.925 dct 2 &  
# sh scripts/train_scripts/train_ic.sh DEIT $SIZE  224 0.925 tofu 3 &  
sh scripts/train_scripts/train_ic.sh DEIT $SIZE  224 0.925 tome 4   
# sh scripts/train_scripts/train_ic.sh DEIT $SIZE 224 0.925 pitome 1 