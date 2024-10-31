SIZE=S

sh train_scripts/train_ic.sh DEIT $SIZE 224 1.0 none 5 &  
sh train_scripts/train_ic.sh DEIT $SIZE  224 0.925 dct 2 &  
sh train_scripts/train_ic.sh DEIT $SIZE  224 0.925 tofu 3 &  
sh train_scripts/train_ic.sh DEIT $SIZE  224 0.925 tome 4 &  
sh scripts/train_scripts/train_ic.sh DEIT $SIZE 224 0.925 pitome 1 