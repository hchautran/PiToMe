
# for input_size in 224 
# do
#     for model_size in  'S'  # for deit
#     # for model_size in 'L' 'H' #for mae
#     do
#         # for algo in  'pitome' 
#         for ratio in '0.9' '0.925' '0.95' '0.975'  
#         do
#             for algo in 'pitome' 'tome' 
#             # for algo in 'pitome' 
#             do
#             # Indented for loop body:
#             echo "running DEIT $model_size $algo."
#             sh scripts/eval_scripts/eval_ic.sh DEIT $model_size $input_size $ratio $algo  
#             done
#         done
#     done
# done



for input_size in 224 
do
    for model_size in 'B' 'L' 'H' #for mae
    do
        # for algo in  'pitome' 
        for ratio in '0.9' '0.925' '0.95' '0.975'  
        do
            # for algo in 'tome' 'pitome' 'dct' 'tofu' 'diffrate' 
            for algo in 'pitome' 'tome'
            # for algo in pitome 
            do
            # Indented for loop body:
            echo "running MAE $model_size $algo."
            sh scripts/eval_scripts/eval_ic.sh MAE $model_size $input_size $ratio $algo  
            done
        done
    done
done