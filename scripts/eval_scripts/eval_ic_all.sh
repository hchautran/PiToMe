
# Loop through each element in the array
for model in 'DEIT' 'MAE' 
do 
    for input_size in 224 
    do
        for model_size in 'T' 'S' 'B' # for deit
        # for model_size in 'L' 'H' #for mae
        do
            # for algo in  'pitome' 
            for ratio in '0.9' '0.925' '0.95' '0.975'  
            do
                # for algo in 'tome' 'pitome' 'dct' 'tofu' 'diffrate' 
                # for algo in 'tome' 'pitome'
                for algo in pitome 
                do
                # Indented for loop body:
                echo "running $model $model_size $algo."
                sh scripts/eval_scripts/eval_ic.sh $model $model_size $input_size $ratio $algo  
                done
            done
        done
    done
done