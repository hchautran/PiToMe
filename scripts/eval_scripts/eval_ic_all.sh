
# Loop through each element in the array
for model in 'DEIT'
do 
    for input_size in 224 384
    do
        for model_size in 'T' 'S' 'B'
        # for model size in 'B' 'L' 'H'
        do
            for algo in  'pitome' 
            do
                for ratio in '0.925' 
                do
                # Indented for loop body:
                echo "running $model $model_size $algo."
                sh scripts/eval_scripts/eval_ic.sh $model $model_size $input_size $ratio $algo  
                done
            done
        done
    done
done