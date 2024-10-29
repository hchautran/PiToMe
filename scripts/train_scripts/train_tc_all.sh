
algo=$1
# Loop through each element in the array
for algo in 'pitome' 'tome' 'tofu' 'diffrate' 'dct'
do
    for dataset in  'imdb'  

    do
        for model in 'bert-base-uncased' 'distilbert-base-uncased' 
        do 
            if [ "$algo" = "none" ];
            then
                sh scripts/train_scripts/train_tc.sh $model none 1.0 $dataset 
            else
                for ratio in '0.8' '0.70' 
                do
                    echo "running $model $size $algo $ratio."
                    sh scripts/train_scripts/train_tc.sh $model $algo $ratio $dataset  
                done
            fi
        done
    done
done