
algo=$1
# Loop through each element in the array
for dataset in  'sst2' 'imdb'  
do
    for model in 'bert-base-uncased' 'distilbert-base-uncased' 
    do 
        if [ "$algo" = "none" ];
        then
            sh train_scripts/train_tc.sh $model none 1.0 $dataset $2
        else
            for ratio in '0.8' '0.75' '0.70' 
            do
                echo "running $model $size $algo $ratio."
                sh train_scripts/train_tc.sh $model $algo $ratio $dataset  $2 
            done
        fi
    done
done