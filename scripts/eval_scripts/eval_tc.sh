python main_tc.py --algo $1 --ratio $2 --task $3 --model $4 --eval
for alpha in 0.5 0.0 
do
   python main_tc.py --algo $1 --ratio $2 --task $3 --model $4 --eval --alpha $alpha
done