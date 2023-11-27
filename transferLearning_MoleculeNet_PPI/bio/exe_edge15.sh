
split=random

model_file=$1
lr=$2
resultFile_dir=$3
resultFile_name=$4
device=$5
batch_size=$6
epochs=$7
dataset_dir=$8
decay=$9
gnn_type=$10


### 

aug1='4'

for aug2 in 1 2 3 4
do
echo 'aug1: '$aug1, 'aug2: '$aug2
for runseed in 0 1 2 3 4 5 6 7 8 9
do
python my_finetune.py --eval_train 1 --batch_size $batch_size --model_file weights/manual/coordinate/joao_none$aug1$aug2$model_file --split $split --epochs $epochs --device $device \
--runseed $runseed --gnn_type $gnn_type --lr $lr --resultFile_name $resultFile_dir$aug1$aug2'_'$resultFile_name --dataset_dir $dataset_dir --decay $decay --num_workers 1 
done
done
