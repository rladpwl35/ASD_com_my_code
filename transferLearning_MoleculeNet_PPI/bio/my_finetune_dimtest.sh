#### gcn fine-tuning
#split=species
split=random

model_file=$1
lr=$2
resultFile_name=$3
device=$4
batch_size=$5
epochs=$6
dataset_dir=$7
decay=$8
gnn_type=$9
kfold=${10}

### for gcn
for runseed in 0 1 2 3 4 5 6 7 8 9
do
python my_finetune_dimtest.py --eval_train 1 --batch_size $batch_size --model_file $model_file --split $split --epochs $epochs --device $device \
--runseed $runseed --gnn_type $gnn_type --lr $lr --resultFile_name $resultFile_name --dataset_dir $dataset_dir --decay $decay --num_workers 1 \
--kfold $kfold
done