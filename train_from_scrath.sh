#### gcn fine-tuning
#split=species
split=random

batch_size=$1
lr=$2
resultFile_name=$3
device=$4

### for gcn
for runseed in 0 1 2 3 4 5 6 7 8 9
do
python train_from_scratch.py --batch_size $batch_size --split $split --epochs 50 --device $device --runseed $runseed --gnn_type gcn --lr $lr --resultFile_name $resultFile_name
done




