#### gcn fine-tuning
#split=species
split = random

model_file=$1
lr=$2
resultFile_name=$3

### for gcn
for runseed in 0 1 2 3 4 5 6 7 8 9
do
python finetune.py --model_file $model_file --split $split --epochs 50 --device 7 --runseed $runseed --gnn_type gcn --lr $lr --resultFile_name $resultFile_name
done
