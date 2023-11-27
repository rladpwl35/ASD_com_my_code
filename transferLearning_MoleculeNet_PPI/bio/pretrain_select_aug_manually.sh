#### pretrain graphcl selecting augmentation manually

device=$1
gnn_type=$2
root_unsupervised=$3
emb_dim=$4
num_layer=$5
gamma=$6

### for gcn
aug1=0
for aug2 in 0 1 2 3 4
do
	echo 'aug1: '$aug1, 'aug2: '$aug2
	python pretrain_joao.py --gamma $gamma --emb_dim $emb_dim --device $device --gnn_type $gnn_type --root_unsupervised \
		$root_unsupervised --aug_mode 'none' --aug1 $aug1 --aug2 $aug2 --num_workers 1 --num_layer $num_layer
done
