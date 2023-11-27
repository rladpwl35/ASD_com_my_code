python pretrain_joao.py --gamma 0.001 --emb_dim 128 --device 3 --gnn_type gat --root_unsupervised spars20_unsupervised/processed/only_cc --aug_mode 'none' --aug1 1 --aug2 4 --num_workers 1 --num_layer 1

python pretrain_joao.py --gamma 0.1 --emb_dim 128 --device 3 --gnn_type gat --root_unsupervised spars20_unsupervised/processed/only_cc --aug_mode 'none' --aug1 2 --aug2 1 --num_workers 1 --num_layer 1

python pretrain_joao.py --gamma 0.1 --emb_dim 128 --device 3 --gnn_type gat --root_unsupervised spars20_unsupervised/processed/only_cc --aug_mode 'none' --aug1 2 --aug2 2 --num_workers 1 --num_layer 1

python pretrain_joao.py --gamma 0.001 --emb_dim 128 --device 3 --gnn_type gat --root_unsupervised spars20_unsupervised/processed/only_cc --aug_mode 'none' --aug1 2 --aug2 0 --num_workers 1 --num_layer 1

python pretrain_joao.py --gamma 0.001 --emb_dim 128 --device 3 --gnn_type gat --root_unsupervised spars20_unsupervised/processed/only_cc --aug_mode 'none' --aug1 2 --aug2 1 --num_workers 1 --num_layer 1
