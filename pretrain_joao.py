import argparse

from loader import BioDataset_graphcl, DataLoader
#from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

import pandas as pd


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

#grapn CL 모델 
class graphcl(nn.Module):
    def __init__(self, gnn, emb_dim):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(emb_dim, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        #print('x : ',x)
        #print('x.shape: ', x.shape)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2): 
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        
        pos_sim = sim_matrix[np.arange(batch_size), np.arange(batch_size)]
        
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        
        loss = - torch.log(loss).mean()
        return loss

#학습 함수 (tqdm 진행률 나타내주는 함수)
def train(args, loader, model, optimizer, device, gamma_joao):
    model.train()

    train_loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    # for step, batch in enumerate(loader, desc="Iteration"):
        _, batch1, batch2 = batch
        #print("batch1", batch1.edge_index)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        loss = model.loss_cl(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        #print(train_loss_accum/(step+1))

    # joao
    aug_prob = loader.dataset.aug_prob
    loss_aug = np.zeros(25)
    for n in range(25):
        _aug_prob = np.zeros(25)
        _aug_prob[n] = 1
        loader.dataset.set_augProb(_aug_prob)

        count, count_stop = 0, len(loader.dataset)//(loader.batch_size)+1 # for efficiency, we only use around 10% of data to estimate the loss
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            # for step, batch in enumerate(loader, desc="Iteration"):
                _, batch1, batch2 = batch
                batch1 = batch1.to(device)
                batch2 = batch2.to(device)

                x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
                x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
                loss = model.loss_cl(x1, x2)

                loss_aug[n] += loss.item()
                count += 1
                if count == count_stop:
                    break
        loss_aug[n] /= count
        

    # view selection, projected gradient descent, reference: https://arxiv.org/abs/1906.03563
    beta = 1
    gamma = gamma_joao

    b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1/25))
    mu_min, mu_max = b.min()-1/25, b.max()-1/25
    mu = (mu_min + mu_max) / 2

    # bisection method
    while abs(np.maximum(b-mu, 0).sum() - 1) > 1e-2:
        if np.maximum(b-mu, 0).sum() > 1:
            mu_min = mu
        else:
            mu_max = mu
        mu = (mu_min + mu_max) / 2

    aug_prob = np.maximum(b-mu, 0)
    aug_prob /= aug_prob.sum()
    print('aug_prob : ', aug_prob)
    print("what is return value : ", train_loss_accum/(step+1))
    print("Device is : ", device)
    return train_loss_accum/(step+1), aug_prob


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--aug_mode', type=str, default = 'sample') 
    parser.add_argument('--aug1', type=str, default = 0) 
    parser.add_argument('--aug2', type=str, default = 0)
    parser.add_argument('--aug_strength', type=float, default = 0.2)
    parser.add_argument('--root_unsupervised', type=str, default = 'my_data')
    

    parser.add_argument('--gamma', type=float, default = 0.1)
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    root_unsupervised = 'dataset/' + args.root_unsupervised
    dataset = BioDataset_graphcl(root_unsupervised, data_type='unsupervised')
    dataset.set_augMode(args.aug_mode)
    if args.aug_mode == 'none':
        dataset.set_aug(args.aug1, args.aug2)
    dataset.set_augStrength(args.aug_strength)
    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    #for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        #print(batch)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    model = graphcl(gnn, args.emb_dim)
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    aug_prob = np.ones(25) / 25
    print(aug_prob)
    
    
    pretrain_loss_list = []
    for epoch in range(1, args.epochs+1):
        
        dataset.set_augProb(aug_prob)
        pretrain_loss, aug_prob = train(args, loader, model, optimizer, device, args.gamma)
        
        pretrain_loss_list.append(pretrain_loss)
        
        print(epoch, pretrain_loss, aug_prob)
        
        
        if epoch % 20 == 0:
            if args.aug_mode == "sample":    
                if 'abide' in args.root_unsupervised:
                    if 'cc200' in args.root_unsupervised:
                        unsupervised_path = str(args.root_unsupervised).strip("/abide/only_cc_roi200")
                        if args.num_layer == 1:
                            torch.save(model.gnn.state_dict(), "./weights/abide/layer1/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim)+'_cc200' + ".pth")
                        elif  args.num_layer == 2:
                            torch.save(model.gnn.state_dict(), "./weights/abide/layer2/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim)+'_cc200' + ".pth")
                    else:
                        unsupervised_path = str(args.root_unsupervised).strip("/abide/only_cc_roi200")
                        if args.num_layer == 1:
                            torch.save(model.gnn.state_dict(), "./weights/abide/layer1/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim)+'_cc200' + ".pth")
                        elif  args.num_layer == 2:
                            torch.save(model.gnn.state_dict(), "./weights/abide/layer2/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim)+'_cc200' + ".pth")
                elif 'Coordinate' in args.root_unsupervised:
                    unsupervised_path = str(args.root_unsupervised).strip("/processed_ROICoordinate")
                    torch.save(model.gnn.state_dict(), "./weights/coordinate/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                elif 'cc' in args.root_unsupervised:
                    unsupervised_path = str(args.root_unsupervised).strip('/processed/only_cc')
                    if args.num_layer == 1:
                        torch.save(model.gnn.state_dict(), "./weights/only_cc/layer1/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    elif  args.num_layer == 2:
                        torch.save(model.gnn.state_dict(), "./weights/only_cc/layer2/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    elif  args.num_layer == 3:
                        torch.save(model.gnn.state_dict(), "./weights/only_cc/layer3/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    elif  args.num_layer == 4:
                        torch.save(model.gnn.state_dict(), "./weights/only_cc/layer4/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    else:
                        torch.save(model.gnn.state_dict(), "./weights/only_cc/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                else:
                    unsupervised_path = str(args.root_unsupervised).strip("/abide/")
                    
                    if args.num_layer == 1:
                        torch.save(model.gnn.state_dict(), "./weights/layer1/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    elif  args.num_layer == 2:
                        torch.save(model.gnn.state_dict(), "./weights/layer2/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    elif  args.num_layer == 3:
                        torch.save(model.gnn.state_dict(), "./weights/layer3/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    elif  args.num_layer == 4:
                        torch.save(model.gnn.state_dict(), "./weights/layer4/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    else:
                        torch.save(model.gnn.state_dict(), "./weights/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
            if args.aug_mode == 'none':
                if 'abide' in args.root_unsupervised:
                    unsupervised_path = str(args.root_unsupervised).strip("/abide/")
                    torch.save(model.gnn.state_dict(), "./weights/manual/abide/joao_" + str(args.aug_mode)+ str(args.aug1)+str(args.aug2) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                if 'Coordinate' in args.root_unsupervised:
                    unsupervised_path = str(args.root_unsupervised).strip("/processed_ROICoordinate")
                    torch.save(model.gnn.state_dict(), "./weights/manual/coordinate/joao_" + str(args.aug_mode)+ str(args.aug1)+str(args.aug2) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                if 'cc' in args.root_unsupervised:
                    unsupervised_path = str(args.root_unsupervised).strip('/processed/only_cc')
                    if args.num_layer == 1:
                        torch.save(model.gnn.state_dict(), "./weights/manual/only_cc/layer1/joao_" + str(args.aug_mode)+ str(args.aug1) +str(args.aug2)+ '_'+ str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    elif  args.num_layer == 2:
                        torch.save(model.gnn.state_dict(), "./weights/manual/only_cc/layer2/joao_" + str(args.aug_mode)+ str(args.aug1) +str(args.aug2) + '_'+ str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    elif  args.num_layer == 3:
                        torch.save(model.gnn.state_dict(), "./weights/manual/only_cc/layer3/joao_" + str(args.aug_mode)+ str(args.aug1) +str(args.aug2) + '_'+ str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    elif  args.num_layer == 4:
                        torch.save(model.gnn.state_dict(), "./weights/manual/only_cc/layer4/joao_" + str(args.aug_mode)+ str(args.aug1) +str(args.aug2 + '_') + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                    else:
                        torch.save(model.gnn.state_dict(), "./weights/manual/only_cc/joao_" + str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
                else:
                    unsupervised_path = str(args.root_unsupervised)
                    torch.save(model.gnn.state_dict(), "./weights/manual/layer2/joao_" + str(args.aug_mode)+ str(args.aug1)+str(args.aug2) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) + ".pth")
    plt.plot(np.array(pretrain_loss_list), 'r',label='Pretrain Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if args.aug_mode == "none":
        if 'Coordinate' in args.root_unsupervised:
            plt.savefig("./pretrain_loss/manual/coordinate/joao_"+ str(args.aug_mode)+ str(args.aug1)+str(args.aug2)+'_'+str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) +'.png')
        else:
            plt.savefig("./pretrain_loss/manual/roi/joao_"+ str(args.aug_mode)+ str(args.aug1)+str(args.aug2)+'_'+str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) +'.png')
    else:
        if args.num_layer == 1:
            plt.savefig("./pretrain_loss/layer1/joao_"+ str(args.aug_mode)+ str(args.aug1)+str(args.aug2)+'_'+str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) +'.png')
        elif  args.num_layer == 2:
            plt.savefig("./pretrain_loss/layer2/joao_"+ str(args.aug_mode)+ str(args.aug1)+str(args.aug2)+'_'+str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) +'.png')
        elif  args.num_layer == 3:
            plt.savefig("./pretrain_loss/layer3/joao_"+ str(args.aug_mode)+ str(args.aug1)+str(args.aug2)+'_'+str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) +'.png')
        elif  args.num_layer == 4:
            plt.savefig("./pretrain_loss/layer4/joao_"+ str(args.aug_mode)+ str(args.aug1)+str(args.aug2)+'_'+str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) +'.png')
        else:
            plt.savefig("./pretrain_loss/joao_"+ str(args.aug_mode)+ str(args.aug1)+str(args.aug2)+'_'+str(args.gamma) + '_' + str(args.gnn_type) + '_' + 'batch' + str(args.batch_size) + '_' + unsupervised_path + '_' + str(epoch) + 'emb' + str(args.emb_dim) +'.png')
    
if __name__ == "__main__":
    main()

