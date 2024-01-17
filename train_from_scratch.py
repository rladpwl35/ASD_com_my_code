### Train from scratch for ASD classification ###

import argparse 

from loader import BioDataset
from torch.utils.data.dataloader import DataLoader
from dataloader import DataLoaderFinetune
from splitters import random_split, species_split
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

import pandas as pd

import os
import pickle
#from pretrain_greverative_joao import graphcl
from pretrain_joao import graphcl

torch.autograd.set_detect_anomaly(True)

criterion = nn.BCEWithLogitsLoss()

def train(args, model, device, loader, optimizer, num_tasks):
    with torch.autograd.detect_anomaly():
        model.train()
    
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            pred = model(batch)
            y = F.one_hot(batch.y, num_classes=num_tasks)
            y = y.view(pred.shape).to(torch.float64)
        
            optimizer.zero_grad()
            loss = criterion(pred.double(), y.double())
            loss.backward()
        
            optimizer.step()
        
def eval(args, model, device, loader, num_tasks):
    model.eval()
    y_true = []
    y_scores = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        
        y = F.one_hot(batch.y, num_classes=num_tasks)    
        y_true.append(y.view(pred.shape).detach().cpu())
        y_scores.append(pred.detach().cpu())
        
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_scores = torch.cat(y_scores, dim = 0).numpy()
    
    
    
    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            roc_list.append(roc_auc_score(y_true[:,i], y_scores[:,i]))
            print("y_true", y_true[:,i])
            print("y_score", y_scores[:,i])
        else:
            roc_list.append(np.nan)
            
    print("roc_list :", roc_list)
            
    return np.array(roc_list) #y_true.shape[1]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--split', type=str, default = "random", help='Random or species split')
    parser.add_argument('--resultFile_name', type=str, default='')
    parser.add_argument('--num_tasks', type=int, default='2')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.runseed)
        
    
    dataset = torch.load('dataset/my_supervised/processed/geometric_data_processed.pt')
    
    y = []
    for i in range(len(dataset)):
        y.append(int(dataset[i].y))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
    skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=666)
    x_idx=range(len(dataset))
    
    train_val_dataset=[]
    train_dataset=[]
    val_dataset = []
    test_dataset=[]

    for train_val_index, test_index in skf.split(x_idx, y):
        tmp=0
    
    for j in range(len(train_val_index)):
        train_val_dataset.append(dataset[train_val_index[j]])
    for j in range(len(test_index)):
        test_dataset.append(dataset[test_index[j]])


    train_y = []
    train_x_idx = range(len(train_val_dataset))
    for i in range(len(train_val_dataset)):
        train_y.append(int(train_val_dataset[i].y))    

    for train_index, val_index in skf2.split(train_x_idx, train_y):
        tmp=0
    
    for j in range(len(train_index)):
        train_dataset.append(dataset[train_index[j]])
    for j in range(len(val_index)):
        val_dataset.append(dataset[val_index[j]])   
        

    #train_loader = DataLoaderFinetune(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    #val_loader = DataLoaderFinetune(val_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
    #test_loader = DataLoaderFinetune(test_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
    train_loader = DataLoaderFinetune(train_dataset, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoaderFinetune(val_dataset, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoaderFinetune(test_dataset,  shuffle=False, num_workers = args.num_workers)
    
    
    num_tasks = args.num_tasks
    
    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type = args.gnn_type)
    
    model_graphcl = graphcl(model.gnn)
    model.gnn = model_graphcl.gnn
    
    model.to(device)
    
    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    train_acc_list = []
    val_acc_list = []
    test_acc_list =  []
    
    for epoch in range(1, args.epochs+1):
        print("=====epoch "+ str(epoch))
        train(args, model, device, train_loader, optimizer, num_tasks=args.num_tasks)
        
        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader, num_tasks=args.num_tasks)
        else:
            train_acc = 0
            print("ommitting training evaluation")
        val_acc = eval(args, model, device, val_loader, num_tasks=args.num_tasks)
        
        val_acc_list.append(np.mean(val_acc))
        train_acc_list.append(train_acc)
        
        test_acc = eval(args, model, device, test_loader, num_tasks=args.num_tasks)
        test_acc_list.append(test_acc)
        
        print("")
        
        #print("train_acc: ",train_acc_list)
        #print("valication acc : ", val_acc_list)
        #print("test_acc : ", test_acc_list)
        
    with open('./results/'+args.resultFile_name+'.res', 'a+') as f:
        f.write(str(args.runseed) + ' ' + str(np.array(val_acc_list).max()) + ' ' + str(np.array(test_acc_list)[np.array(val_acc_list).argmax()]))
        f.write('\n')
        
if __name__ == "__main__":
    main()