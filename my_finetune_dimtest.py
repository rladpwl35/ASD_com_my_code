### Finetuning for ASD classification ###


import argparse 

from loader import BioDataset
from torch.utils.data.dataloader import DataLoader
from dataloader import DataLoaderFinetune
from splitters import random_split, species_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model_dimtest import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, roc_curve

import pandas as pd

import os
import pickle
#from pretrain_greverative_joao import graphcl
from pretrain_joao import graphcl

torch.autograd.set_detect_anomaly(True)

criterion = nn.BCEWithLogitsLoss()

def train(args, model, device, loader, val_loader, optimizer, num_tasks):
    with torch.autograd.detect_anomaly():
        model.train()
        
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            #print("batch: ", batch)
            batch = batch.to(device)
            pred = model(batch)
            y = F.one_hot(batch.y, num_classes=num_tasks)
            y = y.view(pred.shape).to(torch.float64)
        
            optimizer.zero_grad()
            loss = criterion(pred.double(), y.double())
            loss.backward()
            
            
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        for i, data in enumerate(val_loader):
           
            data = data.to(device)
            val_pred = model(data)
            val_y = F.one_hot(data.y, num_classes=num_tasks)
            val_y = val_y.view(val_pred.shape).to(torch.float64)
        
            val_loss += criterion(val_pred.double(), val_y.double())
        val_loss = val_loss/len(val_loader)    
    return loss.item(), val_loss.item()
        
def eval(args, model, device, loader, num_tasks):
    model.eval()
    y_true = []
    y_scores = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        #print("batch: ", batch)
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        
        y = F.one_hot(batch.y, num_classes=num_tasks)    
        y_true.append(y.view(pred.shape).detach().cpu())
        y_scores.append(pred.detach().cpu())
        
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_scores = torch.cat(y_scores, dim = 0).numpy()
    #print("!!!y_true : ", y_true)
    
    #print("!!! y_true[:,i]", y_true[:,0])
    
    
    
    roc_list = []
    fper_list = []
    tper_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            roc_list.append(roc_auc_score(y_true[:,i], y_scores[:,i]))
            #print("y_true", y_true[:,i])
            #print("y_score", y_scores[:,i])
            fper, tper, _ = roc_curve(y_true[:,i], y_scores[:,i])
            #print("fper, tper : ", fper, ", ",tper)
            fper_list.append(fper)
            tper_list.append(tper)
        else:
            roc_list.append(np.nan)
            
    #print("roc_list :", roc_list)
            
    return np.array(roc_list),fper_list,tper_list  #y_true.shape[1]

def plot_roc_curve(fper, tper, label):
        plt.clf()
        plt.plot(fper, tper, color='red', label='ROC')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic Curve ({label})')
        plt.legend()
##배치 설정 안함.
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



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
    parser.add_argument('--dataset_dir', type=str, default='')
    parser.add_argument('--kfold', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.runseed)
        
    
    dataset = torch.load(args.dataset_dir)
    
    y = []
    for i in range(len(dataset)):
        y.append(int(dataset[i].y))
    
    if args.kfold == 0:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
        skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=666)
    else :
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=666)
        skf2 = StratifiedKFold(n_splits=args.kfold-1, shuffle=True, random_state=666)
        
    x_idx=range(len(dataset))
    
    kfold = 1
    for train_val_index, test_index in skf.split(x_idx, y):
        train_val_dataset=[]
        train_dataset=[]
        val_dataset = []
        test_dataset=[]
        print()
        print("The fold is ", kfold)
        print()
        if kfold == 2 and args.kfold == 0:
            break
        else: 
            for j in range(len(train_val_index)):
                train_val_dataset.append(dataset[train_val_index[j]])
            for j in range(len(test_index)):
                test_dataset.append(dataset[test_index[j]])


            train_y = []
            train_x_idx = range(len(train_val_dataset))
            for i in range(len(train_val_dataset)):
                train_y.append(int(train_val_dataset[i].y))    

            print("train_val_index : ", train_val_index)
            print("train_y : ", train_y)
            print("test_index : ", test_index)

            tmp = 0
            for train_index, val_index in skf2.split(train_x_idx, train_y):
                tmp=0

            for j in range(len(train_index)):
                train_dataset.append(dataset[train_index[j]])
            for j in range(len(val_index)):
                val_dataset.append(dataset[val_index[j]])   

            print("val_index : ", val_index)

            #print(train_val_dataset)
            #print(train_dataset)
            #print(val_dataset)
            #print(test_dataset)    

            #train_loader = DataLoaderFinetune(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
            #val_loader = DataLoaderFinetune(val_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
            #test_loader = DataLoaderFinetune(test_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)

            train_loader = DataLoaderFinetune(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
            val_loader = DataLoaderFinetune(val_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
            test_loader = DataLoaderFinetune(test_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)

            #print("start")
            #for data in test_loader:
            #    print("batch:", data.batch)
            #    print("x:", data.x)

            num_tasks = args.num_tasks

            #set up model

            if args.model_file == "none":
                model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type = args.gnn_type)

                model_graphcl = graphcl(model.gnn)
                model.gnn = model_graphcl.gnn
            else:    
                model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type = args.gnn_type)

                checkpoint = torch.load(args.model_file)
                model_graphcl = graphcl(model.gnn)
                model_graphcl.gnn.load_state_dict(checkpoint)
                model.gnn = model_graphcl.gnn

            model.to(device)
            #set up optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

            train_acc_list = []
            val_acc_list = []
            test_acc_list =  []

            #for roc plot
            test_fper_list = []
            test_tper_list= []

            # for loss plot
            loss_values = []
            val_loss_values = []

            #for AUC plot
            test_acc_mean_list = []

            for epoch in range(1, args.epochs+1):

                print("=====epoch "+ str(epoch))
                #train(args, model, device, train_loader, optimizer, num_tasks=args.num_tasks)

                running_loss, val_running_loss = train(args, model, device, train_loader, val_loader, optimizer, num_tasks=args.num_tasks)
                loss_values.append(running_loss)
                val_loss_values.append(val_running_loss)
                print("running_loss and val_running_loss: ", running_loss, val_running_loss)

                print("====Evaluation")
                if args.eval_train:
                    train_acc, _, _ = eval(args, model, device, train_loader, num_tasks=args.num_tasks)
                else:
                    train_acc = 0
                    print("ommitting training evaluation")
                val_acc, _, _ = eval(args, model, device, val_loader, num_tasks=args.num_tasks)
                #print("val roc list: ", val_acc)

                val_acc_list.append(np.mean(val_acc))
                train_acc_list.append(np.mean(train_acc))


                test_acc, test_fper, test_tper = eval(args, model, device, test_loader, num_tasks=args.num_tasks)
                #print("test roc list: ", val_acc)
                test_acc_list.append(test_acc)
                test_acc_mean_list.append(np.mean(test_acc))
                test_fper_list.append(test_fper)
                test_tper_list.append(test_tper)



                print()

                #print("train_acc: ",train_acc_list)
                #print("valication acc : ", val_acc_list)
                #print("test_acc : ", test_acc_list)
                #print()

            #loss plot    
            # plt.plot(np.array(loss_values), 'r',label='Train loss')
            # plt.plot(np.array(val_acc_list), 'orange',label='Validation Metric (AUROC)')
            # plt.plot(np.array(val_loss_values), 'b', label='Validation loss')
            # plt.title('Loss & Validation Metric (AUROC)')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.savefig("./results/"+args.resultFile_name+"_"+str(args.runseed)+'_fold'+str(kfold)+'.png')

            fig, ax1 = plt.subplots()
            ax1.plot(np.array(loss_values), color='red',label='Train loss')
            ax1.plot(np.array(val_loss_values), color='blue', label='Validation loss')
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            ax2.plot(np.array(val_acc_list), color='orange',label='Validation Metric')
            ax2.plot(np.array(train_acc_list), color='green',label='Train Metric')
            ax2.set_ylim(0.3, 1)
            plt.title('Loss & Metric (AUROC)')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax2.set_ylabel('AUROC')
            ax2.legend(loc='upper right')
            plt.savefig("./results/"+args.resultFile_name+"_"+str(args.runseed)+'_fold'+str(kfold)+'.png')
            plt.close()

            #auc plot
            # plt.plot(np.array(val_acc_list), 'orange',label='Validation Metric (AUROC)')
            # plt.plot(np.array(test_acc_mean_list), 'green', label='Test Metric (AUROC)')
            # plt.title('Evaluation')
            # plt.xlabel('Epoch')
            # plt.ylabel('AUROC')
            # plt.legend()
            # plt.savefig("./results/"+args.resultFile_name+"_"+str(args.runseed)+"_Evaluation"+'_fold'+str(kfold)+'.png')

            #print("np.array(val_acc_list).argmax()", np.array(val_acc_list).argmax())
            for i in range(5):
                print(len(test_fper_list[i][0]))
            for i in range(5):
                print(len(test_tper_list[i][0]))
            #print("test_fper_list.shape", len(test_fper_list))
            #print("np.array(val_acc_list)", np.array(val_acc_list))
            #print("test_fper_list.shape", len(test_fper_list[0]))
            #print("test!!!! : ", test_fper_list[np.array(val_acc_list).argmax()][0])

            plot_roc_curve(test_fper_list[np.array(val_acc_list).argmax()][0], test_tper_list[np.array(val_acc_list).argmax()][0], "ASD")
            plt.savefig("./results/"+args.resultFile_name+"_"+"roc_curve_label0"+ str(args.runseed)+'_fold'+str(kfold)+'.png')
            plot_roc_curve(test_fper_list[np.array(val_acc_list).argmax()][1], test_tper_list[np.array(val_acc_list).argmax()][1], "ASD+ADHD")
            plt.savefig("./results/"+args.resultFile_name+"_"+"roc_curve_label1"+ str(args.runseed)+'_fold'+str(kfold)+'.png')

            with open('./results/'+args.resultFile_name+'_fold'+str(kfold)+'.res', 'a+') as f:
                f.write(str(args.runseed) + ' ' + str(np.array(val_acc_list).max()) + ' ' + str(np.array(test_acc_list)[np.array(val_acc_list).argmax()]))
                f.write('\n')

            with open('./results/'+args.resultFile_name+'_fold'+str(kfold)+'_fper-tper.txt', 'a+') as f:
                f.write(str(test_fper_list[np.array(val_acc_list).argmax()][0])+str(test_tper_list[np.array(val_acc_list).argmax()][0]) \
                    +str(test_fper_list[np.array(val_acc_list).argmax()][1])+str(test_tper_list[np.array(val_acc_list).argmax()][1]))
                f.write('\n')

            ## encoder의 respresentation tsne 결과




            #for data in test_loader:
            #    model(data.to(device), do_visualize=True)
    
                #plt.savefig('./finetune_tsne/'+args.resultFile_name+"_tsne_seed"+str(args.runseed)+'.png')

        kfold += 1
    
if __name__ == "__main__":
    main()