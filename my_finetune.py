 ### Finetuning for ASD classification ###


import argparse 
import re 

from loader import BioDataset
from torch.utils.data.dataloader import DataLoader
from dataloader import DataLoaderFinetune
from splitters import random_split, species_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.ops import sigmoid_focal_loss

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

import pandas as pd

import os
import pickle
#from pretrain_greverative_joao import graphcl
from pretrain_joao import graphcl

from halfhop import HalfHop 
from torch_geometric.nn import summary

torch.autograd.set_detect_anomaly(True)

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()

def train(args, model, device, loader, val_loader, optimizer, num_tasks, halfhop):
    #weight =  torch.FloatTensor([1.0,2.0]).to(device)
    #criterion = nn.BCELoss(weight=weight)
    with torch.autograd.detect_anomaly():
        model.train()
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            keep_batch = batch.batch
            batch = batch.to(device)

            pred = model(batch)
        
            y = F.one_hot(batch.y, num_classes=num_tasks)
            print('y: ',y)
            print('y: ',y.shape)
            print('pred', pred)
            print('pred.shape', pred.shape)
            y = y.view(pred.shape).to(torch.float64)
        
            optimizer.zero_grad()
            loss = criterion(pred.double(), y.double())
            loss.backward()
            print(f'train_loss {loss}, y {y[0].double()}, pred {pred[0].double()}')
            
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        for i, data in enumerate(val_loader):
           
            data = data.to(device)
            val_pred = model(data)
            val_y = F.one_hot(data.y, num_classes=num_tasks)
            val_y = val_y.view(val_pred.shape).to(torch.float64)
        
            val_loss += criterion(val_pred.double(), val_y.double())
            print(f'validation {val_loss}, y {val_y[0].double()}, val_pred {val_pred[0].double()}')
        val_loss = val_loss/len(val_loader)  

    return loss.item(), val_loss.item() , val_y.double() , val_pred.double()
        
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
    
    
    
    roc_list = []
    fper_list = []
    tper_list = []

    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            roc_list.append(roc_auc_score(y_true[:,i], y_scores[:,i]))
            fper, tper, _ = roc_curve(y_true[:,i], y_scores[:,i])
            #)
            fper_list.append(fper)
            tper_list.append(tper)
            
        else:
            roc_list.append(np.nan)
    
    y_pred_one_hot = []
    for i in range(len(y_scores)):
        if y_scores[i,0] < y_scores[i,1]:
            y_pred_new = 0
        else: y_pred_new = 1
        y_pred_one_hot.append(y_pred_new)      
    acc = accuracy_score(y_true[:,0], np.array(y_pred_one_hot))   
    cm = confusion_matrix(y_true[:,0],y_pred_one_hot)
    print("acc",acc)    
    
    return np.array(roc_list),fper_list,tper_list, acc, cm  #y_true.shape[1]

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
    parser.add_argument('--halfhop', type=str, default='yes')
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
    
    if args.kfold == 1:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=613)
        skf2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=613)
    else :
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=613)
        skf2 = StratifiedKFold(n_splits=args.kfold-1, shuffle=True, random_state=613)
        
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
        if kfold == 2 and (args.kfold == 1 or args.kfold == 0):
            break
        else: 
            test_y=[]
            for j in range(len(train_val_index)):
                train_val_dataset.append(dataset[train_val_index[j]])
            for j in range(len(test_index)):
                test_dataset.append(dataset[test_index[j]])
                test_y.append(int(test_dataset[j].y))

            train_y = []
            train_x_idx = range(len(train_val_dataset))
            for i in range(len(train_val_dataset)):
                train_y.append(int(train_val_dataset[i].y))    

            print("train_val_index : ", train_val_index)
            print("train_y : ", train_y)
            print("test_index : ", test_index)
            print("test_y: ", test_y)

            tmp = 0
            for train_index, val_index in skf2.split(train_x_idx, train_y):
                tmp=0

            for j in range(len(train_index)):
                train_dataset.append(train_val_dataset[train_index[j]])
            for j in range(len(val_index)):
                val_dataset.append(train_val_dataset[val_index[j]])   

            print("val_index : ", val_index)

            print(train_dataset)
            
            if args.halfhop == 'yes':
                for i in range(len(train_dataset)):
                    transform = HalfHop(alpha=0.5)
                    train_dataset[i] = transform(train_dataset[i])
                    
            print(train_dataset)
            
            train_loader = DataLoaderFinetune(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
            val_loader = DataLoaderFinetune(val_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)
            test_loader = DataLoaderFinetune(test_dataset, batch_size=10*args.batch_size, shuffle=False, num_workers = args.num_workers)


            num_tasks = args.num_tasks

            #set up model

            if args.model_file == "none":
                model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type = args.gnn_type)

                model_graphcl = graphcl(model.gnn, args.emb_dim)
                model.gnn = model_graphcl.gnn
            else:    
                model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type = args.gnn_type)

                checkpoint = torch.load(args.model_file)
                model_graphcl = graphcl(model.gnn, args.emb_dim)
                model_graphcl.gnn.load_state_dict(checkpoint)
                model.gnn = model_graphcl.gnn

            model.to(device)
            #set up optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

            train_auc_list = []
            val_auc_list = []
            test_auc_list =  []
            test_acc_list = []
            train_acc_list = []
            val_acc_list = []
            test_cm_list = []
            sensitive_list = []
            specific_list = []

            #for roc plot
            test_fper_list = []
            test_tper_list= []

            # for loss plot
            loss_values = []
            val_loss_values = []

            #for AUC plot
            test_auc_mean_list = []

            for epoch in range(1, args.epochs+1):

                print("=====epoch "+ str(epoch))

                running_loss, val_running_loss, val_y, val_pred = train(args, model, device, train_loader, val_loader, optimizer, num_tasks=args.num_tasks, halfhop=args.halfhop)
                loss_values.append(running_loss)
                val_loss_values.append(val_running_loss)
                print("running_loss and val_running_loss: ", running_loss, val_running_loss)

                print("====Evaluation")
                if args.eval_train:
                    train_auc, _, _, train_acc, train_cm = eval(args, model, device, train_loader, num_tasks=args.num_tasks)
                else:
                    train_auc = 0
                    print("ommitting training evaluation")
                val_auc, _, _, val_acc, val_cm = eval(args, model, device, val_loader, num_tasks=args.num_tasks)

                val_auc_list.append(np.mean(val_auc))
                val_acc_list.append(val_acc)
                print('train_auc', train_auc)
                train_auc_list.append(np.mean(train_auc))
                train_acc_list.append(train_acc)


                test_auc, test_fper, test_tper, test_acc, test_cm = eval(args, model, device, test_loader, num_tasks=args.num_tasks)
                #print("test roc list: ", val_auc)
                sensitivity = test_cm[0,0]/(test_cm[0,0]+test_cm[0,1])
                sensitive_list.append(sensitivity)
                specificity = test_cm[1,1]/(test_cm[1,0]+test_cm[1,1])
                specific_list.append(specificity)
                test_auc_list.append(test_auc)
                test_acc_list.append(test_acc)
                test_auc_mean_list.append(np.mean(test_auc))
                test_cm_list.append(test_cm)
                test_fper_list.append(test_fper)
                test_tper_list.append(test_tper)
                
                if epoch == 1:
                    max_model = model
                if epoch > 1 :
                    max_auc, _, _, _ , _= eval(args, max_model, device, test_loader, num_tasks=args.num_tasks)
                    max_eval = np.mean(max_auc)
                    cur_eval = np.mean(val_auc)
                    if cur_eval > max_eval: 
                        max_model = model
                    else: 
                        max_model = max_model
                file_name = args.resultFile_name.split('/')
                weight_path = f'/nasdata3/kyj/graphcl/GraphCL_Automated/transferLearning_MoleculeNet_PPI/bio/fintune_weight/{file_name[-1]}_fold{str(kfold)}_seed{args.runseed}.pt'
                torch.save(max_model.state_dict(), weight_path)


            print(val_y, val_pred)
            #loss plot    
            # plt.plot(np.array(loss_values), 'r',label='Train loss')
            # plt.plot(np.array(val_auc_list), 'orange',label='Validation Metric (AUROC)')
            # plt.plot(np.array(val_loss_values), 'b', label='Validation loss')
            # plt.title('Loss & Validation Metric (AUROC)')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.savefig("./results/"+args.resultFile_name+"_"+str(args.runseed)+'_fold'+str(kfold)+'.png')

            fig, ax1 = plt.subplots()
            ax1.plot(np.array(loss_values), color='red',label='Train loss')
            ax1.plot(np.array(val_loss_values), color='blue', label='Validation loss')
           # ax1.set_ylim(0, 1)
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            ax2.plot(np.array(val_auc_list), color='orange',label='Validation Metric')
            print(np.shape(np.array(train_auc_list)))
            ax2.plot(np.array(train_auc_list), color='green',label='Train Metric')
            ax2.set_ylim(0.3, 1)
            plt.title('Loss & Metric (AUROC)')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax2.set_ylabel('AUROC')
            ax2.legend(loc='upper right')
            plt.savefig("./results/"+args.resultFile_name+"_"+str(args.runseed)+'_fold'+str(kfold)+'.png')
            plt.close()

            #accuracy plot
            plt.plot(np.array(train_acc_list), 'orange',label='Train Metric (ACC)')
            plt.plot(np.array(val_acc_list), 'green', label='Validation Metric (ACC)')
            plt.title('Train and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('ACC')
            plt.legend()
            plt.savefig("./results/"+args.resultFile_name+"_"+str(args.runseed)+"_accuracy"+'_fold'+str(kfold)+'.png')
            plt.close()
            
            #confusion matrix plot 
            
            sns.heatmap(test_cm_list[np.array(val_auc_list).argmax()], annot=True, cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig("./results/"+args.resultFile_name+"_"+str(args.runseed)+"_CM"+'_fold'+str(kfold)+'.png')
            plt.close()
            
            #auc plot
            # plt.plot(np.array(val_auc_list), 'orange',label='Validation Metric (AUROC)')
            # plt.plot(np.array(test_auc_mean_list), 'green', label='Test Metric (AUROC)')
            # plt.title('Evaluation')
            # plt.xlabel('Epoch')
            # plt.ylabel('AUROC')
            # plt.legend()
            # plt.savefig("./results/"+args.resultFile_name+"_"+str(args.runseed)+"_Evaluation"+'_fold'+str(kfold)+'.png')

            #print("np.array(val_auc_list).argmax()", np.array(val_auc_list).argmax())
            for i in range(5):
                print(len(test_fper_list[i][0]))
            for i in range(5):
                print(len(test_tper_list[i][0]))
            #print("test_fper_list.shape", len(test_fper_list))
            #print("np.array(val_auc_list)", np.array(val_auc_list))
            #print("test_fper_list.shape", len(test_fper_list[0]))
            #print("test!!!! : ", test_fper_list[np.array(val_auc_list).argmax()][0])

            plot_roc_curve(test_fper_list[np.array(val_auc_list).argmax()][0], test_tper_list[np.array(val_auc_list).argmax()][0], "ASD")
            plt.savefig("./results/"+args.resultFile_name+"_"+"roc_curve_label0"+ str(args.runseed)+'_fold'+str(kfold)+'.png')
            plot_roc_curve(test_fper_list[np.array(val_auc_list).argmax()][1], test_tper_list[np.array(val_auc_list).argmax()][1], "ASD+ADHD")
            plt.savefig("./results/"+args.resultFile_name+"_"+"roc_curve_label1"+ str(args.runseed)+'_fold'+str(kfold)+'.png')
            plt.close()
            with open('./results/'+args.resultFile_name+'_fold'+str(kfold)+'.res', 'a+') as f:
                f.write(str(args.runseed) + ' ' + str(np.array(val_auc_list).max()) + ' ' + str(np.array(test_auc_list)[np.array(val_auc_list).argmax()])\
                    + ' ' + str(np.array(test_acc_list)[np.array(val_auc_list).argmax()]) + ' ' + str(np.array(sensitive_list)[np.array(val_auc_list).argmax()]) \
                    + ' ' + str(np.array(specific_list)[np.array(val_auc_list).argmax()]))
                f.write('\n')

            with open('./results/'+args.resultFile_name+'_fold'+str(kfold)+'_fper-tper.txt', 'a+') as f:
                f.write(str(test_fper_list[np.array(val_auc_list).argmax()][0])+str(test_tper_list[np.array(val_auc_list).argmax()][0]) \
                    +str(test_fper_list[np.array(val_auc_list).argmax()][1])+str(test_tper_list[np.array(val_auc_list).argmax()][1]))
                f.write('\n')
            
            with open('./results/'+args.resultFile_name+'_fold'+str(kfold)+"_seed"+str(args.runseed)+'_loss_array.txt', 'w') as f:
                f.write(str(loss_values))
                f.write('\n')
                f.write(str(val_loss_values))
                f.write('\n')
                f.write(str(val_auc_list))
                f.write('\n')
                f.write(str(train_auc_list))
                f.write('\n')
                f.write(str(train_acc_list))
                f.write('\n')
                f.write(str(val_acc_list))
                f.write('\n')
                
                
                
            ## encoder의 respresentation tsne 결과

            #for data in test_loader:
            #    model(data.to(device), do_visualize=True)
    
                #plt.savefig('./finetune_tsne/'+args.resultFile_name+"_tsne_seed"+str(args.runseed)+'.png')

        kfold += 1
    
if __name__ == "__main__":
    main()