import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, recall_score, make_scorer, precision_score

import torch
import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, StratifiedKFold





parser = argparse.ArgumentParser(description='For the comparison of other models')
parser.add_argument('--model', type=str, default='LR',
                    help='LR, SVM, MLP (default: LR)')
parser.add_argument('--edge', type=int, default=5,
                    help='edge sparsity 5, 10 or 15 (default: 5)')
parser.add_argument('--random_state', type=int, default=0,
                    help='0, 10, 20, ... 90 (default: 0)')
parser.add_argument('--random_state_mlp', type=int, default=613,
                    help='0, 10, 20, ... 90 (default: 0)')

args = parser.parse_args()


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
        'fn': cm[1, 0], 'tp': cm[1, 1]}


#---------------------------------------------------------
#       Logistic Regression
#---------------------------------------------------------
if args.model == 'LR':

    dataset = torch.load(f'./dataset/spars{args.edge}_supervised/only_cc_balanced/processed/geometric_data_processed.pt')
    edge5_X = []
    edge5_y = []
    for i in range(len(dataset)):
        data = dataset[i]
        x = np.array(data.x).flatten()
        x = list(x)
        edge5_X.append(x)
        y = np.array(data.y)
        y = list(y)
        edge5_y.append(y)
    edge5_y = np.array(edge5_y).reshape(-1)
    edge5_X = np.array(edge5_X)
    print(edge5_y)
    
    edge5_df = pd.DataFrame(edge5_X)
    edge5_df['target'] = edge5_y
    edge5_df.shape    


    #input data 
    x=edge5_df.iloc[:,:-1] 
    #output data
    y=edge5_df.iloc[:,-1]

    auc_list = []
    acc_list = []
    cv_specificity = []
    cv_recall = []


    result_skfold = StratifiedKFold(n_splits=5)

    result_clf = LogisticRegression(max_iter=5000)
    
    n_iter = 0
    for train_idx, test_idx in result_skfold.split(x, y):
        X_train, X_test = x.iloc[train_idx,:], x.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]
        
    
        #fit
        print(X_train)
        print(y_train)
        result_clf.fit(X_train, y_train)
        
        #predict
        
        pred = result_clf.predict(X_test)
        n_iter += 1
        #precision_score, recall_score() 함수를 이용해 정밀도와 재현율 계산, 출력 
        acc = np.round(accuracy_score(y_test, pred),3)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
        auc = np.round(metrics.auc(fpr, tpr),3)
        recall = np.round(recall_score(y_test, pred),3)
        specificity = np.round(recall_score(y_test, pred, pos_label = 0),3)
        print('\n{} 교차검증 acc :{}, 교차검증 auc :{}, 교차검증 recall :{}, 교차검증 spesificity :{}'.format(n_iter, acc, auc, recall, specificity))
        #append()함수로 리스트에 저장 
        acc_list.append(acc)
        auc_list.append(auc)
        cv_specificity.append(specificity)
        cv_recall.append(recall)

    print('\n')
    print('\n 평균 검증 정확도', np.mean(acc_list), np.mean(auc_list), np.mean(cv_specificity), np.mean(cv_recall))
    
    with open('./results/'+'comparison_result/'+'ASD_vs_COM_'+str(args.model)+"_edge"+str(args.edge)+'.res', 'a+') as f:
        f.write(str( np.mean(acc_list)) + ' ' +   str(np.round(np.std(acc_list),3) ) + ' ' + str(np.mean(auc_list) )+ ' ' + str(np.round(np.std(auc_list),3))+ ' '\
            + str(np.mean(cv_specificity))+' ' + str(np.round(np.std(cv_specificity),3) ) + ' ' + str(np.mean(cv_recall))+' ' +str( np.round(np.std(cv_recall),3)))
        f.write('\n')

    
    # ---------------------------------------
    # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,
                                            # random_state=i,
                                            # stratify=y)
# 
# 
    # log_regression = LogisticRegression(max_iter=5000)
    #log_regression.fit(x_train,y_train)
# 
# 
    # acc = cross_val_score(log_regression, x, y, cv = 5, scoring='accuracy')
    # auc = cross_val_score(log_regression, x, y, cv = 5, scoring='roc_auc')
    # print()
    # print(pd.DataFrame(cross_validate(log_regression, x, y, cv =5, scoring=['accuracy','roc_auc'], return_train_score=True)))
# 
    # print('mean acc: ', acc.mean())
    # print('SD acc: ', acc.std())
    # print('mean auc: ', auc.mean())
    # print('SD auc: ', auc.std())
    # print('mean sensitivity : ', sensitivity.mean())
        
    #     # Testing the model using the testing data
    #     y_pred_proba = log_regression.predict(x_test)
    #     print("y_test", y_test)
    #     print("y_pred_proba",y_pred_proba)
    #     train_pred = log_regression.predict(x_train)
  
    #     # Calculating the accuracy of the model
    #     accuracy = accuracy_score(y_pred_proba, y_test)
    #     auc = roc_auc_score(y_test, y_pred_proba)
    #     accuracy2 = accuracy_score(train_pred , y_train)
    #     auc2 = roc_auc_score(y_train, train_pred)
    #     print(accuracy2)
    #     print(auc2)
    #     auc_list.append(auc)
    #     acc_list.append(accuracy)
    
    # mean_acc = np.mean(acc_list)
    # std_acc = np.std(acc_list)
    # mean_auc = np.mean(auc_list)
    # std_auc = np.std(auc_list)
    # # Print the accuracy of the model
    # print(f"The logistic regression with edge {args.edge} is {mean_acc*100}% +/-{std_acc} accurate")
    # print(f"The logistic regression with edge {args.edge} is {mean_auc*100}% +/-{std_auc} AUC")
    
    # with open('./results/'+'comparison_result/'+'ASD_vs_COM_'+str(args.model)+"_edge"+str(args.edge)+'.res', 'a+') as f:
                # f.write(str(acc.mean()) + ' ' +  str(acc.std()) + ' ' + str(auc.mean()) + ' ' + str(auc.std()))
                # f.write('\n')
   
#---------------------------------------------------------
#       SVM
#---------------------------------------------------------    
if args.model == 'SVM':
    dataset = torch.load(f'./dataset/spars{args.edge}_supervised/only_cc_balanced/processed/geometric_data_processed.pt')
    edge5_X = []
    edge5_y = []
    for i in range(len(dataset)):
        data = dataset[i]
        x = np.array(data.x).flatten()
        x = list(x)
        edge5_X.append(x)
        y = np.array(data.y)
        y = list(y)
        edge5_y.append(y)
    edge5_y = np.array(edge5_y).reshape(-1)
    edge5_X = np.array(edge5_X)
    print(edge5_y)
    
    edge5_df = pd.DataFrame(edge5_X)
    edge5_df['target'] = edge5_y
    edge5_df.shape    


    #input data 
    x=edge5_df.iloc[:,:-1]
    #output data
    y=edge5_df.iloc[:,-1]
    
    scaler = StandardScaler()
    scaler.fit(x)
    X_scaled = scaler.transform(x)

    
    
    auc_list = []
    acc_list = []
    cv_specificity = []
    cv_recall = []


    result_skfold = StratifiedKFold(n_splits=5)

    svc=svm.SVC(probability=True, random_state=100)
    result_clf = svm.SVC(kernel = 'linear', random_state=100)
    
    n_iter = 0
    for train_idx, test_idx in result_skfold.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx,:], X_scaled[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]
        
    
        #fit
        print(X_train.shape)
        print(y_train.shape)
        result_clf.fit(X_train, y_train)
        
        #predict
        
        pred = result_clf.predict(X_test)
        n_iter += 1
        #precision_score, recall_score() 함수를 이용해 정밀도와 재현율 계산, 출력 
        acc = np.round(accuracy_score(y_test, pred),3)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
        auc = np.round(metrics.auc(fpr, tpr),3)
        recall = np.round(recall_score(y_test, pred),3)
        specificity = np.round(recall_score(y_test, pred, pos_label = 0),3)
        print('\n{} 교차검증 acc :{}, 교차검증 auc :{}, 교차검증 recall :{}, 교차검증 spesificity :{}'.format(n_iter, acc, auc, recall, specificity))
        #append()함수로 리스트에 저장 
        acc_list.append(acc)
        auc_list.append(auc)
        cv_specificity.append(specificity)
        cv_recall.append(recall)

    print('\n')
    print('\n 평균 검증 정확도', np.mean(acc_list), np.mean(auc_list), np.mean(cv_specificity), np.mean(cv_recall))
    
    with open('./results/'+'comparison_result/'+'ASD_vs_COM_'+str(args.model)+"_edge"+str(args.edge)+'.res', 'a+') as f:
        f.write(str( np.mean(acc_list)) + ' ' +   str(np.round(np.std(acc_list),3) ) + ' ' + str(np.mean(auc_list) )+ ' ' + str(np.round(np.std(auc_list),3))+ ' '\
            + str(np.mean(cv_specificity))+' ' + str(np.round(np.std(cv_specificity),3) ) + ' ' + str(np.mean(cv_recall))+' ' +str( np.round(np.std(cv_recall),3)))
        f.write('\n')

    
    
    

    # i = args.random_state
    # x_train,x_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.20,
                                            # random_state=i,
                                            # stratify=y)    
    # Defining the parameters grid for GridSearchCV
    # param_grid={'C':[0.1,1,10,100],
                # 'gamma':[0.0001,0.001,0.1,1],
                # 'kernel':['rbf','poly']}
  
    # Creating a support vector classifier
    # svc=svm.SVC(probability=True, random_state=100)
    # svm_clf =svm.SVC(kernel = 'linear', random_state=100)

    
    # Creating a model using GridSearchCV with the parameters grid
    # svm=GridSearchCV(svc,param_grid, cv=5)
    
    # Training the model using the training data
    # svm_clf.fit(x_train,y_train)
    
    # Testing the model using the testing data
    # y_pred = svm_clf.predict(x_test)
    # print(y_pred)
    # print(y_test)
    # train_pred = svm_clf.predict(x_train)
    # acc = cross_val_score(svm_clf, X_scaled, y, cv = 5, scoring='accuracy')
    # auc = cross_val_score(svm_clf, X_scaled, y, cv = 5, scoring='roc_auc')
    # y_pred = cross_val_predict(svm_clf, X_scaled, y, cv=5)
    # cm = confusion_matrix(y, y_pred)
    # sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    # specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    # print(sensitivity)
    # print(specificity)
    # print()
    # print(pd.DataFrame(cross_validate(svm_clf, X_scaled, y, cv=5, scoring=['accuracy','roc_auc','confusion_matrix_scorer'], return_train_score=True)))
# 
    # print('mean acc: ', acc.mean())
    # print('SD acc: ', acc.std())
    # print('mean auc: ', auc.mean())
    # print('SD auc: ', auc.std())
# 
# 
    # Calculating the accuracy of the model
    # accuracy = accuracy_score(y_pred, y_test)
    # auc = roc_auc_score(y_test, y_pred)
    # accuracy2 = accuracy_score(train_pred, y_train)
    # auc2 = roc_auc_score(y_train, train_pred)
    # print("The accuracy with train data :", accuracy2)
    # print("The auc with train data :", auc2)

    # Print the accuracy of the model
    # print(f"The SVM with edge {args.edge} & random stsate {args.random_state} is {accuracy*100}%  accurate")
    # print(f"The SVM with edge {args.edge} & random stsate {args.random_state}is {auc*100}%  AUC")
    # with open('./results/'+'comparison_result/'+'ASD_vs_COM_'+str(args.model)+"_edge"+str(args.edge)+'.res', 'a+') as f:
                # f.write(str(acc.mean()) + ' ' +  str(acc.std()) + ' ' + str(auc.mean()) + ' ' + str(auc.std()))
                # f.write('\n')
#---------------------------------------------------------
#       MLP
#---------------------------------------------------------        
        
if args.model == 'MLP':
    dataset = torch.load(f'./dataset/spars{args.edge}_supervised/only_cc_balanced/processed/geometric_data_processed.pt')
    edge5_X = []
    edge5_y = []
    for i in range(len(dataset)):
        data = dataset[i]
        x = np.array(data.x).flatten()
        x = list(x)
        edge5_X.append(x)
        y = np.array(data.y)
        y = list(y)
        edge5_y.append(y)
    edge5_y = np.array(edge5_y).reshape(-1)
    edge5_X = np.array(edge5_X)
    print(edge5_y)
    
    
    edge5_df = pd.DataFrame(edge5_X)
    edge5_df['target'] = edge5_y
    edge5_df.shape    


    #input data 
    x=edge5_df.iloc[:,:-1] 
    #output data
    y=edge5_df.iloc[:,-1]


    
    
    
    auc_list = []
    acc_list = []
    cv_specificity = []
    cv_recall = []


    result_skfold = StratifiedKFold(n_splits=5)

    mlp_rs = args.random_state_mlp
    result_clf = MLPClassifier(random_state=mlp_rs, max_iter=5000)
    
    n_iter = 0
    for train_idx, test_idx in result_skfold.split(x, y):
        X_train, X_test = x.iloc[train_idx,:], x.iloc[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]
        
    
        #fit
        print(X_train.shape)
        print(y_train.shape)
        result_clf.fit(X_train, y_train)
        
        #predict
        
        pred = result_clf.predict(X_test)
        n_iter += 1
        #precision_score, recall_score() 함수를 이용해 정밀도와 재현율 계산, 출력 
        acc = np.round(accuracy_score(y_test, pred),3)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
        auc = np.round(metrics.auc(fpr, tpr),3)
        recall = np.round(recall_score(y_test, pred),3)
        specificity = np.round(recall_score(y_test, pred, pos_label = 0),3)
        print('\n{} 교차검증 acc :{}, 교차검증 auc :{}, 교차검증 recall :{}, 교차검증 spesificity :{}'.format(n_iter, acc, auc, recall, specificity))
        #append()함수로 리스트에 저장 
        acc_list.append(acc)
        auc_list.append(auc)
        cv_specificity.append(specificity)
        cv_recall.append(recall)

    print('\n')
    print('\n 평균 검증 정확도', np.mean(acc_list), np.mean(auc_list), np.mean(cv_specificity), np.mean(cv_recall))
    
    with open('./results/'+'comparison_result/'+'ASD_vs_COM_'+str(args.model)+"_edge"+str(args.edge)+'.res', 'a+') as f:
        f.write(str( np.mean(acc_list)) + ' ' +   str(np.round(np.std(acc_list),3) ) + ' ' + str(np.mean(auc_list) )+ ' ' + str(np.round(np.std(auc_list),3))+ ' '\
            + str(np.mean(cv_specificity))+' ' + str(np.round(np.std(cv_specificity),3) ) + ' ' + str(np.mean(cv_recall))+' ' +str( np.round(np.std(cv_recall),3)))
        f.write('\n')

    
    #i = args.random_state
    # mlp_rs = args.random_state_mlp
    # x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,
    #                                         random_state=i,
    #                                         stratify=y) 
    #clf = MLPClassifier(random_state=mlp_rs, max_iter=300).fit(x_train, y_train)
    # clf = MLPClassifier(random_state=mlp_rs, max_iter=5000)
    #----------------------------------------------
    # acc = cross_val_score(clf, x, y, cv = 10, scoring='accuracy')
    # auc = cross_val_score(clf, x, y, cv = 10, scoring='roc_auc')
    # print()
    # print(pd.DataFrame(cross_validate(clf, x, y, cv =10, scoring=['accuracy','roc_auc'], return_train_score=True)))
    #----------------------------------------------
    # print('mean acc: ', acc.mean())
    # print('SD acc: ', acc.std())
    # print('mean auc: ', auc.mean())
    # print('SD auc: ', auc.std())
    
    # y_pred = clf.predict(x_test)
    # print(y_pred)
    # train_pred = clf.predict(x_train)
    
    # accuracy = accuracy_score(y_pred, y_test)
    # auc = roc_auc_score(y_test, y_pred)
    # accuracy2 = accuracy_score(train_pred, y_train)
    # auc2 = roc_auc_score(y_train, train_pred)
    # print("The accuracy with train data :", accuracy2)
    # print("The auc with train data :", auc2)
    
    # print(f"The mlp with edge {args.edge} & random stsate {args.random_state} & random state mlp {args.random_state_mlp} is {accuracy*100}%  accurate")
    # print(f"The mlp with edge {args.edge} & random stsate {args.random_state} & random state mlp {args.random_state_mlp} is {auc*100}%  AUC")
    # with open('./results/'+'comparison_result/'+'ASD_vs_CN_'+str(args.model)+"_edge"+str(args.edge)+'.res', 'a+') as f:
    #             f.write(str(acc.mean()) + ' ' +  str(acc.std()) + ' ' + str(auc.mean()) + ' ' + str(auc.std()))
    #             f.write('\n')