import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='make figure gain accuracy')
parser.add_argument('--encocer', type=str, default='gat',
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--file_dir', type=str, default='./',
                    help='')
parser.add_argument('--filename', type=str, default='',
                    help='')
parser.add_argument('--save_dir', type=str, default='./', help='')    
parser.add_argument('--out_filename', type=str, default='', help='')               
args = parser.parse_args()

#gin encoder accuracy gain

##no pretrain acc
edge = 'edge20'
with open(f'{args.file_dir}/{args.filename}','r') as file:
    x = file.readlines()
    asd = float(x[51].split()[1])
    com = float(x[53].split()[1])
    mean_no_pretrain = np.mean([asd, com])
    
# pretrain acc


gcn_mean_arr = []
gin_mean_arr = []
graphsage_mean_arr = []
file_list = open(f"./{edge}/gat_file_list",'r')
res_file_list = [i for i in file_list.read().split()]
for i in range(len(res_file_list)):
    with open(f'./{edge}/{res_file_list[i]}', 'r') as file:
        x = file.readlines()
        asd = float(x[10].split()[1])
        com = float(x[12].split()[1])
        mean = np.mean([asd, com])
        #print(asd, com, mean)
        gin_mean_arr.append(mean)
print(gin_mean_arr)
gin_mean_arr =  np.array(gin_mean_arr) - np.array(mean_no_pretrain) 
#gin_mean_arr = np.array(mean_no_pretrain) - np.array(gin_mean_arr) 
gin_mean_arr_np = np.reshape(gin_mean_arr,[5,5])

#-----------------------------------------
arr_sum = []
sum_axis0 = gin_mean_arr_np.sum(axis = 0)
sum_axis1 = gin_mean_arr_np.sum(axis = 1)
sum_all = np.concatenate((sum_axis0, sum_axis1), axis=0)
sum_all = np.reshape(sum_all, [2,5])
mean_all = np.mean(sum_all, axis = 0)
print("mean_all: ", mean_all)
ND = mean_all[0]
SUB = mean_all[1]
EP =mean_all[2]
AM = mean_all[3]
ID = mean_all[4]

#sum_result = 

print('sum_axis0: ', sum_axis0)
print('sum_axis1: ', sum_axis1)


x = np.arange(5)
augmentations = ['Node Drop','Subgraph','Edge Pert','Attr Mask','Identity']
values = mean_all

plt.bar(x, values)
plt.xticks(x, augmentations)

plt.show()
#plt.savefig(f'./{encoder}_manual_fig_{edge}_joaoGain_sum_with_no_pretrain.png')

#-----------------------------------------

gin_mean_arr = pd.DataFrame(gin_mean_arr_np, columns=['Node Drop', 'Subgraph', 'Edge Pertb', 'Attr Mask', 'Identity'],\
    index=['Node Drop', 'Subgraph', 'Edge Pert', 'Attr Mask', 'Identity'])
print(gin_mean_arr)

plt.pcolor(gin_mean_arr)

plt.set_cmap('bwr')

plt.xticks(np.arange(0.5, len(gin_mean_arr.columns), 1), gin_mean_arr.columns,rotation=45)

plt.yticks(np.arange(0.5, len(gin_mean_arr.index), 1), gin_mean_arr.index)

plt.title(f'GIN & HCP & partial corr_{(edge)} + ROIonehot', fontsize=20)

plt.xlabel('Augmenation 1', fontsize=14)

plt.ylabel('Augmentation 2', fontsize=14)


for y in range(5):
   for x in range(5):
      plt.text(x + 0.5, y + 0.5, '%.2f' % gin_mean_arr_np[y, x],
         horizontalalignment='center',
         verticalalignment='center',
      )

plt.colorbar()
plt.clim(-0.2, 0.2)
plt.savefig(f'{args.save_dir}/{args.out_filename}.png', bbox_inches = 'tight')