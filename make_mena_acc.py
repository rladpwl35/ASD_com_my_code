import numpy as np
import sys

res_file = sys.argv[1]

with open(res_file, 'r') as f:
	f = f.readlines()
	for i in range(len(f)):
		f[i] = f[i].rstrip("\n")
		f[i] = f[i].replace("[","")
		f[i] = f[i].replace("]","")
					    
	f = [[float(num) for num in line.split()] for line in f]

	ASD=[]
	ASD_ADHD=[]
	ACC=[]
	sen=[]
	spe=[]
	for i in range(len(f)):
		ASD.append(f[i][2])
		ASD_ADHD.append(f[i][3])
		ACC.append(f[i][4])
		sen.append(f[i][5])
		spe.append(f[i][6])

ASD_auc_mean = np.mean(ASD)
ASD_auc_std = np.std(ASD)
ASD_ADHD_auc_mean = np.mean(ASD_ADHD)
ASD_auc_std = np.std(ASD_ADHD)
ACC_mean=np.mean(ACC)
ACC_std=np.std(ACC)
sen_mean=np.mean(sen)
sen_std=np.std(sen)
spe_mean=np.mean(spe)
spe_std=np.std(spe)
print("ASD_acc_mean: ",  ASD_auc_mean)
print('ASD_acc_std: ',ASD_auc_std)
print("ASD_ADHD_acc_mean: ", ASD_ADHD_auc_mean)
print("ASD_acc_std: ", ASD_auc_std)
print("ACC_mean: ", ACC_mean)
print("ACC_std: ", ACC_std)
print("sensitivity_mean: ", sen_mean)
print("sensitivity_std: ", sen_std)
print("specitic_mean: ", spe_mean)
print("specitic__std: ", spe_std)


with open(res_file, 'a+') as f:
	f.write(str(ASD_auc_mean))
	f.write('\n')
	f.write(str(ASD_auc_std))
	f.write('\n')
	f.write(str(ASD_ADHD_auc_mean))
	f.write('\n')
	f.write(str(ASD_auc_std))
	f.write('\n')
	f.write(str(ACC_mean))
	f.write('\n')
	f.write(str(ACC_std))
	f.write('\n')
	f.write(str(sen_mean))
	f.write('\n')
	f.write(str(sen_std))
	f.write('\n')
	f.write(str(spe_mean))
	f.write('\n')
	f.write(str(spe_std))
	f.write('\n')
