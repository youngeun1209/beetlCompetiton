#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 20:53:21 2021

@author: yelee
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 12:00:33 2021

@author: yelee
"""

# Generating files with label for leaderboard test
import time
from random import shuffle
import collections

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import tensorflow as tf

import mne
from mne.decoding import CSP
from mne.io.array.array import RawArray

import numpy as np
from numpy.random import RandomState
import pickle
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold

from braindecode.util import np_to_var, var_to_np
from braindecode.util import set_random_seeds

import sys
import os
__file__ = '/exSSD/projects/beetlCompetition/code'
sys.path.append(__file__)
from util.models import EEGyeModel
from util.utilfunc import get_balanced_batches
from util.preproc import balance_set,plot_confusion_matrix
from util.focalloss import FocalLoss as FocalLoss

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

cuda = torch.cuda.is_available()
print('gpu condition: ', cuda)
device = 'cuda' if cuda else 'cpu'



SEED=42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
rng = RandomState(SEED)

# %% pytorch GPU allocation
GPU_NUM = 2 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

# %% data structure
class TrainObject(object):
    def __init__(self, X, y):
        assert len(X) == len(y)
        # reshape to keep the values in time-domain
        e,c,t = X.shape
        X = np.reshape(X, (e,c*t))
        # Normalised, you could choose other normalisation strategy
        mean = np.mean(X,axis=1,keepdims=True)
        # here normalise across channels as an example, unlike the in the sleep kit
        std = np.std(X, axis=1, keepdims=True)
        X = (X - mean) / std
        # un-reshape
        X = np.reshape(X, (e,c,t))
        # we scale it to 1000 as a better training scale of the shallow CNN
        # according to the orignal work of the paper referenced above
        self.X = X.astype(np.float32) * 1e3
        self.y = y.astype(np.int64)

class TestObject(object):
    def __init__(self, X):
        # reshape to keep the values in time-domain
        e,c,t = X.shape
        X = np.reshape(X, (e,c*t))
        mean=np.mean(X,axis=1,keepdims=True)
        std=np.std(X,axis=1,keepdims=True)
        X=(X-mean)/(std)
        # un-reshape
        X = np.reshape(X, (e,c,t))
        #we scale it to 1000 as a better training scale of the shallow CNN
        #according to the orignal work of the paper
        self.X = X.astype(np.float32)*1e3

def relabel(l):
    if l == 0: return 0
    elif l == 1: return 1
    else: return 2


       
# %%        
MI_s1=None
MI_s2=None
MI_s3=None
MI_s4=None
MI_s5=None       

#
X_test_set = []
X_tr_v_set = []
Y_tr_v_set = []

window_size = 750


# %% channel select
used_chan = ['Fz','FC1','FC2','C5','C3','C1','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2']

# %
pilotname='S1'
savebase0='/exSSD/projects/beetlCompetition/data/finalMI/'
savebase = savebase0+pilotname+'/'+'testing/'

# read crop data
ch_names =['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 
           'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 
           'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8',
           'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 
           'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 
           'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 
           'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2',
           'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
#here we used the ones arround motor cortex
good_indices=[]
for i in range(0,len(used_chan)):
    good_indices.append(ch_names.index(used_chan[i]))

for i in range(5,15):
    with open (savebase+"race"+str(i+1)+"_padsData.npy", 'rb') as fp:
        Xt = pickle.load(fp)
    if i==5:
        X0 = Xt
    else:
        X0 = np.concatenate((X0,Xt))
    print('run',i+1)
    print(X0.shape)   
    
X_test = X0[:,good_indices,:window_size]

X_test_set.append(X_test)

# training
savebase = savebase0+pilotname+'/'+'training/'

# read crop data
for i in range(0,5):
    with open (savebase+"race"+str(i+1)+"_padsData.npy", 'rb') as fp:
        Xt = pickle.load(fp)
    with open (savebase+"race"+str(i+1)+"_padsLabel.npy", 'rb') as fp:
        Yt = pickle.load(fp)
    if i==0:
        X0 = Xt
        Y0 = Yt
    else:
        X0 = np.concatenate((X0,Xt))
        Y0 = np.concatenate((Y0,Yt))
    print('run',i+1)
    print(X0.shape)  
    print(Y0.shape)  
    
X_train = X0[:,good_indices,:window_size]

# y_t = np.array([relabel(l) for l in Y0])
y_train = Y0
X_tr_v_set.append(X_train)
Y_tr_v_set.append(y_train)


# %
# test
pilotname='S2'
savebase0='/exSSD/projects/beetlCompetition/data/finalMI/'
savebase = savebase0+pilotname+'/'+'testing/'

# read crop data
ch_names =['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 
           'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 
           'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8',
           'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 
           'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 
           'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 
           'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2',
           'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
#here we used the ones arround motor cortex
good_indices=[]
for i in range(0,len(used_chan)):
    good_indices.append(ch_names.index(used_chan[i]))

for i in range(5,15):
    with open (savebase+"race"+str(i+1)+"_padsData.npy", 'rb') as fp:
        Xt = pickle.load(fp)
    if i==5:
        X0 = Xt
    else:
        X0 = np.concatenate((X0,Xt))
    print('run',i+1)
    print(X0.shape)   
    
X_test = X0[:,good_indices,:window_size]
X_test_set.append(X_test)


# train
savebase = savebase0+pilotname+'/'+'training/'

# read crop data
for i in range(0,5):
    with open (savebase+"race"+str(i+1)+"_padsData.npy", 'rb') as fp:
        Xt = pickle.load(fp)
    with open (savebase+"race"+str(i+1)+"_padsLabel.npy", 'rb') as fp:
        Yt = pickle.load(fp)
    if i==0:
        X0 = Xt
        Y0 = Yt
    else:
        X0 = np.concatenate((X0,Xt))
        Y0 = np.concatenate((Y0,Yt))
    print('run',i+1)
    print(X0.shape)  
    print(Y0.shape)  
    
X_train = X0[:,good_indices,:window_size]
# y_t = np.array([relabel(l) for l in Y0])
y_train = Y0
X_tr_v_set.append(X_train)
Y_tr_v_set.append(y_train)


# %
pilotname='S3'
savebase0='/exSSD/projects/beetlCompetition/data/finalMI/'
savebase = savebase0+pilotname+'/'+'testing/'

# read crop data
ch_names =['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 
           'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 
           'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8',
           'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 
           'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 
           'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 
           'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2',
           'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
#here we used the ones arround motor cortex
good_indices=[]
for i in range(0,len(used_chan)):
    good_indices.append(ch_names.index(used_chan[i]))

for i in range(5,15):
    with open (savebase+"race"+str(i+1)+"_padsData.npy", 'rb') as fp:
        Xt = pickle.load(fp)
    if i==5:
        X0 = Xt
    else:
        X0 = np.concatenate((X0,Xt))
    print('run',i+1)
    print(X0.shape)   
    
X_test = X0[:,good_indices,:window_size]
X_test_set.append(X_test)


# train
savebase = savebase0+pilotname+'/'+'training/'

# read crop data
for i in range(0,5):
    with open (savebase+"race"+str(i+1)+"_padsData.npy", 'rb') as fp:
        Xt = pickle.load(fp)
    with open (savebase+"race"+str(i+1)+"_padsLabel.npy", 'rb') as fp:
        Yt = pickle.load(fp)
    if i==0:
        X0 = Xt
        Y0 = Yt
    else:
        X0 = np.concatenate((X0,Xt))
        Y0 = np.concatenate((Y0,Yt))
    print('run',i+1)
    print(X0.shape)  
    print(Y0.shape)  
    
X_train = X0[:,good_indices,:window_size]
# y_t = np.array([relabel(l) for l in Y0])
y_train = Y0
X_tr_v_set.append(X_train)
Y_tr_v_set.append(y_train)

# %
pilotname='S4'
savebase0='/exSSD/projects/beetlCompetition/data/finalMI/'
savebase = savebase0+pilotname+'/'+'testing/'

# read crop data
ch_names =['Fp1', 'Fp2', 'F3', 
            'Fz', 'F4', 'FC5', 'FC1', 'FC2','FC6', 'C5', 'C3',
           'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
           'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz', 
           'P2', 'P4', 'P6', 'P8']

good_indices=[]
for i in range(0,len(used_chan)):
    good_indices.append(ch_names.index(used_chan[i]))
    
with open (savebase+"testing_s4"+"X.npy", 'rb') as fp:
     Xt = pickle.load(fp)
X_test = Xt[:,good_indices,:window_size]
X_test_set.append(X_test)
      
   
# train     
savebase = savebase0+pilotname+'/'+'training/'

with open (savebase+"training_s4"+"X.npy", 'rb') as fp:
    Xt = pickle.load(fp)
with open (savebase+"training_s4"+"y.npy", 'rb') as fp:
    Yt = pickle.load(fp)
    Y0 = Yt
X_train = Xt[:,good_indices,:window_size]
# y_t = np.array([relabel(l) for l in Yt])
y_train = Y0
X_tr_v_set.append(X_train)
Y_tr_v_set.append(y_train)  

# %
pilotname='S5'
savebase0='/exSSD/projects/beetlCompetition/data/finalMI/'
savebase = savebase0+pilotname+'/'+'testing/'

# read crop data
ch_names =['Fp1', 'Fp2', 'F3', 
            'Fz', 'F4', 'FC5', 'FC1', 'FC2','FC6', 'C5', 'C3',
           'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
           'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz', 
           'P2', 'P4', 'P6', 'P8']

good_indices=[]
for i in range(0,len(used_chan)):
    good_indices.append(ch_names.index(used_chan[i]))
    
with open (savebase+"testing_s5"+"X.npy", 'rb') as fp:
    Xt = pickle.load(fp)
X_test = Xt[:,good_indices,:window_size]
X_test_set.append(X_test)
           
   
# train    
savebase = savebase0+pilotname+'/'+'training/'

# read crop data
with open (savebase+"training_s5"+"X.npy", 'rb') as fp:
    Xt = pickle.load(fp)    
with open (savebase+"training_s5"+"y.npy", 'rb') as fp:
    Yt = pickle.load(fp)
    Y0 = Yt
    
X_train = Xt[:,good_indices,:window_size]
# y_t = np.array([relabel(l) for l in Yt])
y_train = Y0
X_tr_v_set.append(X_train)
Y_tr_v_set.append(y_train)  


# %% Train / valid   
# training rate = 4/5
nCV = 3

# Sub 0,1,2
# 17 / 8
sub0_tr_v_t0 = np.squeeze(np.array(np.where(Y_tr_v_set[0]==0)))
sub0_tr_v_t1 = np.squeeze(np.array(np.where(Y_tr_v_set[0]==1)))
sub0_tr_v_t2 = np.squeeze(np.array(np.where(Y_tr_v_set[0]==2)))
sub0_tr_v_t3 = np.squeeze(np.array(np.where(Y_tr_v_set[0]==3)))
print(len(sub0_tr_v_t0), len(sub0_tr_v_t1), len(sub0_tr_v_t2), len(sub0_tr_v_t3))

sub1_tr_v_t0 = np.squeeze(np.array(np.where(Y_tr_v_set[1]==0)))
sub1_tr_v_t1 = np.squeeze(np.array(np.where(Y_tr_v_set[1]==1)))
sub1_tr_v_t2 = np.squeeze(np.array(np.where(Y_tr_v_set[1]==2)))
sub1_tr_v_t3 = np.squeeze(np.array(np.where(Y_tr_v_set[1]==3)))
print(len(sub1_tr_v_t0), len(sub1_tr_v_t1), len(sub1_tr_v_t2), len(sub1_tr_v_t3))

sub2_tr_v_t0 = np.squeeze(np.array(np.where(Y_tr_v_set[2]==0)))
sub2_tr_v_t1 = np.squeeze(np.array(np.where(Y_tr_v_set[2]==1)))
sub2_tr_v_t2 = np.squeeze(np.array(np.where(Y_tr_v_set[2]==2)))
sub2_tr_v_t3 = np.squeeze(np.array(np.where(Y_tr_v_set[2]==3)))
print(len(sub2_tr_v_t0), len(sub2_tr_v_t1), len(sub2_tr_v_t2), len(sub2_tr_v_t3))

sub012_t01_cv = [0 for a in range(nCV)]
sub012_t01_cv[0] = [list(range(0,17)), list(range(17,25))]
sub012_t01_cv[1] =[list(range(0,9))+list(range(17,25)), list(range(9,17))]
sub012_t01_cv[2] = [list(range(8,25)), list(range(0,8))]

sub012_t2_cv = [0 for a in range(nCV)]
sub012_t2_cv[0] = [list(range(0,9)), list(range(16,20))] # 9, 4
sub012_t2_cv[1] = [list(range(9,18)),list(range(21,25))] # 9, 4
sub012_t2_cv[2] = [list(range(17,25)), list(range(0,4))] # 8, 4

sub012_t3_cv = [0 for a in range(nCV)]
sub012_t3_cv[0] = [list(range(0,8)), list(range(16,20))] # 8, 4
sub012_t3_cv[1] = [list(range(8,16)),list(range(21,25))] # 8, 4
sub012_t3_cv[2] = [list(range(16,25)), list(range(0,4))] # 9, 4


# Sub 3,4
# 20 / 10
sub34_tr_v_t0 = np.squeeze(np.array(np.where(Y_tr_v_set[3]==0)))
sub34_tr_v_t1 = np.squeeze(np.array(np.where(Y_tr_v_set[3]==1)))
sub34_tr_v_t2 = np.squeeze(np.array(np.where(Y_tr_v_set[3]==2)))
sub34_tr_v_t3 = np.squeeze(np.array(np.where(Y_tr_v_set[3]==3)))

print(len(sub34_tr_v_t0), len(sub34_tr_v_t1), len(sub34_tr_v_t2), len(sub34_tr_v_t3))

sub34_t01_cv = [0 for a in range(nCV)]
sub34_t01_cv[0] = [list(range(0,20)), list(range(20,30))]
sub34_t01_cv[1] = [list(range(0,10))+list(range(20,30)), list(range(10,20))]
sub34_t01_cv[2] = [list(range(10,30)), list(range(0,10))]

sub34_t23_cv = [0 for a in range(nCV)]
sub34_t23_cv[0] = [list(range(0,10)), list(range(20,25))]
sub34_t23_cv[1] = [list(range(10,20)), list(range(25,30))]
sub34_t23_cv[2] = [list(range(20,30)), list(range(0,5))]


# index maker
cv_index = [[0 for a in range(nCV)] for b in range(5)]
for j, cv_ind in enumerate(cv_index[0]):
    # sub 0
    cv_index[0][j] = [np.concatenate((sub0_tr_v_t0[sub012_t01_cv[j][0]],sub0_tr_v_t1[sub012_t01_cv[j][0]],sub0_tr_v_t2[sub012_t2_cv[j][0]],sub0_tr_v_t3[sub012_t3_cv[j][0]])),
                      np.concatenate((sub0_tr_v_t0[sub012_t01_cv[j][1]],sub0_tr_v_t1[sub012_t01_cv[j][1]],sub0_tr_v_t2[sub012_t2_cv[j][1]],sub0_tr_v_t3[sub012_t3_cv[j][1]]))]
    # sub 1
    cv_index[1][j] = [np.concatenate((sub1_tr_v_t0[sub012_t01_cv[j][0]],sub1_tr_v_t1[sub012_t01_cv[j][0]],sub1_tr_v_t2[sub012_t2_cv[j][0]],sub1_tr_v_t3[sub012_t3_cv[j][0]])),
                      np.concatenate((sub1_tr_v_t0[sub012_t01_cv[j][1]],sub1_tr_v_t1[sub012_t01_cv[j][1]],sub1_tr_v_t2[sub012_t2_cv[j][1]],sub1_tr_v_t3[sub012_t3_cv[j][1]]))]
    # sub 2,
    cv_index[2][j] = [np.concatenate((sub2_tr_v_t0[sub012_t01_cv[j][0]],sub2_tr_v_t1[sub012_t01_cv[j][0]],sub2_tr_v_t2[sub012_t2_cv[j][0]],sub2_tr_v_t3[sub012_t3_cv[j][0]])),
                      np.concatenate((sub2_tr_v_t0[sub012_t01_cv[j][1]],sub2_tr_v_t1[sub012_t01_cv[j][1]],sub2_tr_v_t2[sub012_t2_cv[j][1]],sub2_tr_v_t3[sub012_t3_cv[j][1]]))]

    # sub 3,4
    for i in [3,4]:
        cv_index[i][j] = [np.concatenate((sub34_tr_v_t0[sub34_t01_cv[j][0]],sub34_tr_v_t1[sub34_t01_cv[j][0]],sub34_tr_v_t2[sub34_t23_cv[j][0]],sub34_tr_v_t3[sub34_t23_cv[j][0]])),
                          np.concatenate((sub34_tr_v_t0[sub34_t01_cv[j][1]],sub34_tr_v_t1[sub34_t01_cv[j][1]],sub34_tr_v_t2[sub34_t23_cv[j][1]],sub34_tr_v_t3[sub34_t23_cv[j][1]]))]


# duplicate check for training / validation
for i in range(0,5):
    for j, cv_ind in enumerate(cv_index[i]):
        
        u,c = np.unique(np.concatenate(cv_index[i][j]), return_counts=True)
        dup = list(u[c>1])
        if dup:
            print('DUPLICATE!!! sub%d, cv%d'%(i,j))
        

X_train_set = [[0 for a in range(nCV)] for b in range(5)]
X_valid_set=[[0 for a in range(nCV)] for b in range(5)]
Y_train_set=[[0 for a in range(nCV)] for b in range(5)]
Y_valid_set=[[0 for a in range(nCV)] for b in range(5)]

for i in range(0,5):
    for j, cv_ind in enumerate(cv_index[i]):
        tr_index, val_index = cv_ind
        print(tr_index.shape, val_index.shape)        
        X_train_set[i][j]  = X_tr_v_set[i][tr_index]
        y_l = np.array([relabel(l) for l in Y_tr_v_set[i][tr_index]])
        Y_train_set[i][j] = y_l
        
        
        X_valid_set[i][j]  = X_tr_v_set[i][val_index]
        y_l = np.array([relabel(l) for l in Y_tr_v_set[i][val_index]])
        Y_valid_set[i][j] = y_l


# % the number of training samples
for i in range(0,5):
    for j, cv_ind in enumerate(cv_index[i]):
        print(collections.Counter(Y_tr_v_set[i]))
        print(collections.Counter(Y_train_set[i][j]))
        print(collections.Counter(Y_valid_set[i][j]))
        print('\n')

# %% 
train_set=[[0 for a in range(nCV)] for b in range(5)]
valid_set=[[0 for a in range(nCV)] for b in range(5)]
test_set=[]
for i in range(0,5):
    for j, cv_ind in enumerate(cv_index[i]):
        train_set[i][j] = TrainObject(X_train_set[i][j], y=Y_train_set[i][j])
        valid_set[i][j] = TrainObject(X_valid_set[i][j], y=Y_valid_set[i][j])
    test_set.append(TestObject(X_test_set[i]))

# %%
input_time_length = train_set[0][0].X.shape[2]
in_chans = train_set[0][0].X.shape[1]
window_size = input_time_length
n_classes = len(np.unique(train_set[0][0].y))

# %% load model + freeze param
model_set = [[0 for a in range(nCV)] for b in range(5)]
savebase_log='/exSSD/projects/beetlCompetition/learning_logs/task2_MI/check_final/'

for i in range(0,5): 
    print('subject ' + str(i))
    X_test = test_set[i].X
    print('overall test size')
    print(X_test.shape)
    
    print(cuda)
    set_random_seeds(seed=42, cuda=cuda)
    input_time_length = X_test.shape[2]
    print(input_time_length)
    in_chans=X_test.shape[1]
    labelsize=3
    
    for j, cv_ind in enumerate(cv_index[i]):
        
        model_ = EEGyeModel(in_chans, n_classes, input_time_length, 
                                     reductionsize=50, if_reduction=True, if_deep=True)

        if cuda:
            model_.cuda()
        
        checkpoint = torch.load(savebase_log+'cnn_model_MI'+'.pth')
        model_.load_state_dict(checkpoint['model_state_dict'])
        
        # freeze
        # ct = 0
        # for child in model_.children():
        #     print(ct)
        #     if ct < 1:
        #         for param in child.parameters():
        #             param.requires_grad = False
        #     ct+=1
       
        # model 
        model_.eval()
        model_set[i][j] = model_
    
# %% Fine-Tuning
batch_size = 1000
n_epoch = 5000

tune_model_set = [[0 for a in range(nCV)] for b in range(5)]

Tlosses_set =[[0 for a in range(nCV)] for b in range(5)]
Taccuracies_set = [[0 for a in range(nCV)] for b in range(5)]
Tauc_set = [[0 for a in range(nCV)] for b in range(5)]

Vlosses_set = [[0 for a in range(nCV)] for b in range(5)]
Vaccuracies_set = [[0 for a in range(nCV)] for b in range(5)]
Vauc_set = [[0 for a in range(nCV)] for b in range(5)]
    
for i in range(0,5):
    for j, cv_ind in enumerate(cv_index[i]):
        model = model_set[i][j]
        loss=[]
        
        
        if cuda:
            model.cuda()
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.9) # 0.5 * 0.001
    
        kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        criterion = FocalLoss(**kwargs)
    
        total_epoch = -1
        Tlosses, Taccuracies, Tauc = [], [], []
        Vlosses, Vaccuracies, Vauc = [], [], []
        highest_acc = 0
        highest_auc = 0
        
        savedir = savebase_log
        savename = savedir + 'cnn_test_model_sub%d_cv%d_ing.pth'%(i+1, j)
        
        start=time.time()
        for i_epoch in range(n_epoch):
            total_epoch += 1
            # Randomize batches ids and get iterater 'i_trials_in_batch'
            i_trials_in_batch = get_balanced_batches(len(train_set[i][j].X), rng, shuffle=True, batch_size=batch_size)
            # Set model to training mode
            model.train()
            for i_trials in i_trials_in_batch:
                # Have to add empty fourth dimension to X for training
                batch_X = train_set[i][j].X[i_trials][:, :, :, None]
                batch_y = train_set[i][j].y[i_trials]
                # convert from nparray tepocho torch tensor
                net_in = np_to_var(batch_X)
                if cuda:
                    net_in = net_in.cuda()
                net_target = np_to_var(batch_y)
                if cuda:
                    net_target = net_target.cuda()
                # Remove gradients of last backward pass from all parameters
                optimizer.zero_grad()
                # Compute outputs of the network
                outputs = model(net_in)
                # Compute the loss
                loss = criterion(outputs.cpu(), torch.from_numpy(batch_y))
                # Do the backpropagation
                loss.backward()
                # Update parameters with the optimizer
                optimizer.step()
            # Set model to evaluation mode
            model.eval()
            print("Sub {:d}, CV {:d}, Epoch {:d}".format(i+1, j, total_epoch))
            average_acc = []
            average_loss = []
            
            # Here we compute training accuracy and validation accuracy of current model
            for setname, dataset in (('Train', train_set[i][j]), ('Valid', valid_set[i][j])):
                i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=False, batch_size=60)
                outputs=None
                for i_trials in i_trials_in_batch:
                    batch_X = dataset.X[i_trials][:, :, :, None]
                    batch_y = dataset.y[i_trials]
                    net_in = np_to_var(batch_X)
                    if cuda:
                        net_in = net_in.cuda()
                    toutputs = model(net_in)
                    if outputs is None:
                        temp = toutputs.cpu()
                        outputs = temp.detach().numpy()
                    else:
                        temp = toutputs.cpu()
                        outputs = np.concatenate((outputs,temp.detach().numpy()))
                loss = criterion(torch.tensor(outputs), torch.tensor(dataset.y))
                print("{:6s} Loss: {:.5f}".format(
                    setname, float(var_to_np(loss))))
                predicted_labels = np.argmax((outputs), axis=1)
                accuracy = np.mean(dataset.y  == predicted_labels)
                # one - hot encoding
                label_one_hot = np.array(tf.one_hot(dataset.y, n_classes))
                auc = roc_auc_score(label_one_hot, outputs,multi_class='ovr')
            
                print("{:6s} Accuracy: {:.1f}%".format(setname, accuracy * 100))
                print("{:6s} AUC: {:.3f}".format(setname, auc))
                
                if setname == 'Train':
                    Tlosses.append(loss)
                    Taccuracies.append(accuracy)
                    Tauc.append(auc)
                    current_Tacc=accuracy
                elif setname == 'Valid':
                    Vlosses.append(loss)
                    Vaccuracies.append(accuracy)
                    Vauc.append(auc)
                    if auc>=highest_auc:
                        torch.save({
                            'in_chans': in_chans,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'n_classes': 3,
                            'input_time_length': window_size
                        }, savename)
        
                        highest_auc=auc
                        print('model saved')
                    if (i_epoch+1)%100 == 0:
                        plot_confusion_matrix(dataset.y, predicted_labels, 
                                              classes=['LH', 'RH', 'Other'], normalize=True,
                                              title='Validation confusion matrix')
                        plt.show()
                else:
                    average_acc.append(accuracy)
                    average_loss.append(accuracy)
        end = time.time()
        
        print('time is {}'.format(end-start))
        tune_model_set[i][j] = model
        Tlosses_set[i][j] = Tlosses
        Taccuracies_set[i][j] = Taccuracies
        Tauc_set[i][j] = Tauc
        Vlosses_set[i][j] = Vlosses
        Vaccuracies_set[i][j] = Vaccuracies
        Vauc_set[i][j] = Vauc
        
    
# %%
for i in range(0,5):
    
    for j, cv_ind in enumerate(cv_index[i]):
        t = np.arange(0.0, len(Tlosses_set[i][j]), 1)+1
        plt.plot(t, Tlosses_set[i][j], 'r', t, Vlosses_set[i][j], 'y')
        plt.legend(('Training loss', 'validation loss'))
        plt.show()
        
        plt.plot(t, Taccuracies_set[i][j], 'r', t, Vaccuracies_set[i][j], 'y')
        plt.legend(('Training accuracy', 'Validation accuracy'))
        plt.show()
        
        plt.plot(t, Tauc_set[i][j], 'r', t, Vauc_set[i][j], 'y')
        plt.legend(('Training AUC', 'Validation AUC'))
        plt.show()
    

# %% test
predicted_labels_list = [[0 for a in range(nCV)] for b in range(5)]

average_acc=[[0 for a in range(nCV)] for b in range(5)]
average_auc=[[0 for a in range(nCV)] for b in range(5)]
average_loss=[[0 for a in range(nCV)] for b in range(5)]
for i in range(0,5):
    print('subject ' + str(i))
    for j, cv_ind in enumerate(cv_index[i]):
        for setname, dataset in (('Valid', valid_set[i][j]), ('Test', test_set[i])):
            
            savename = savedir + 'cnn_test_model_sub%d_cv%d_ing.pth'%(i+1,j)
            if cuda:
                model.cuda()
            checkpoint = torch.load(savename)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            # shuffle=False to make sure it's in the orignial order
            i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=False,
                                                     batch_size=10)
            outputs = None
            for i_trials in i_trials_in_batch:
                # Have to add empty fourth dimension to X
                batch_X = dataset.X[i_trials][:,:,:,None]
                net_in = np_to_var(batch_X)
                if cuda:
                    net_in = net_in.cuda()
                toutputs = model(net_in)
                if outputs is None:
                    temp=toutputs.cpu()
                    outputs=temp.detach().numpy()
                else:
                    temp=toutputs.cpu()
                    outputs=np.concatenate((outputs,temp.detach().numpy()))
            
            
            predicted_labels = np.argmax((outputs), axis=1)
            print(predicted_labels.shape)
            if setname == 'Test':
                # predicted_labels
                predicted_labels_list[i][j] = predicted_labels
                
            else:
                accuracy = np.mean(dataset.y  == predicted_labels)
                average_acc[i][j] = accuracy
                loss = criterion(torch.tensor(outputs), torch.tensor(dataset.y))
                average_loss[i][j] = loss
                # one - hot encoding
                label_one_hot = np.array(tf.one_hot(dataset.y, n_classes))
                auc = roc_auc_score(label_one_hot, outputs,multi_class='ovr')
                average_auc[i][j] = auc
                print("Sub {:d} Loss: {:.5f}".format(
                    i+1, float(var_to_np(loss))))
                print("Sub {:d} Accuracy: {:.1f}".format(i+1, accuracy * 100))
                print("Sub {:d} AUC: {:.3f}".format(i+1, auc))
                plot_confusion_matrix(dataset.y, predicted_labels, 
                                      classes=['LH', 'RH', 'Other'], normalize=True,
                                      title='Validation confusion matrix')
                plt.savefig(savedir+'sub{:d}_cv{:d}_auc{:.3f}.png'.format(i+1, j, auc), dpi=300)
                plt.show()
print(np.mean(np.array(average_acc),axis=1)*100)
print(np.mean(np.array(average_acc))*100)
print(np.mean(np.array(average_auc),axis=1))
print(np.mean(average_auc))


# %% Prediction voting
def mode(pred):
    mode_list = []
    kk=[]
    cv0, cv1, cv2 = pred[0], pred[1], pred[2]
    for i in range(len(cv0)): 
        li = [cv0[i], cv1[i], cv2[i]]
        count, mode = 1, cv1[i]
        k_=1
        for k, x in enumerate(li):
            if li.count(x) > count:
                count = li.count(x)
                mode = x
                k_=k
                # print('%d th answer: %d' %(i,k))
        
        mode_list.append(mode)
        kk.append(k_)
    print(np.squeeze(np.array(kk)))
    return mode_list


final_index = []
for i in range(0,5):
    # select the most common results among three cross-valided models
    final_index.append(mode(predicted_labels_list[i])) 

#%%
MI_label = final_index[0]
for i in range(1,5):
    MI_label = np.concatenate((MI_label,final_index[i]))

print(MI_label.shape)
MI_label = MI_label.astype(int)
MI_label[:100]
#lable 0,1,2

np.savetxt(savedir+'pred_MI_label_ing.txt',MI_label,delimiter=',',fmt="%d")


    
print(collections.Counter(MI_label))


  
