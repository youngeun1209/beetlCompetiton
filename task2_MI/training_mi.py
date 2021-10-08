#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 23:38:28 2021

@author: yelee
"""

from braindecode.util import set_random_seeds, np_to_var, var_to_np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014001, Cho2017, PhysionetMI
from moabb.paradigms import MotorImagery
import numpy as np
from numpy.random import RandomState
import pickle
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import mne
from mne.decoding import CSP

import sys
import os
__file__ = '/exSSD/projects/beetlCompetition/code'
sys.path.append(__file__)
from util.models import EEGyeModel
from util.utilfunc import get_balanced_batches
from util.preproc import plot_confusion_matrix
from util.focalloss import FocalLoss as FocalLoss

import scipy.io as sio
import os
import numpy as np

import datetime
import tensorflow as tf
from sklearn.metrics import roc_auc_score


cuda = torch.cuda.is_available()
print('gpu: ', cuda)
device = 'cuda' if cuda else 'cpu'


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
rng = RandomState(seed)

# %% pytorch GPU allocation
GPU_NUM = 1 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

# %% create directory
def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise
           
# %%
subj = 1
for dataset in [BNCI2014001(), PhysionetMI(), Cho2017()]:
    data = dataset.get_data(subjects=[subj])
    ds_name = dataset.code
    ds_type = dataset.paradigm
    sess = 'session_T' if ds_name == "001-2014" else 'session_0'
    run = sorted(data[subj][sess])[0]
    ds_ch_names = data[subj][sess][run].info['ch_names']  # [0:22]
    ds_sfreq = data[subj][sess][run].info['sfreq']
    print("{} is an {} dataset, acquired at {} Hz, with {} electrodes\nElectrodes names: ".format(ds_name, ds_type, ds_sfreq, len(ds_ch_names)))
    print(ds_ch_names)
    print()


# %%
ds_src1 = Cho2017()
ds_src2 = PhysionetMI()
ds_tgt = BNCI2014001()

fmin, fmax = 4, 32
raw = ds_tgt.get_data(subjects=[1])[1]['session_T']['run_1']
# tgt_channels = raw.pick_types(eeg=True).ch_names
used_chan = ['Fz','FC1','FC2','C5','C3','C1','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2']

sfreq = 250.
prgm_2classes = MotorImagery(n_classes=2, channels=used_chan, resample=sfreq, fmin=fmin, fmax=fmax)#baseline=(0.0,0.2), 
prgm_4classes = MotorImagery(n_classes=4, channels=used_chan, resample=sfreq, fmin=fmin, fmax=fmax)

sub_list = ds_src1.subject_list
X_src1, label_src1, m_src1 = prgm_2classes.get_data(dataset=ds_src1, subjects=sub_list)
sub_list = ds_src2.subject_list
X_src2, label_src2, m_src2 = prgm_4classes.get_data(dataset=ds_src2, subjects=sub_list)
sub_list = ds_tgt.subject_list
X_tgt, label_tgt, m_tgt = prgm_4classes.get_data(dataset=ds_tgt, subjects=sub_list)

print("First source dataset has {} trials with {} electrodes and {} time samples".format(*X_src1.shape))
print("Second source dataset has {} trials with {} electrodes and {} time samples".format(*X_src2.shape))
print("Target dataset has {} trials with {} electrodes and {} time samples".format(*X_tgt.shape))

print ("\nSource dataset 1 include labels: {}".format(np.unique(label_src1)))
print ("Source dataset 2 include labels: {}".format(np.unique(label_src2)))
print ("Target dataset 1 include labels: {}".format(np.unique(label_tgt)))

# %%
def relabel(l):
    if l == 'left_hand': return 0
    elif l == 'right_hand': return 1
    else: return 2


y_src1 = np.array([relabel(l) for l in label_src1])
y_src2 = np.array([relabel(l) for l in label_src2])
y_tgt = np.array([relabel(l) for l in label_tgt])

print("Only right-/left-hand labels are used and first source dataset does not have other labels:")
print(np.unique(y_src1), np.unique(y_src2), np.unique(y_tgt))

# %% the number of training samples
import collections
print(collections.Counter(y_src1))
print(collections.Counter(y_src2))
print(collections.Counter(y_tgt))

# %% train / valid / test split in same number
# source data   #t0     #t1     #t2
# Tr from src1	3736	3778	
# Tr from src2	2480	2438	4920
# Tr from tgt	704	    704	    2000
# tr	        6920	6920	6920
#
# val from tgt	300	    300	    300
#
# te from src1  1204	1162	0
# te from tgt	292	    292	    292
# te            1496	1454	292

src1_t0 = np.squeeze(np.array(np.where(y_src1==0)))
src1_t1 = np.squeeze(np.array(np.where(y_src1==1)))

tgt_t0 = np.squeeze(np.array(np.where(y_tgt==0)))
tgt_t1 = np.squeeze(np.array(np.where(y_tgt==1)))
tgt_t2 = np.squeeze(np.array(np.where(y_tgt==2)))

tr_scr1_t0 = src1_t0[0:3736]
tr_src1_t1 = src1_t1[0:3778]

te_src1_t0 = src1_t0[3736:]
te_src1_t1 = src1_t1[3778:]

tr_tgt_t0 = tgt_t0[0:704]
tr_tgt_t1 = tgt_t1[0:704]
tr_tgt_t2 = tgt_t2[0:2000]

val_tgt_t0 = tgt_t0[704:1004]
val_tgt_t1 = tgt_t1[704:1004]
val_tgt_t2 = tgt_t2[2000:2300]

te_tgt_t0 = tgt_t0[1004:]
te_tgt_t1 = tgt_t1[1004:]
te_tgt_t2 = tgt_t2[2300:]

tr_src1 = np.concatenate((tr_scr1_t0, tr_src1_t1))
tr_tgt = np.concatenate((tr_tgt_t0, tr_tgt_t1, tr_tgt_t2))

val_tgt = np.concatenate((val_tgt_t0, val_tgt_t1, val_tgt_t2))

te_src1 = np.concatenate((te_src1_t0, te_src1_t1))
te_tgt = np.concatenate((te_tgt_t0, te_tgt_t1, te_tgt_t2))


# %%
window_size = min(X_src1.shape[2], X_src2.shape[2], X_tgt.shape[2]) # 750

X_train = np.concatenate((X_src1[tr_src1, :, :window_size], X_src2[:, :, :window_size], X_tgt[tr_tgt, :, :window_size]))
y_train = np.concatenate((y_src1[tr_src1], y_src2, y_tgt[tr_tgt]))

X_val = X_tgt[val_tgt, :, :window_size]
y_val = y_tgt[val_tgt]

X_test = np.concatenate((X_src1[te_src1, :, :window_size], X_tgt[te_tgt, :, :window_size]))
y_test = np.concatenate((y_src1[te_src1], y_tgt[te_tgt]))

print("Train:  there are {} trials with {} electrodes and {} time samples".format(*X_train.shape))
print("\nValidation: there are {} trials with {} electrodes and {} time samples".format(*X_val.shape))
print("\nTest: there are {} trials with {} electrodes and {} time samples".format(*X_test.shape))

# %% the number of training samples
import collections
print(collections.Counter(y_train))
print(collections.Counter(y_val))
print(collections.Counter(y_test))

# %% CSP
# n_components = 6
# csp = CSP(n_components=n_components, transform_into='csp_space', reg=None, log=None, norm_trace=False)
# csp.fit(X_train, y_train)

# X_train_csp = csp.transform(X_train)
# X_val_csp = csp.transform(X_val)
# X_test_csp = csp.transform(X_test)

# csp_filter = csp.filters_[:n_components]
# csp_mean = csp.mean_
# csp_std = csp.std_

# %%
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

train_set = TrainObject(X_train, y=y_train)
valid_set = TrainObject(X_val, y=y_val)
test_set = TrainObject(X_test, y=y_test)


# %%
input_time_length = X_train.shape[2]
in_chans = train_set.X.shape[1]
window_size = input_time_length
n_classes = len(np.unique(train_set.y))


lr_set = [1e-5] #[0.01, 0.001, 0.0001, 0.00001, 0.000001]

modelset=[]
for lr_ind in range(len(lr_set)):
    model = EEGyeModel(in_chans, n_classes, input_time_length, 
                                 reductionsize=50, if_reduction=True, if_deep=True)
    
    if cuda:
        model.cuda()

    modelset.append(model)
    
# %%
batch_size = 2000
n_epoch = 2000

Tlosses_lr, Taccuracies_lr, Tauc_lr = [], [], []
Vlosses_lr, Vaccuracies_lr, Vauc_lr = [], [], []

savename=[]
savedir=[]

for lr_ind, lr in enumerate(lr_set):
    model = modelset[lr_ind]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.9)
    kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
    criterion = FocalLoss(**kwargs)

    total_epoch = -1
    Tlosses, Taccuracies, Tauc = [], [], []
    Vlosses, Vaccuracies, Vauc = [], [], []
    highest_acc = 0
    highest_auc = 0
    
    savedir.append('/exSSD/projects/beetlCompetition/learning_logs/task2_MI/check_final/')
    makedirs(savedir[lr_ind])
    savename.append(savedir[lr_ind] + 'cnn_model_MI.pth')
    
    
    start=time.time()
    for i_epoch in range(n_epoch):
        total_epoch += 1
        # Randomize batches ids and get iterater 'i_trials_in_batch'
        i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True,
                                                 batch_size=batch_size)
        # Set model to training mode
        model.train()
        for i_trials in i_trials_in_batch:
            # Have to add empty fourth dimension to X for training
            batch_X = train_set.X[i_trials][:, :, :, None]
            batch_y = train_set.y[i_trials]
            # convert from nparray to torch tensor
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
        print("lr {:f}, Epoch {:d}".format(lr,total_epoch))
        average_acc = []
        average_loss = []
    
        # Here we compute training accuracy and validation accuracy of current model
        for setname, dataset in (('Train', train_set), ('Valid', valid_set)):
            i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=False,
                                                     batch_size=200)
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
                    }, savename[lr_ind])
    
                    highest_auc=auc
                    print('model saved')
            else:
                average_acc.append(accuracy)
                average_loss.append(loss)
    end = time.time()
    
    print('time is {}'.format(end-start))
    Tlosses_lr.append(Tlosses)
    Taccuracies_lr.append(Taccuracies)
    Tauc_lr.append(Tauc)

    Vlosses_lr.append(Vlosses)
    Vaccuracies_lr.append(Vaccuracies)
    Vauc_lr.append(Vauc)
   
    modelset[lr_ind] = model
    
# %%
for lr_ind, lr in enumerate(lr_set):
    t = np.arange(0.0, len(Tlosses_lr[lr_ind]), 1)+1
    plt.plot(t, Tlosses_lr[lr_ind], 'r', t, Vlosses_lr[lr_ind], 'y')
    plt.legend(('Training loss', 'validation loss'))
    plt.show()
    
    plt.plot(t, Taccuracies_lr[lr_ind], 'r', t, Vaccuracies_lr[lr_ind], 'y')
    plt.legend(('Training accuracy', 'Validation accuracy'))
    plt.show()
    
    
    plt.plot(t, Tauc_lr[lr_ind], 'r', t, Vauc_lr[lr_ind], 'y')
    plt.legend(('Training auc', 'Validation auc'))
    plt.show()

# %%plot together
t = np.arange(0.0, len(Tlosses_lr[0]), 1)+1
for lr_ind, lr in enumerate(lr_set):
    plt.plot(t, Tlosses_lr[lr_ind])
plt.legend((lr_set))
plt.title(('Training loss'))
plt.show()

for lr_ind, lr in enumerate(lr_set):
    plt.plot(t, Vlosses_lr[lr_ind])
plt.legend((lr_set))
plt.title(('Validation loss'))
plt.show()

for lr_ind, lr in enumerate(lr_set):
    plt.plot(t, Taccuracies_lr[lr_ind])
plt.legend((lr_set))
plt.title(('Training accuracy'))
plt.show()

for lr_ind, lr in enumerate(lr_set):
    plt.plot(t, Vaccuracies_lr[lr_ind])
plt.legend((lr_set))
plt.title(('Validation accuracy'))
plt.show()

for lr_ind, lr in enumerate(lr_set):
    plt.plot(t, Tauc_lr[lr_ind])
plt.legend((lr_set))
plt.title(('Training auc'))
plt.show()

for lr_ind, lr in enumerate(lr_set):
    plt.plot(t, Vauc_lr[lr_ind])
plt.legend((lr_set))
plt.title(('Validation auc'))
plt.show()
    
# %%
for lr_ind, lr in enumerate(lr_set):   
    model = modelset[lr_ind]
    if cuda:
        model.cuda()
    checkpoint = torch.load(savename[lr_ind])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    average_acc, average_loss, average_auc = [], [], []
    setname = 'testset'
    dataset = test_set
    
    i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=False,
                                             batch_size=30)
    outputs=None
    for i_trials in i_trials_in_batch:
        # Have to add empty fourth dimension to X
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
    print("lr: {:f}, {:6s} Loss: {:.5f}".format(lr,setname, float(var_to_np(loss))))
    predicted_labels = np.argmax((outputs), axis=1)
    accuracy = np.mean(dataset.y  == predicted_labels)
    # one - hot encoding
    label_one_hot = np.array(tf.one_hot(dataset.y, n_classes))
    auc = roc_auc_score(label_one_hot, outputs,multi_class='ovr')
    print("lr: {:f}, {:6s} Accuracy: {:.1f}%".format(lr, setname, accuracy * 100))
    print("lr: {:f}, {:6s} AUC: {:.3f}".format(lr, setname, auc))
    plot_confusion_matrix(dataset.y, predicted_labels,
                          classes=['LH','RH','Other'], normalize=True,
                          title='Validation confusion matrix')
    plt.savefig(savedir[lr_ind]+'Training_test_lr_{:f}_auc{:.3f}.png'.format(lr,auc), dpi=300)
    plt.show()


# %% variable save
for lr_ind, lr in enumerate(lr_set):
    var_file = savedir[lr_ind] + 'losses.pth'
    
    torch.save({'Tlosses_lr': Tlosses_lr,
                'Taccuracies_lr': Taccuracies_lr,
                'Tauc_lr': Tauc_lr,
                'Vlosses_lr': Vlosses_lr,
                'Vaccuracies_lr': Vaccuracies_lr,
                'Vauc_lr': Vauc_lr,
                }, var_file)
