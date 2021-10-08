#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 00:10:41 2021

@author: yelee
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:17:58 2021

@author: yelee
"""

from braindecode.util import set_random_seeds
from braindecode.util import np_to_var, var_to_np
import matplotlib.pyplot as plt 
import numpy as np
from numpy.random import RandomState
import os.path as osp
import pickle
import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch import optim
import torch.nn.functional as F

import argparse
import sys
import os
__file__ = '/exSSD/projects/beetlCompetition/code'
sys.path.append(__file__)
from util.dsn import DeepSleepNet
from util.utilfunc import get_balanced_batches
from util.preproc import plot_confusion_matrix
from util.focalloss import FocalLoss as FocalLoss

import tensorflow as tf
# from EEGModels import EEGNet, ShallowConvNet, DeepConvNet

cuda = torch.cuda.is_available()
print('gpu: ', cuda)
device = 'cuda' if cuda else 'cpu'


seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
rng = RandomState(seed)
set_random_seeds(seed=seed, cuda=cuda)

# %% pytorch GPU allocation
GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check


# %% Load data
# Directly download data and indicate their location:
savebase = '/exSSD/projects/beetlCompetition/data/'

# source
sourcebase = savebase + 'SleepSource/'

with open(osp.join(sourcebase, "headerInfo.npy"), 'rb') as f:
        info_source = pickle.load(f)

X_source, y_source = [], []
y_sub = []
subN = 0

for i_sub in range(39):
    for i_record in [1,2]:
        with open(osp.join(sourcebase, "training_s%dr%dX.npy" %(i_sub,i_record)), 'rb') as f:
            X = pickle.load(f)
            X_source.append(X)
        with open(osp.join(sourcebase, "training_s%dr%dy.npy" %(i_sub,i_record)), 'rb') as f:
            y = pickle.load(f)
            y_source.append(y)
            y_sub.append(np.tile(i_sub,len(y)))
X_source = np.concatenate(X_source)
y_source = np.concatenate(y_source)
y_sub = np.concatenate(y_sub)
print("Source: there are {} trials with {} electrodes and {} time samples".format(*X_source.shape))

# %% phase 1 target data use
# target - s0r1, s0r2, s1r1, s1r2, s2r1, s2r2, s3r1, s3r2, s4r1, s4r2, s5r1, s5r2
targetbase = savebase + 'leaderboardSleep/sleep_target/'

with open(osp.join(targetbase, "headerInfo.npy"), 'rb') as f:
    info_target = pickle.load(f)
    
X_target, y_target = [], []
for i_sub in [0,1,4,5]:
    for i_record in [1]:
        with open(osp.join(targetbase, "leaderboard_s%dr%dX.npy" %(i_sub,i_record)), 'rb') as f:
            X_target.append(pickle.load(f))
        with open(osp.join(targetbase, "leaderboard_s%dr%dy.npy" %(i_sub,i_record)), 'rb') as f:
            y_target.append(pickle.load(f))
for i_sub in [2,3]:
    for i_record in [1,2]:
        with open(osp.join(targetbase, "leaderboard_s%dr%dX.npy" %(i_sub,i_record)), 'rb') as f:
            X_target.append(pickle.load(f))
        with open(osp.join(targetbase, "leaderboard_s%dr%dy.npy" %(i_sub,i_record)), 'rb') as f:
            y_target.append(pickle.load(f))
X_target = np.concatenate(X_target)
y_target = np.concatenate(y_target)
print("Target: there are {} trials with {} electrodes and {} time samples".format(*X_target.shape))

print("Combining source and target for training:")

X_val, y_val = [], []
for i_sub in [4,5]:
    for i_record in [2]:
        with open(osp.join(targetbase, "leaderboard_s%dr%dX.npy" %(i_sub,i_record)), 'rb') as f:
            X_val.append(pickle.load(f))
        with open(osp.join(targetbase, "leaderboard_s%dr%dy.npy" %(i_sub,i_record)), 'rb') as f:
            y_val.append(pickle.load(f))
X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)
print("\nValidation: there are {} trials with {} electrodes and {} time samples".format(*X_val.shape))

X_test, y_test = [], []
for i_sub in [0,1]:
    for i_record in [2]:
        with open(osp.join(targetbase, "leaderboard_s%dr%dX.npy" %(i_sub,i_record)), 'rb') as f:
            X_test.append(pickle.load(f))
        with open(osp.join(targetbase, "leaderboard_s%dr%dy.npy" %(i_sub,i_record)), 'rb') as f:
            y_test.append(pickle.load(f))
X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)
print("\nTest: there are {} trials with {} electrodes and {} time samples".format(*X_test.shape))

# %% combine
X_train = np.concatenate([X_source, X_target])
y_train = np.concatenate([y_source, y_target])
print("Combining source and target for training:")
print("Train:  there are {} trials with {} electrodes and {} time samples".format(*X_train.shape))


# % the number of training samples
import collections
print(collections.Counter(y_train))
print(collections.Counter(y_val))
print(collections.Counter(y_test))

# %%
class TrainObject(object):
    def __init__(self, X, y):
        assert len(X) == len(y)
        mean = np.mean(X, axis=2, keepdims=True)
        # Here normalise across the window, when channel size is not large enough
        # In motor imagery kit, we put axis = 1, across channel as an example
        std = np.std(X, axis=2, keepdims=True)
        X = (X - mean) / std
        # we scale it to 1000 as a better training scale of the shallow CNN
        # according to the orignal work of the paper referenced above
        self.X = X.astype(np.float32)*1e3
        self.y = y.astype(np.int64)

train_set = TrainObject(X_train, y=y_train)
valid_set = TrainObject(X_val, y=y_val)
test_set = TrainObject(X_test, y=y_test)



# %%
input_time_length = X_train.shape[2]
in_chans = X_train.shape[1]
n_classes = len(np.unique(y_train))

# %% DeepSleepNet

model = DeepSleepNet(Fs = 100, ch=in_chans, nclass=n_classes)
model.cuda()
# loss
parser = argparse.ArgumentParser(description="Feature Mearusement")
parser.add_argument("--lr",default=0.05, type=float, help="learning rate")
# parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
args = parser.parse_args()

criterion = nn.CrossEntropyLoss()
# criterion = seq_cel if args.loss == 'ce' else gdl

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.5*1e-3)
# optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.5*1e-3)

lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.1)



# %%
savebase_log='/exSSD/projects/beetlCompetition/learning_logs/task1_Sleep/check_final3/'

savename = savebase_log + 'DSN_model_sleep.pth'

total_epoch = -1
Tlosses, Taccuracies = [], []
Vlosses, Vaccuracies = [], []
highest_acc = 0

start = time.time()
batch_size = 2000
n_epoch = 2000

for i_epoch in range(n_epoch):
    total_epoch += 1
    # Randomize batches ids and get iterater 'i_trials_in_batch'
    i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True, batch_size=batch_size)
    # Set model to training mode
    model.train()
    for i_trials in i_trials_in_batch:
        # Have to add empty fourth dimension to X for training
        batch_X = train_set.X[i_trials]
        batch_y = train_set.y[i_trials]
        # convert from ndarray to torch tensor
        net_in = np_to_var(batch_X)
        net_target = np_to_var(F.one_hot(torch.tensor(batch_y), num_classes=n_classes))
        if cuda:
            net_in = net_in.cuda()
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
    print("Epoch {:d}".format(total_epoch))
    average_acc, average_loss = [], []
    
    # here we compute training accuracy and validation accuracy of current model
    for setname, dataset in (('Train', train_set), ('Valid', valid_set)):
        i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=False, batch_size=60)
        outputs=None
        for i_trials in i_trials_in_batch:
            batch_X = dataset.X[i_trials]
            batch_y = dataset.y[i_trials]
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
        loss = criterion(torch.tensor(outputs), torch.tensor(dataset.y))
        print("{:6s} Loss: {:.5f}".format(setname, float(var_to_np(loss))))
        predicted_labels = np.argmax((outputs), axis=1)
        accuracy = np.mean(dataset.y  == predicted_labels)
        print("{:6s} Accuracy: {:.1f}%".format(setname, accuracy * 100))
        
        if setname == 'Train':
            Tlosses.append(loss)
            Taccuracies.append(accuracy)
            current_Tacc=accuracy
        elif setname == 'Valid':
            Vlosses.append(loss)
            Vaccuracies.append(accuracy)
            if accuracy>=highest_acc:
                torch.save({
                    'in_chans': in_chans,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'n_classes': 6,
                    'input_time_length':input_time_length
                }, savename)

                highest_acc=accuracy
                plot_confusion_matrix(dataset.y, predicted_labels, classes=['0', '1', '2', '3', '4', '5'], 
                                      normalize=True, title='Validation confusion matrix')
                plt.show()
        else:
            average_acc.append(accuracy)
            average_loss.append(accuracy)
end=time.time()

print('time is {}'.format(end-start))

# %%
t = np.arange(0., len(Tlosses), 1)+1
plt.plot(t, Tlosses, 'r', t, Vlosses, 'y')
plt.legend(('training loss', 'validation loss'))
plt.savefig(savebase_log+'Training_test_loss_acc{:.3f}.png'.format(accuracy), dpi=300)
plt.show()

plt.figure()
plt.plot(t, Taccuracies, 'r', t, Vaccuracies, 'y')
plt.legend(('training accuracy', 'validation accuracy'))
plt.savefig(savebase_log+'Training_test_accuracy_acc{:.3f}.png'.format(accuracy), dpi=300)
plt.show()


# %%
input_time_length = X_train.shape[2]
in_chans = X_train.shape[1]

if cuda:
    model.cuda()
checkpoint = torch.load(savename)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

average_acc, average_loss = [], []
setname = 'testset'
dataset = test_set

i_trials_in_batch = get_balanced_batches(len(dataset.X), rng, shuffle=False, batch_size=30)
outputs=None
for i_trials in i_trials_in_batch:
    # Have to add empty fourth dimension to X
    batch_X = dataset.X[i_trials]
    batch_y = dataset.y[i_trials]
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

loss = criterion(torch.tensor(outputs), torch.tensor(dataset.y))
  
print("{:6s} Loss: {:.5f}".format(
    setname, float(var_to_np(loss))))
predicted_labels = np.argmax((outputs), axis=1)
accuracy = np.mean(dataset.y  == predicted_labels)

print("{:6s} Accuracy: {:.1f}%".format(setname, accuracy * 100))
plot_confusion_matrix(dataset.y, predicted_labels, classes=['0', '1', '2', '3', '4', '5'], 
                      normalize=True, title='Validation confusion matrix')

plt.savefig(savebase_log+'Training_test_accuracy{:.3f}.png'.format(accuracy), dpi=300)
plt.show()

