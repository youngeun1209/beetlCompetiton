import torch.nn as nn
import numpy as np
from braindecode.models.modules import Expression
from braindecode.util import np_to_var
from torch.nn import init
from braindecode.models.functions import safe_log, square
from torch.nn.functional import elu
import torch
def base_model(in_chans, 
               n_filters_time=40,
               n_filters_spat=40,
               filter_time_length=25,
               pool_time_length=75,
               pool_time_stride=15,
               conv_nonlin=square,
               pool_nonlin=safe_log,
               drop_prob=0.5,
               batch_norm=True,
               batch_norm_alpha=0.1,
               pool2d_flag = True):

    fixmodel = CreateBase(in_chans, 
                          n_filters_time,
                          n_filters_spat,
                          filter_time_length,
                          pool_time_length,
                          pool_time_stride,
                          conv_nonlin,
                          pool_nonlin,
                          drop_prob,
                          batch_norm,
                          batch_norm_alpha,
                          pool2d_flag)
    return fixmodel


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)


def _transpose_spat_to_time(x):
    x = x.permute(0, 3, 2, 1)
    return x.contiguous()


def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class CreateBase(nn.Module):
    def __init__(self, in_chans,
                 n_filters_time=40,
                 n_filters_spat=40,
                 filter_time_length=25,
                 pool_time_length=75,
                 pool_time_stride=15,
                 conv_nonlin=square,
                 pool_nonlin=safe_log,
                 drop_prob=0.5,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 pool2d_flag = True):
        super(CreateBase, self).__init__()
        self.pool2d_flag = pool2d_flag
        
        #       block1
        self.activation = {}
        self.conv_time = nn.Conv2d(1, n_filters_time,
                                   (filter_time_length, 1),
                                   stride=1, )
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        init.constant_(self.conv_time.bias, 0)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_spat,
                                   (1, in_chans),
                                   stride=(1, 1),
                                   bias=not batch_norm)
        init.xavier_uniform_(self.conv_spat.weight, gain=1)
        self.bnorm = nn.BatchNorm2d(n_filters_spat,
                                    momentum=batch_norm_alpha,
                                    affine=True)
        init.constant_(self.bnorm.weight, 1)
        init.constant_(self.bnorm.bias, 0)
        self.conv_nonlin = Expression(conv_nonlin)
        if self.pool2d_flag:
            self.pool = nn.AvgPool2d(
                kernel_size=(pool_time_length, 1),
                stride=(pool_time_stride, 1))
        self.pool_nonlin = Expression(pool_nonlin)
        self.drop = nn.Dropout(p=drop_prob)

    def forward(self, data):
        data = self.conv_spat(self.conv_time(_transpose_time_to_spat(data)))
        data = self.conv_nonlin(self.bnorm(data))
        if self.pool2d_flag:
            data = self.pool(data)
        
        data = self.drop(self.pool_nonlin(data))
        return data


class EEGClassifier(nn.Module):

    def __init__(self, n_out_time, filtersize=50, n_classes=4):
        super(EEGClassifier, self).__init__()

        self.conv_class_s = nn.Conv2d(filtersize, n_classes,
                                      (n_out_time, 1), bias=True)
        init.xavier_uniform_(self.conv_class_s.weight, gain=1)
        init.constant_(self.conv_class_s.bias, 0)
        self.softmax_s = nn.LogSoftmax(dim=1)
        self.squeeze_s = Expression(_squeeze_final_output)

    def forward(self, data):
        return self.squeeze_s(self.softmax_s(self.conv_class_s(data)))


class EEGyeModel(nn.Module):
    def __init__(self, in_chans, n_classes, input_time_length, 
                 return_feature=False,reductionsize=50,cat_features=0,
                 if_reduction=True,if_deep=False,
                 n_filters_time=40,
                 n_filters_spat=40,
                 filter_time_length=25,
                 pool_time_length=75,
                 pool_time_stride=15,
                 conv_nonlin=square,
                 pool_nonlin=safe_log,
                 drop_prob=0.5,
                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 pool2d_flag = True):
        
        super(EEGyeModel, self).__init__()
        self.basenet = base_model(in_chans,
                                  n_filters_time,
                                  n_filters_spat,
                                  filter_time_length,
                                  pool_time_length,
                                  pool_time_stride,
                                  conv_nonlin,
                                  pool_nonlin,
                                  drop_prob,
                                  batch_norm,
                                  batch_norm_alpha,
                                  pool2d_flag)
        
        x0 = np_to_var(np.ones(
            (1, in_chans, input_time_length, 1),
            dtype=np.float32))

        x0 = self.basenet(x0)
        n_out = x0.cpu().data.numpy().shape
        filtersize=n_out[1]
        n_out_time = n_out[2]
#         print('feature shape is: ', n_out)
        
        if if_reduction:
            self.feature_reduction = nn.Conv2d(filtersize, reductionsize,
                                      (n_out_time, 1), bias=True)
            init.xavier_uniform_(self.feature_reduction.weight, gain=1)
            init.constant_(self.feature_reduction.bias, 0)
            if if_deep:
                self.deep1 = nn.Conv2d(reductionsize, reductionsize,
                                      (1, 1), bias=True)
                init.xavier_uniform_(self.deep1.weight, gain=1)
                init.constant_(self.deep1.bias, 0)
                
                self.deep2 = nn.Conv2d(reductionsize, reductionsize,
                                      (1, 1), bias=True)
                init.xavier_uniform_(self.deep2.weight, gain=1)
                init.constant_(self.deep2.bias, 0)
                
                self.deep3 = nn.Conv2d(reductionsize, reductionsize,
                                      (1, 1), bias=True)
                init.xavier_uniform_(self.deep3.weight, gain=1)
                init.constant_(self.deep3.bias, 0)
                x0=self.deep3(self.deep2(self.deep1(self.feature_reduction(x0))))
            else:
                x0=self.feature_reduction(x0)
            n_out = x0.cpu().data.numpy().shape
#             print('feature reduction shape is: ', n_out)
            self.classifier = EEGClassifier(n_out_time=1,
                                        n_classes=n_classes,filtersize=reductionsize+cat_features)
        else:
            self.classifier = EEGClassifier(n_out_time=n_out_time,
                                        n_classes=n_classes,filtersize=filtersize+cat_features)
        
        
        self.return_feature = return_feature
        self.if_reduction = if_reduction
        self.if_deep = if_deep
    def forward(self, data,cat_feature=None):
        feature = self.basenet(data)
        
        if self.return_feature:
            return feature
        if self.if_reduction:
            if self.if_deep:
                feature = self.deep3(self.deep2(self.deep1(self.feature_reduction(feature))))
            else:
                feature = self.feature_reduction(feature)
            if cat_feature is not None:
                    feature = torch.cat((feature,cat_feature),1)
            y = self.classifier(feature)
            return y
        else:
            if cat_feature is not None:
                feature = torch.cat((feature,cat_feature),1)
            y = self.classifier(feature)
            return y
      
        
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook