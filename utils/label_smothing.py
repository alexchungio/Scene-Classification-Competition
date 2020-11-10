#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : label_smothing.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/10 上午10:46
# @ Software   : PyCharm
#-------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSR(nn.Module):

    def __init__(self, epsilon=0.1, reduction='mean'):
        super(LSR, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.epsilon = epsilon
        self.reduction = reduction

    def _onehot(self, labels, classes, value=1):
        """
            convert labels to one hot vectors
        :param labels:
        :param classes:
        :param value:
        :return:
        """
        # (N, classes)
        one_hot = torch.zeros(labels.size(0), classes)

        # (N,) => (N,1)
        index = labels.view(labels.size(0), -1)
        # (N, 1)
        value_add = torch.ones(labels.size(0), 1).fill_(value)

        # remove to same device
        one_hot = one_hot.to(labels.device)
        value_add = value_add.to(labels.device)

        # scatter add operation
        one_hot.scatter_add_(dim=1, index=index, src=value_add)

        # one_hot = F.one_hot(labels, num_classes=5)
        return one_hot

    def _smooth_label(self, target, depth, factor):
        """
        label smooth
        :param target:
        :param depth:
        :param factor:
        :return:
        """
        one_hot = self._onehot(target, classes=depth, value= 1 - factor)
        one_hot += factor / depth

        return one_hot


    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                             .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))

        x = self.log_softmax(x)
        smoothing_target = self._smooth_label(target, depth=x.size(1), factor=self.epsilon)

        # cross entropy loss
        loss = torch.sum(- x * smoothing_target, dim=1)

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)

        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')