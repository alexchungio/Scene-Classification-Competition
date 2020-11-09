#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tools.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/5 下午7:31
# @ Software   : PyCharm
#-------------------------------------------------------
from __future__ import absolute_import


import os
import time
import torch
import torch.nn as nn
import torch.utils.data  as data
import torch.nn.init as init
import adabound
from utils.radam import RAdam, AdamW


__all__ = ['get_mean_and_std', 'init_params', 'AverageMeter', 'get_optimizer', 'save_checkpoint', 'accuracy', 'precision']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    data_loader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in data_loader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.__div__(len(dataset))
    std.__div__(len(dataset))

    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(model, args):
    parameters = []
    for name, param in model.named_parameters():
        # custom layer execute
        if 'fc' in name or 'class' in name or 'last_linear' in name or 'ca' in name or 'sa' in name:
            parameters.append({'params': param, 'lr': args.lr * args.lr_times})
        else:
            parameters.append({'params': param, 'lr': args.lr})

    if args.optimizer == 'sgd':
        return torch.optim.SGD(parameters,
                            # model.parameters(),
                               args.lr,
                               momentum=args.momentum, nesterov=args.nesterov,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(parameters,
                                # model.parameters(),
                                   args.lr,
                                   alpha=args.alpha,
                                   weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(parameters,
                                # model.parameters(),
                                args.lr,
                                betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'AdaBound':
        return adabound.AdaBound(parameters,
                                # model.parameters(),
                                lr=args.lr, final_lr=args.final_lr)
    elif args.optimizer == 'radam':
        return RAdam(parameters, lr=args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)

    else:
        raise NotImplementedError


def save_checkpoint(state, path):
    """

    :param state:
    :param path:
    :return:
    """
    try:
        print('Saving state at {}'.format(time.ctime()))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
    except Exception as e:
        print('Failed due to {}'.format(e))


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    :param output: [batch_size, num_classes]
    :param target: [batch_size, 1]
    :param topk:
    :return:
    """
    max_k = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, dim=1, largest=True, sorted=True)

    # transpose => (max_k, batch_size)
    pred = pred.t()
    # => [batch_size, max_k]
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.__div__(batch_size))
    return res

def precision(output, target):
    pass


