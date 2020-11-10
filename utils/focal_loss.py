#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : focal_loss.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/10 下午3:20
# @ Software   : PyCharm
#-------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import time
import random
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_focal_loss():
    """

    :return:
    """
    focal_loss = lambda gamma, prop: - np.power(1 - prop, gamma) * np.log(prop)

    probs = np.arange(0.01, 1.0, 0.01)

    gamma_0_loss = focal_loss(0, probs)
    gamma_05_loss = focal_loss(0.5, probs)
    gamma_1_loss = focal_loss(1, probs)
    gamma_2_loss = focal_loss(2, probs)
    gamma_5_loss = focal_loss(5, probs)

    plt.figure(figsize=(12, 6))
    plt.plot(probs, gamma_0_loss, label=r'$\gamma = 0$')
    plt.plot(probs, gamma_05_loss, label=r'$\gamma = 0.5$')
    plt.plot(probs, gamma_1_loss, label=r'$\gamma = 1$')
    plt.plot(probs, gamma_2_loss, label=r'$\gamma = 2$')
    plt.plot(probs, gamma_5_loss, label=r'$\gamma = 5$')
    plt.xlabel('probability of ground truth class')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


class FocalLoss(nn.Module):
    """
    reference https://github.com/clcarwin/focal_loss_pytorch
    """
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(int, float)): self.alpha = torch.tensor([alpha, 1-alpha], device=gamma.device)
        if isinstance(alpha,list): self.alpha = torch.tensor(alpha, device=gamma.device)
        self.reduction = reduction

    def forward(self, pred, target):
        if pred.dim()>2:
            pred = pred.view(pred.size(0), pred.size(1), -1)  # N,C,H,W => N,C,H*W
            pred = pred.transpose(1,2)    # N,C,H*W => N,H*W,C
            pred = pred.contiguous().view(-1, pred.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        pred_softmax = F.softmax(pred)
        pred_softmax = pred_softmax.gather(1,target)
        pred_softmax = pred_softmax.view(-1)
        pt = Variable(pred_softmax.data)

        diff = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            alpha_t = self.alpha.gather(0,target.data.view(-1))
            loss = -1 * alpha_t * diff * pt._log()
        else:
            loss = -1 * diff * pt.log_()

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


def main():
    # plot_focal_loss()
    start_time = time.time()
    maxe = 0
    for i in range(1000):
        data = torch.rand(4, 3).to(device)
        label = torch.tensor([0, 2, 1, 0], dtype=torch.long, device=device)

        loss_1 = FocalLoss(gamma=0)(data, label)
        loss_2 = nn.CrossEntropyLoss()(data, label)
        a = loss_1.data
        b = loss_2.data
        if abs(a - b) > maxe: maxe = abs(a - b)
    print('time:', time.time() - start_time, 'max_error:', maxe)


if __name__ == "__main__":
    main()


