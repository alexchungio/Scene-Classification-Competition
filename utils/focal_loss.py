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
import numpy as np
import matplotlib.pyplot as plt



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



def main():
    plot_focal_loss()



if __name__ == "__main__":
    main()

