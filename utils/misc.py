#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : misc.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/5 下午7:19
# @ Software   : PyCharm
#-------------------------------------------------------

import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt


_all__ = ['read_class_names', 'plt_imshow', 'plot_image_class', 'accuracy', 'precision']


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def plt_imshow(image, title=None):
    """

    :param image:
    :param title:
    :return:
    """
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)  # clip to (0, 1)
    plt.imshow(image)
    if title is not None:
        plt.title(title)


def plot_image_class(images, labels, index_class):
    """

    :param images:
    :param labels:
    :return:
    """
    labels = [index_class[index] for index in labels.numpy()]

    fig, axs = plt.subplots(6, 6, figsize=(18, 18))
    for n, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(6, 6, n + 1)
        plt_imshow(img, label)
        plt.axis('off')
    plt.show()


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