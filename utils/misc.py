#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : misc.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/2 下午7:43
# @ Software   : PyCharm
#-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


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

