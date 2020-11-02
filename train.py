#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/2 上午9:49
# @ Software   : PyCharm
#-------------------------------------------------------

import os

import pathlib
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models

from libs.configs import cfgs
from data.dataset import DataProvider, data_loader
from utils.misc import plt_imshow, plot_image_class


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    dataset_dir = pathlib.Path(cfgs.DATASET_PATH)

    labels_path = list(dataset_dir.glob('./*.csv'))[0]

    train_data_path = os.path.join(cfgs.DATASET_PATH, 'train')
    test_data_path = os.path.join(cfgs.DATASET_PATH, 'test')

    train_loader, val_loader, class_index, index_class = data_loader(train_data_path, labels_path, batch_size=256,
                                                                     split_ratio=0.9)


if __name__ == "__main__":

    main()



