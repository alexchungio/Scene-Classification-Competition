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
import glob
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, models, transforms

from libs.configs import cfgs
from data.dataset import DataProvider
from utils.misc import plt_imshow, plot_image_class


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def main():
    plt.ion()  # interactive mode


    dataset_dir = pathlib.Path(cfgs.DATASET_PATH)

    labels_path = list(dataset_dir.glob('./*.csv'))[0]

    train_data_path = os.path.join(cfgs.DATASET_PATH, 'train')
    test_data_path = os.path.join(cfgs.DATASET_PATH, 'test')

    images_labels = pd.read_csv(labels_path, sep=',', header=0)

    train_labels = images_labels['label']

    labels_count = train_labels.value_counts(sort=False)


    category = sorted(set(train_labels.values.tolist()))

    class_index = {c:i for i, c in enumerate(category)}
    index_class = {i:c for i, c in enumerate(category)}


    # shuffle DataFrame

    train_val_images = [os.path.join(train_data_path, img) for img in images_labels['filename'].tolist()]
    train_val_labels =  [class_index[c] for c in images_labels['label'].tolist()]

    data_loader = DataProvider(train_val_images, train_val_labels, split_ratio=0.9)

    train_loader = data_loader.data_loader(batch_size=256, phase='train', shuffle=True, num_worker=4)
    val_loader = data_loader.data_loader(batch_size=256, phase='val', shuffle=False, num_worker=4)

    image_tensor, labels_tensor = next(iter(val_loader))

    plot_image_class(image_tensor[:36], labels_tensor[:36], index_class)



if __name__ == "__main__":
    main()
