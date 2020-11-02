#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dataset.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/2 下午8:00
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import glob
import pathlib
from PIL import Image
import matplotlib.pyplot as plt

import torch

import torch.utils.data as data
from torchvision import datasets, models, transforms


class SceneDataset(data.Dataset):

    def __init__(self, data, target, transforms=None):
        super(SceneDataset, self).__init__()

        self.data = data
        self.target = target
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.pil_loader(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class DataProvider(object):

    def __init__(self, images, labels, split_ratio):

        self.train_images = images[: int(len(images) * split_ratio)]
        self.train_labels = labels[: int(len(labels) * split_ratio)]
        self.val_images = images[int(len(images) * split_ratio):]
        self.val_labels = labels[int(len(labels) * split_ratio):]

        # Data augmentation and normalization for training
        # Just normalization for validation
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def data_loader(self, batch_size, phase='train', shuffle=False, num_worker=4):
        assert phase in ['train', 'val']
        if phase == 'train':
            dataset = SceneDataset(self.train_images, self.train_labels, self.data_transforms[phase])
            loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)
        else:
            dataset = SceneDataset(self.val_images, self.val_labels, self.data_transforms[phase])
            loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)

        return loader