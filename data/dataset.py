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
import pandas as pd
import pathlib
from PIL import Image
import matplotlib.pyplot as plt

import torch

import torch.utils.data as data
from torchvision import datasets, models, transforms

from libs.configs import cfgs
from utils.misc import plt_imshow, plot_image_class


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

    def __init__(self, images, labels=None, split_ratio=0.0):

        self.train_images = images[: int(len(images) * split_ratio)]
        self.val_images = images[int(len(images) * split_ratio):]
        self.train_labels = labels[: int(len(labels) * split_ratio)]
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

    def loader(self, batch_size, phase='train', shuffle=False, num_worker=4):
        assert phase in ['train', 'val']
        if phase == 'train':
            dataset = SceneDataset(self.train_images, self.train_labels, self.data_transforms[phase])
            loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)
        else:
            dataset = SceneDataset(self.val_images, self.val_labels, self.data_transforms[phase])
            loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)

        return loader


def data_loader(image_path, label_path=None, batch_size=256, split_ratio=0.9):

    """

    :param image_path:
    :param label_path:
    :param phase:
    :param batch_size:
    :param split_ratio:
    :return:
    """

    images_labels = pd.read_csv(label_path, sep=',', header=0)

    train_labels = images_labels['label']

    # labels_count = train_labels.value_counts(sort=False)

    category = sorted(set(train_labels.values.tolist()))

    class_index = {c: i for i, c in enumerate(category)}
    index_class = {i: c for i, c in enumerate(category)}

    # shuffle DataFrame

    images = [os.path.join(image_path, img) for img in images_labels['filename'].tolist()]
    labels = [class_index[c] for c in images_labels['label'].tolist()]

    data_provider = DataProvider(images, labels, split_ratio=split_ratio)

    train_loader = data_provider.loader(batch_size=batch_size, phase='train', shuffle=True, num_worker=4)
    val_loader = data_provider.loader(batch_size=batch_size, phase='val', shuffle=False, num_worker=4)

    return train_loader, val_loader, class_index, index_class



def main():
    plt.ion()  # interactive mode

    dataset_dir = pathlib.Path(cfgs.DATASET_PATH)

    labels_path = list(dataset_dir.glob('./*.csv'))[0]

    train_data_path = os.path.join(cfgs.DATASET_PATH, 'train')
    test_data_path = os.path.join(cfgs.DATASET_PATH, 'test')

    train_loader, val_loader, class_index, index_class = data_loader(train_data_path, labels_path, batch_size=256, split_ratio=0.9)

    image_tensor, labels_tensor = next(iter(val_loader))

    plot_image_class(image_tensor[:36], labels_tensor[:36], index_class)



if __name__ == "__main__":
    main()