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
from PIL import Image
import matplotlib.pyplot as plt

import torch

import torch.utils.data as data
from torchvision import datasets, models, transforms

from configs.cfgs import args
from utils.misc import plt_imshow, plot_image_class
from data.transforms import get_transforms
from utils.misc import read_class_names


index_class = read_class_names(args.classes)
class_index = {c:i for i,c in index_class.items()}


class SceneDataset(data.Dataset):

    def __init__(self, root, transforms=None, target_transform=None):
        super(SceneDataset, self).__init__()

        assert '.txt' in root, "invalid data path"

        self.data = list(open(root))

        self.transforms = transforms
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index].strip().split(',')

        # doing this so that it is consistent with all other datasets

        # to return a PIL Image
        try:
            img = self.pil_loader(img_path)
            target = int(target)
        except:
            print('Corrupted due to cannot read {}'.format(img_path))
            return self[index+1]

        if self.transforms is not None:
            img = self.transforms(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class DataProvider(object):

    def __init__(self):
        pass

    def __call__(self, data_path, batch_size, backbone=None, phase='test', num_worker=4):

        assert phase in ['train', 'val', 'test']

        transforms = get_transforms(args.image_size, mode=phase, backbone=backbone)
        dataset = SceneDataset(data_path, transforms)
        if phase == 'train':
            loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        else:
            loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

        return loader


# def data_loader(image_path, label_path=None, batch_size=256, split_ratio=0.9):
#
#     """
#
#     :param image_path:
#     :param label_path:
#     :param phase:
#     :param batch_size:
#     :param split_ratio:
#     :return:
#     """
#
#     images_labels = pd.read_csv(label_path, sep=',', header=0)
#
#     train_labels = images_labels['label']
#
#     # labels_count = train_labels.value_counts(sort=False)
#
#     category = sorted(set(train_labels.values.tolist()))
#
#     class_index = {c: i for i, c in enumerate(category)}
#     index_class = {i: c for i, c in enumerate(category)}
#
#     # shuffle DataFrame
#
#     images = [os.path.join(image_path, img) for img in images_labels['filename'].tolist()]
#     labels = [class_index[c] for c in images_labels['label'].tolist()]




def main():
    # plt.ion()  # interactive mode

    data_provider = DataProvider()

    train_loader = data_provider(args.train_data, args.batch_size, backbone=None, phase='train', num_worker=4)
    eval_loader = data_provider(args.val_data, args.batch_size, backbone=None, phase='val', num_worker=4)

    images, labels = next(iter(eval_loader))

    plot_image_class(images[:36], labels[:36], index_class=index_class)




if __name__ == "__main__":
    main()