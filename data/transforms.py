#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : transforms.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/4 上午11:56
# @ Software   : PyCharm
#-------------------------------------------------------

import random
from PIL import Image, ImageFilter
from torchvision import transforms

__all__ = ['get_transforms']


class AspectPreservingResize(object):
    def __init__(self, smallest_side, interpolation=Image.BILINEAR):

        """

        :param smallest_side: int
        :param interpolation:
        """
        self.smallest_size = smallest_side
        self.interpolation = interpolation

    def __call__(self, img):

        w, h = img.size
        scale = self.smallest_size / w if w < h else self.smallest_size / h
        w_target, h_target = int(w * scale), int(h * scale)
        img = img.resize((w_target, h_target), self.interpolation)

        return img


class RelativePreservingResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        """

        :param size:  width, height
        :param interpolation:
        """

        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size

        if w / h < ratio:  # padding width
            # target width
            w_target = h * ratio
            w_padding = (w_target - w) // 2
            img = img.crop((-w_padding, 0, w + w_padding, h))
        else: # padding height
            h_target = int(w / ratio)
            h_padding = (h_target - h) // 2
            img = img.crop((0, -h_padding, w, h + h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        # img.show()
        # resize
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


def get_train_transform(mean, std, size):
    """
    Data augmentation and normalization for training
    :param mean:
    :param std:
    :param size:
    :return:
    """
    train_transform = transforms.Compose([
        RelativePreservingResize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        # RandomRotate(15, 0.3),
        # RandomGaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform


def get_test_transform(mean, std, size):
    """
    Just normalization for validation
    :param mean:
    :param std:
    :param size:
    :return:
    """
    test_transform = transforms.Compose([
        RelativePreservingResize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return test_transform


def get_transforms(size, mode='test', backbone=None):

    assert mode in ['train', 'val', 'test']

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if backbone is not None and backbone in ['nasnetmobile']:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    if mode in ['train']:
        transformations =get_train_transform(mean, std, size)
    else:
        transformations = get_test_transform(mean, std, size)

    return transformations



if __name__ == "__main__":
    img_path = '../img/demo_0.jpg'
    img = Image.open(img_path)
    img_size = img.size  # (150, 81)

    aspect_resize = AspectPreservingResize(256)
    img_0 = aspect_resize(img)
    print(img_0.size)  # (474, 256)
    img_0.show()
    relative_resize = RelativePreservingResize((256, 256))
    img_1 = relative_resize(img)
    print(img_1.size) # (256, 256)
    img_1.show()




