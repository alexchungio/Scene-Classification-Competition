#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : preprocess.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/4 上午9:06
# @ Software   : PyCharm
#-------------------------------------------------------

import os
from glob import glob
import codecs
import random
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from configs.cfgs import args
from utils.misc import read_class_names


base_path = args.dataset

label_files = glob(os.path.join(base_path, '*.csv'))

index_class = read_class_names(args.classes)
class_index = {c:i for i,c in index_class.items()}

def parse_data(label_files):

    images = []
    labels = []
    results = []

    for index, label_file in enumerate(label_files):

        with codecs.open(label_file, 'r', 'utf-8') as fr:
            for line in fr.readlines()[1:]:
                img_name, class_name = line.strip().split(',')

                img_path = os.path.join(base_path, 'train', img_name)
                img_label = class_index[class_name]
                images.append(img_name)
                labels.append(img_label)
                results.append('{},{}'.format(img_path, img_label))

    return images, labels, results


def split_train_val(data, target, n_splits=10, shuffle=True, random_state=2020):
    """

    :param data:
    :param n_splits:
    :param shuffle:
    :param random_state:
    :return:
    """
    split_folds = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    train_data, val_data = [], []
    for train_index,  val_index in split_folds.split(data, target):
        train_data =  list(np.array(data)[train_index])
        val_data = list(np.array(data)[val_index])
        break

    return train_data, val_data


def save_data(data, path):
    with open(path, 'w') as fw:
        for item in data:
            fw.write(item + '\n')

def main():
    images, labels, results = parse_data(label_files)
    train_data, val_data = split_train_val(results, labels, n_splits=10)

    save_data(train_data, args.train_data)
    save_data(val_data, args.val_data)
    print('Done')


if __name__ == "__main__":
    main()




