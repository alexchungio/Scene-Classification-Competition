#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cfgs.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/4 上午9:00
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import argparse


# Root Path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ROOT_PATH = sys.path(__file__)
print(ROOT_PATH)
print (20*"++--")

# Parse arguments
parser = argparse.ArgumentParser(description= 'PyTorch ImageNet Training')

# Dataset
parser.add_argument('--dataset', default='/media/alex/80CA308ECA308288/alex_dataset/scene_classification', type=str)


parser.add_argument('-train', '--train_data', default=os.path.join(ROOT_PATH, 'data', 'labels', 'train.txt'), type=str) #new_shu_label
parser.add_argument('-val', '--val_data', default=os.path.join(ROOT_PATH, 'data', 'labels', 'val.txt'), type=str)
parser.add_argument('--classes', default=os.path.join(ROOT_PATH, 'data', 'classes', 'scene.names'), type=str)

# Checkpoints


# Train
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_classes', default=6, type=int, metavar='N',
                    help='number of classification of image')
parser.add_argument('--image_size', default=224, type=int, metavar='N',
                    help='train and val image size')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful at restart)')
parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                    help='train batch size (default: 256)')

# Architecture


# Misc


# Device setting

parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


if __name__ == "__main__":
    print(args.train_data)