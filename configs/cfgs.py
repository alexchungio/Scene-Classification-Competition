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

from utils.build_model import model_names


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
parser.add_argument('-c', '--checkpoint', default=os.path.join(ROOT_PATH, 'outputs', 'weights', 'ckpt.pth'), type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='resume the train log info')
# Logs
parser.add_argument('-s', '--summary', default=os.path.join(ROOT_PATH, 'outputs', 'summary'), type=str, metavar='PATH',
                    help='path to save logs (default: logs)')
parser.add_argument('--summary_iter', default=100, type=int, help='number of iterator to save logs (default: 1)')
# Train
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful at restart)')

parser.add_argument('--num_classes', default=6, type=int, metavar='N',
                    help='number of classification of image')
parser.add_argument('--image_size', default=224, type=int, metavar='N',
                    help='train and val image size')
parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                    help='train batch size (default: 256)')


# LR
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate，1e-2， 1e-4, 0.001')

parser.add_argument('--lr_times', '--lr_accelerate_times', default=5, type=int,
                    metavar='LR', help='custom layer lr accelerate times')

parser.add_argument('--schedule', type=int, nargs='+', default=[30, 50, 60],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')


# Optimizer
parser.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam', 'AdaBound', 'radam'], metavar='N',
                         help='optimizer (default=sgd)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--no_nesterov', dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum')

parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')

parser.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for RMSprop (default: 0.99)')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# Architecture
parser.add_argument('--arch', metavar='ARCH', default='resnext101_32x16d_wsl',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnext101_32x8d)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base_width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')

# Misc
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# Device setting
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


if __name__ == "__main__":
    print(args.train_data)