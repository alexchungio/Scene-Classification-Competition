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
import time
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler

from libs.configs import cfgs
from data.dataset import DataProvider, data_loader
from utils.misc import plt_imshow, plot_image_class

from tqdm import tqdm



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_epochs = 50
num_classes = 6

def main():


    dataset_dir = pathlib.Path(cfgs.DATASET_PATH)

    labels_path = list(dataset_dir.glob('./*.csv'))[0]

    train_data_path = os.path.join(cfgs.DATASET_PATH, 'train')
    test_data_path = os.path.join(cfgs.DATASET_PATH, 'test')

    train_loader, val_loader, class_index, index_class = data_loader(train_data_path, labels_path, batch_size=64,
                                                                     split_ratio=0.9)


    model = models.squeezenet1_1(pretrained=True)
    # num_fc_features = models.resnet18(pretrained=True)
    in_channels = model.classifier[1].in_channels

    # update model
    # Sequential(
    #   (0): Dropout(p=0.5, inplace=False)
    #   (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
    #   (2): ReLU(inplace=True)
    #   (3): AdaptiveAvgPool2d(output_size=(1, 1))
    # )
    model.classifier[1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        test_loss, test_acc = test(train_loader, model, criterion)
        scheduler.step(epoch=epoch)

        if best_acc < test_acc:
            best_acc = max(best_acc, test_acc)
            best_model_weights = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


def train(train_loader, model, criterion, optimizer):

    sum_acc, sum_loss, num_samples = 0, 0.0, 0.0

    pbar = tqdm(train_loader)
    for inputs, targets in pbar:
        inputs,targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, targets)
        # grad clearing
        optimizer.zero_grad()
        # computer grad
        loss.backward()
        # update params
        optimizer.step()

        sum_acc += torch.sum(preds == targets.data)
        sum_loss += loss.item() * inputs.size(0)

        num_samples += inputs.size(0)

        pbar.set_description('train loss {0}'.format(loss.item()), refresh=False)

    epoch_acc = sum_acc / num_samples
    epoch_loss = sum_loss / num_samples

    pbar.write('\ttrain => loss {:.4f}, acc {:.4f}'.format(epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc


def test(eval_loader, model, criterion):
    sum_acc, sum_loss, num_samples = 0, 0.0, 0.0

    pbar = tqdm(eval_loader)
    with torch.set_grad_enabled(mode=False):
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, targets)

            sum_acc += torch.sum(preds == targets.data)
            sum_loss += loss.item() * inputs.size(0)

            num_samples += inputs.size(0)

            pbar.set_description('train loss {0}'.format(loss.item()), refresh=False)

    epoch_acc = sum_acc / num_samples
    epoch_loss = sum_loss / num_samples
    pbar.write('\teval => loss {:.4f}, acc {:.4f}'.format(epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc

if __name__ == "__main__":

    main()



