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
import copy
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.parallel as parallel
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from configs.cfgs import args
from data.dataset import DataProvider
from utils.build_model import make_model
from utils import get_optimizer, accuracy, AverageMeter, save_checkpoint


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=args.summary)


# use cuda
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# set random sees
if args.manual_seed is None:
    args.manual_seed = random.randint(1, 10000)

random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.backends.cudnn.benchmark = True


state = {k: v for k, v in args._get_kwargs()}

global_step = 0

def main():
    # --------------------------------config-------------------------------
    global use_cuda
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    best_acc = 0.0
    # ------------------------------ load dataset---------------------------
    print('==> Loader dataset {}'.format(args.train_data))
    data_provider = DataProvider()
    train_loader = data_provider(args.train_data, args.batch_size, backbone=None, phase='train', num_worker=4)
    val_loader = data_provider(args.val_data, args.batch_size, backbone=None, phase='val', num_worker=4)

    # ---------------------------------model---------------------------------
    model = make_model(args)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()  # load model to cuda
    # show model size
    print('\t Total params volumes: {:.2f} M'.format(sum(param.numel() for param in model.parameters()) / 1000000.0))

    # --------------------------------criterion & optimizer-----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)

    # Resume model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        # for single or multi gpu
        if len(args.gpu_id) > 1:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # eval model
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc_1, test_acc_5 = test(val_loader, model, criterion, use_cuda)
        print(' Test => loss {:.4f} | acc_top1 {:.4f} acc_top5'.format(test_loss, test_acc_1, test_acc_5))

        return None

    # best_model_weights = copy.deepcopy(model.state_dict())
    since = time.time()
    for epoch in range(start_epoch, args.epochs):
        print('Epoch {}/{} | LR {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr']))

        train_loss, train_acc_1, train_acc_5 = train(train_loader, model, criterion, optimizer, args.summary_iter, use_cuda)
        test_loss, test_acc_1, test_acc_5 = test(val_loader, model, criterion, use_cuda)

        scheduler.step(metrics=test_loss)

        # save logs
        writer.add_scalars(main_tag='epoch/loss', tag_scalar_dict={'train': train_loss, 'val': test_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag='epoch/acc_top1', tag_scalar_dict={'train': train_acc_1, 'val': test_acc_1},
                           global_step=epoch)
        writer.add_scalars(main_tag='epoch/acc_top5', tag_scalar_dict={'train': train_acc_5, 'val': test_acc_5},
                           global_step=epoch)

        # add learning_rate to logs
        writer.add_scalar(tag='lr', scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch)

        #-----------------------------save model-----------------------------
        if test_acc_1 > best_acc and epoch>50:
            best_acc = test_acc_1
            # get param state dict
            if len(args.gpu_id) > 1:
                best_model_weights = model.module.state_dict()
            else:
                best_model_weights = model.state_dict()

            state = {
                'epoch': epoch + 1,
                'acc': best_acc,
                'state_dict': best_model_weights,
                'optimizer': optimizer.state_dict()
            }

            save_checkpoint(state, args.checkpoint)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def train(train_loader, model, criterion, optimizer, summary_iter, use_cuda):

    global global_step
    model.train()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    pbar = tqdm(train_loader)
    for inputs, targets in pbar:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # computer output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record
        acc_1, acc_5 = accuracy(outputs.data, target=targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        acc_top1.update(acc_1.item(), inputs.size(0))
        acc_top5.update(acc_5.item(), inputs.size(0))

        if (global_step + 1) % summary_iter == 0:
            writer.add_scalar(tag='train/loss', scalar_value=loss.cpu().item(), global_step=global_step)
            writer.add_scalar(tag='train/acc_top1', scalar_value= acc_1, global_step=global_step)
            writer.add_scalar(tag='train/acc_top5', scalar_value=acc_5, global_step=global_step)

        # grad clearing
        optimizer.zero_grad()
        # computer grad
        loss.backward()
        # update params
        optimizer.step()

        global_step += 1

        pbar.set_description('train loss {0}'.format(loss.item()), refresh=False)

    pbar.write('\ttrain => loss {:.4f} | acc_top1 {:.4f}  acc_top5 {:.4f}'.format(losses.avg, acc_top1.avg, acc_top5.avg))

    return (losses.avg, acc_top1.avg, acc_top5.avg)


def test(eval_loader, model, criterion, use_cuda):

    model.eval()

    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    pbar = tqdm(eval_loader)
    with torch.set_grad_enabled(mode=False):
        for inputs, targets in pbar:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc_1, acc_5 = accuracy(outputs.data, target=targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            acc_top1.update(acc_1.item(), inputs.size(0))
            acc_top5.update(acc_5.item(), inputs.size(0))

            pbar.set_description('eval loss {0}'.format(loss.item()), refresh=False)

    pbar.write('\teval => loss {:.4f} | acc_top1 {:.4f}  acc_top5 {:.4f}'.format(losses.avg, acc_top1.avg, acc_top5.avg))

    return (losses.avg, acc_top1.avg, acc_top5.avg)

if __name__ == "__main__":

    main()