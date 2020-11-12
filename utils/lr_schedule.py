#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : lr_schedule.py
# @ Description:  https://github.com/Tony-Y/pytorch_warmup/tree/master
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/11 上午11:51
# @ Software   : PyCharm
#-------------------------------------------------------


import torch
import torch.optim.lr_scheduler as lr_scheduler



def main():

    p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32), requires_grad=True)
    p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32), requires_grad=True)

    params = [{'params': [p1]},
              {'params': [p2], 'lr': 0.1}]

    optimizer = torch.optim.SGD(params=params, lr=0.5)

    lambda_1 = lambda epoch: epoch / 10
    lambda_2 = lambda epoch: 0.9 ** epoch

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = [lambda_1, lambda_2])

    for epoch in range(1, 11):
        lr = [x['lr'] for x in optimizer.param_groups]
        print(f'{epoch} {lr}')
        # if step < 5:
        #     self.assertAlmostEqual(lr[0], 0.5 * step / 5)
        #     self.assertAlmostEqual(lr[1], 0.1 * step / 5)
        # else:
        #     self.assertAlmostEqual(lr[0], 0.5)
        #     self.assertAlmostEqual(lr[1], 0.1)
        optimizer.zero_grad()
        optimizer.step()
        scheduler.step(epoch=epoch)


if __name__ == "__main__":
    main()