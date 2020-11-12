#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : mixup.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/10 下午3:21
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import torch

__all__ = ['mix_up']


def mix_up(images, labels, alpha=0.2):
    """
    mix up inputs pairs of target and lambda
    :param images:
    :param labels:
    :return:
    """
    if alpha > 0:
        np.random.beta(alpha, alpha)
        beta_distributed = torch.distributions.beta.Beta(alpha, alpha)
        lambda_ = beta_distributed.sample([]).item()
    else:
        lambda_ = 1

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    lambda_ = max(lambda_, 1 - lambda_)

    mixed_images = lambda_ * images + (1 - lambda_) * images[index, :]

    labels_a = labels.long()
    labels_b = labels[index].long()

    return mixed_images, labels_a, labels_b, lambda_


if __name__ == "__main__":

    alpha = 0.2

    beta_distributed = torch.distributions.beta.Beta(alpha, alpha)

    images = torch.randn((10, 3, 224, 224), dtype=torch.float32)
    labels = torch.randint(0, 6, size=(10, ))

    mixed_images, labels_a, labels_b, lambda_ = mix_up(images, labels, alpha)

    print('Done')
