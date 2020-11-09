#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : eval.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/5 上午10:55
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import time
from PIL import Image
from glob import glob
import torch
from tqdm import tqdm
from torchvision.transforms import transforms

from configs.cfgs import args
from utils.misc import read_class_names
from utils.build_model import make_model
from data.transforms import RelativePreservingResize


args.pretrained = False
index_class = read_class_names(args.classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Inference(object):

    def __init__(self):

        self.index_class = index_class
        self.args = args
        self.model = self.build_model().to(device)
        self.img_preprocess = self.preprocess()


    def build_model(self):
        # build model
        model = make_model(self.args)

        # load state dict
        if torch.cuda.is_available():
            model_state = torch.load(args.checkpoint)
            model.load_state_dict(model_state)
            model = model.cuda()
        else:
            model_state = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(model_state)
        return model

    def preprocess(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            RelativePreservingResize(size=(args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        return transform

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.pre_img(img)
                preprocessed_data[k] = img
        return preprocessed_data


    def inferece(self, data):

        img = data['input_img']

        # (C, H, W) => (N, C, H, W)
        img = img.unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            pred = self.model(img)

        if pred is not None:
            _, pred_label = torch.max(   pred.data, 1)
            result = {'result': self.index_class[pred_label[0].item()]}
        else:
            result = {'result': 'predict score is None'}

        return result



    def postprocess(self):
        pass


def main():

    dataset_dir = '/media/alex/80CA308ECA308288/alex_dataset/scene_classification/test'
    model_infer = Inference()

    image_path = glob(os.path.join(dataset_dir, '*.jpg'))
    start_time = time.perf_counter()

    pbar = tqdm(enumerate(image_path))

    with open(args.test_path, 'w') as fw:

        for i, img_path in pbar:
            img = Image.open(img_path)
            pre_img = model_infer.img_preprocess(img)
            result = model_infer.model(pre_img)
            result_str = '{},{}\n'.format(os.path.basename(img_path), result['result'])
            fw.write(result_str)

    end_time = time.perf_counter()
    print('Inference cost {} second'.format(int(end_time-start_time)))
    print('Done !')





if __name__ == "__main__":
    main()