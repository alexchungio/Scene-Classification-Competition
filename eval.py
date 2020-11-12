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

    def __init__(self, model_path):

        self.model_path = model_path
        self.index_class = index_class
        self.args = args
        self.model = self.build_model().eval()

        self.img_preprocess = self.preprocess()


    def build_model(self):
        # build model
        model = make_model(self.args)

        # load state dict
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).to(device)
            # model = model.to(device)
            model_state = torch.load(self.model_path)
            model.load_state_dict(model_state['state_dict'])
            # model.load_state_dict(model_state['state_dict'], strict=False)
        else:
            model = torch.nn.DataParallel(model)
            model_state = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(model_state['state_dict'])
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
            _, pred_label = torch.max(pred.data, 1)
            result = {'result': self.index_class[pred_label[0].item()]}
        else:
            result = {'result': 'predict score is None'}

        return result

    def postprocess(self):
        pass


def main():

    dataset_dir = '/media/alex/80CA308ECA308288/alex_dataset/scene_classification/test'
    model_infer = Inference(model_path=args.best_checkpoint)

    # image_path = glob(os.path.join(dataset_dir, '*.jpg'))
    image_names = os.listdir(dataset_dir)
    image_names.sort(key=lambda x: int(x[:-4])) # sort image according to index

    start_time = time.perf_counter()

    pbar = tqdm(enumerate(image_names))

    with open(args.test_path, 'w') as fw:

        for i, img_name in pbar:
            img = Image.open(os.path.join(dataset_dir, img_name))
            pre_img = model_infer.img_preprocess(img)
            result = model_infer.inferece({'input_img': pre_img})
            result_str = '{},{}\n'.format(img_name.split('.')[0], result['result'])
            fw.write(result_str)

            pbar.set_description('Processing {}'.format(img_name))

    end_time = time.perf_counter()
    print('Inference cost {} second'.format(int(end_time-start_time)))
    print('Done !')
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
