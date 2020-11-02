import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(orig_images_path, hazy_images_path):
    train_list = []
    val_list = []

    # image_list_haze = glob.glob(hazy_images_path + "*")																	#路径匹配
    # for i3-SSIM-室外合成 in range(len(image_list_haze)):
    # 	image_list_haze[i3-SSIM-室外合成]=image_list_haze[i3-SSIM-室外合成].replace("\\","/")
    image_list_haze_index = os.listdir(hazy_images_path)  # 文件名
    image_dataset = []
    for i in image_list_haze_index:  # 添加路径，并组合为元组
        image_dataset.append((orig_images_path + i, hazy_images_path + i))
    train_list = image_dataset[:8000]
    val_list = image_dataset[12000:12100]

    return train_list, val_list


class dehazing_loader(data.Dataset):

    def __init__(self, orig_images_path, hazy_images_path, mode='train'):

        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path)

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_orig_path, data_hazy_path = self.data_list[index]

        data_orig = Image.open(data_orig_path)
        data_hazy = Image.open(data_hazy_path)

        data_orig = data_orig.resize((640, 480), Image.ANTIALIAS)
        data_hazy = data_hazy.resize((640, 480), Image.ANTIALIAS)

        data_orig = (np.asarray(data_orig) / 255.0)
        data_hazy = (np.asarray(data_hazy) / 255.0)

        data_orig = torch.from_numpy(data_orig).float()
        data_hazy = torch.from_numpy(data_hazy).float()

        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

