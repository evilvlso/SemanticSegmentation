#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LoadThroat.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-09-23 10:39   tango      1.0         None
"""
# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LoadCVC.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-09-14 09:37   tango      1.0         None
"""
import os
import cv2
import imageio
import numpy as np
from PIL import Image
import torch
from einops import rearrange
from torch.utils import data
from torchvision import transforms


# 1060*1075
BasePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

imageValDataPath = os.path.join(BasePath, "SegDatasets/CT/seg/val/img")
masksValDataPath = os.path.join(BasePath, "SegDatasets/CT/seg/val/mask")
imageTrainDataPath = os.path.join(BasePath, "SegDatasets/CT/seg/train/img")
masksTrainDataPath = os.path.join(BasePath, "SegDatasets/CT/seg/train/mask")
# imageValDataPath = os.path.join(BasePath, "SegDatasets/Throat/special/imgs")
# masksValDataPath = os.path.join(BasePath, "SegDatasets/Throat/special/mask")

class DataSetsCT(data.Dataset):
    def __init__(self, kind, image_size=512, num_classes=2):
        """

        :param data:
        """
        super(DataSetsCT, self).__init__()
        if kind == "train":
            self.X = [os.path.join(imageTrainDataPath, i) for i in os.listdir(imageTrainDataPath) if i.endswith("npy")]
            self.Y = [os.path.join(masksTrainDataPath, i.replace("img","mask")) for i in os.listdir(imageTrainDataPath)]
        else:
            self.X = [os.path.join(imageValDataPath, i) for i in os.listdir(imageValDataPath) if i.endswith("npy")]
            self.Y = [os.path.join(masksValDataPath, i.replace("img","mask")) for i in os.listdir(masksValDataPath)]
        self.image_size = image_size
        self.num_classes = num_classes
        self.kind = kind

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """
        训练集和测试集都使用它
        :param item:
        :return:
        """
        x = np.load(self.X[index])
        x= np.expand_dims(x,0)
        y = np.load(self.Y[index])
        y_onehot = np.eye(self.num_classes)[y.reshape([-1])]
        y_onehot = rearrange(y_onehot, "(h w) c->c h w", h=self.image_size)
        return self.X[index], torch.Tensor(x),torch.Tensor(y), torch.Tensor(y_onehot)


def getCVCLoader(kind, **kwargs):
    d = DataSetsCT(kind=kind)
    return data.DataLoader(d, **kwargs)

if __name__ == '__main__':
    d = DataSetsCT(kind="train")
    from torch.utils import data

    dl = data.DataLoader(d, batch_size=4, shuffle=True, num_workers=0, drop_last=True,collate_fn=None)#
    for _, x, y, y_onehot in dl:
        print(x)
        print(y)
        pass
