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
from einops import rearrange
from torch.utils import data
from torchvision import transforms

label_mapping = {
    "background": 0,
    "epiglottis": 1,  # 38
    "throat": 2,  # 75
    "pyriform_sinus": 3,  # 113
    "vocal_cords_open": 4,  # 14
    "vocal_cords_close": 5  # 52
}

{
    "void": 0,  # 空白   黑
    "vocal_folds": 1,  # 声带 红
    "other_tissue": 2,  # 其他组织 绿
    "glottal_space": 3,  # 声门间隙 黄色
    "pathology": 4,  # 病灶  蓝
    "surgical_tool": 5,  # 手术器械 紫
    "intubation": 6,  # 插管  深绿
}

# 1060*1075
BasePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

imageValDataPath = os.path.join(BasePath, "SegDatasets/Throat/segVal/imgs")
masksValDataPath = os.path.join(BasePath, "SegDatasets/Throat/segVal/mask")
# imageValDataPath = os.path.join(BasePath, "SegDatasets/Throat/special/imgs")
# masksValDataPath = os.path.join(BasePath, "SegDatasets/Throat/special/mask")

class DataSetsThroat(data.Dataset):
    def __init__(self, kind,data_volume, image_size=1024, num_classes=1 + 5,mask_size=1024):
        """

        :param data:
        """
        super(DataSetsThroat, self).__init__()
        if "source" in  data_volume:
            imageTrainDataPath = os.path.join(BasePath, "SegDatasets/Throat/segTrainSource/imgs")
            masksTrainDataPath = os.path.join(BasePath, "SegDatasets/Throat/segTrainSource/mask")
        elif "all" in data_volume:
            imageTrainDataPath = os.path.join(BasePath, "SegDatasets/Throat/segTrain/imgs")
            masksTrainDataPath = os.path.join(BasePath, "SegDatasets/Throat/segTrain/mask")
        else:
            imageTrainDataPath = os.path.join(BasePath, "SegDatasets/Throat/cpTrain/imgs")
            masksTrainDataPath = os.path.join(BasePath, "SegDatasets/Throat/cpTrain/mask")
        if kind == "train":
            self.X = [os.path.join(imageTrainDataPath, i) for i in os.listdir(imageTrainDataPath) if i.endswith("jpg")]
            self.Y = [os.path.join(masksTrainDataPath, i.replace("jpg","png")) for i in os.listdir(imageTrainDataPath)]
        else:
            self.X = [os.path.join(imageValDataPath, i) for i in os.listdir(imageValDataPath) if i.endswith("jpg")]
            self.Y = [os.path.join(masksValDataPath, i.replace("jpg","png")) for i in os.listdir(imageValDataPath)]
        self.image_size = image_size
        self.mask_size=mask_size
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
        x = imageio.imread(self.X[index])
        x = Image.fromarray(x)
        x = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )(x)
        y = cv2.imread(self.Y[index], cv2.IMREAD_GRAYSCALE)
        y[y == 113] = 3
        y[y == 52] = 5
        y[y == 75] = 2
        y[y == 14] = 4
        y[y == 38] = 1
        y = cv2.resize(y, (self.mask_size, self.mask_size),interpolation=cv2.INTER_NEAREST)
        y_onehot = np.eye(self.num_classes)[y.reshape([-1])]
        y_onehot = rearrange(y_onehot, "(h w) c->c h w", h=self.mask_size)
        return self.X[index], x, y, y_onehot


def getCVCLoader(kind, **kwargs):
    d = DataSetsThroat(kind=kind)
    return data.DataLoader(d, **kwargs)

if __name__ == '__main__':
    d = DataSetsThroat(kind="train",data_volume="source")
    from torch.utils import data

    dl = data.DataLoader(d, batch_size=4, shuffle=True, num_workers=0, drop_last=True,collate_fn=None)#
    for _, x, y, y_onehot in dl:
        print(x)
        print(y)
        pass
