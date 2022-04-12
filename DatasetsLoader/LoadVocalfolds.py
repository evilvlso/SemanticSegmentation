#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LoadVocalfolds.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
1/17/22 1:40 PM   tango      1.0         None
"""


'''
import imageio
import os
import PIL
import numpy as np
annot="/Users/dongzai/Downloads/vocalfolds-master/annot/patient1"
# imgs="/Users/dongzai/Downloads/vocalfolds-master/img/patient1"
# vis="/Users/dongzai/Downloads/vocalfolds-master/vis/patient2"
vis="/Users/dongzai/Downloads/vocalfolds-master/masks/patient1"
num_classes=7
#      黑
colors = [(0,0,0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
          (128, 64, 12)]


def toColor(mask,num_classes):
    try:
        mask=mask.numpy()
    except :
        mask=mask.data.cpu().numpy()
    finally:
        seg_img = np.zeros((mask.shape[0], mask.shape[1], 3))  # 先建一个模版
        for c in range(num_classes):  # 为每个类别的每个通道画上颜色
            seg_img[:, :, 0] += ((mask[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((mask[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((mask[:, :] == c) * (colors[c][2])).astype('uint8')
        return seg_img

def createVis():
    for root,dirs,files in os.walk(annot):
        for dir in dirs:
            os.makedirs(os.path.join(vis,dir),exist_ok=True)
            for subroot,subdirs,subfiles in os.walk(os.path.join(root,dir)):
                for subfile in subfiles:
                    if "png" in subfile:
                        img=imageio.imread(os.path.join(subroot,subfile))
                        seg_img = np.zeros((img.shape[0], img.shape[1], 3))  # 先建一个模版
                        for c in range(num_classes):  # 为每个类别的每个通道画上颜色
                            seg_img[:, :, 0] += ((img[:, :] == c) * (colors[c][0])).astype('uint8')
                            seg_img[:, :, 1] += ((img[:, :] == c) * (colors[c][1])).astype('uint8')
                            seg_img[:, :, 2] += ((img[:, :] == c) * (colors[c][2])).astype('uint8')
                        b=PIL.Image.fromarray(seg_img.astype(np.uint8), "RGB")
                        # a=PIL.Image.open(os.path.join(subroot.replace("annot","img"),subfile))
                        # c = PIL.Image.blend(a, b, 0.4)
                        # c.save(os.path.join(vis,dir,subfile))
                    b.save(os.path.join(vis,dir,subfile))
'''

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LoadThroat.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-09-23 10:39   tango      1.0         None
"""

import os
import cv2
import imageio
import numpy as np
from PIL import Image
from einops import rearrange
from torch.utils import data
from torchvision import transforms


label_mapping={
    "void": 0,  # 空白   黑
    "vocal_folds": 1,  # 声带 红
    "other_tissue": 2,  # 其他组织 绿
    "glottal_space": 3,  # 声门间隙 黄色
    "pathology": 4,  # 病灶  蓝
    "surgical_tool": 5,  # 手术器械 紫
    "intubation": 6,  # 插管  深绿
}

BasePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# imageValDataPath = os.path.join(BasePath, "SegDatasets/Vocalfolds/val/imgs")
# masksValDataPath = os.path.join(BasePath, "SegDatasets/Vocalfolds/val/annot")
imageValDataPath = os.path.join(BasePath, "SegDatasets/Vocalfolds/special/imgs")
masksValDataPath = os.path.join(BasePath, "SegDatasets/Vocalfolds/special/annot")

imageTrainDataPath = os.path.join(BasePath, "SegDatasets/Vocalfolds/train/imgs")
masksTrainDataPath = os.path.join(BasePath, "SegDatasets/Vocalfolds/train/annot")

class DataSetsVocalfolds(data.Dataset):
    def __init__(self, kind, image_size=512, num_classes=1 + 6,mask_size=512):
        """

        :param data:
        """
        super(DataSetsVocalfolds, self).__init__()
        if kind == "train":
            self.X = [os.path.join(imageTrainDataPath, i) for i in os.listdir(imageTrainDataPath) if i.endswith("png")]
            self.Y = [os.path.join(masksTrainDataPath, i) for i in os.listdir(imageTrainDataPath)]
        else:
            self.X = [os.path.join(imageValDataPath, i) for i in os.listdir(imageValDataPath) if i.endswith("png")]
            self.Y = [os.path.join(masksValDataPath, i) for i in os.listdir(imageValDataPath)]
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
                # transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )(x)
        y = cv2.imread(self.Y[index], cv2.IMREAD_GRAYSCALE)
        # y = cv2.resize(y, (self.mask_size, self.mask_size),interpolation=cv2.INTER_NEAREST)
        y_onehot = np.eye(self.num_classes)[y.reshape([-1])]
        y_onehot = rearrange(y_onehot, "(h w) c->c h w", h=self.mask_size)
        return self.X[index], x, y, y_onehot


def getCVCLoader(kind, **kwargs):
    d = DataSetsVocalfolds(kind=kind)
    return data.DataLoader(d, **kwargs)

if __name__ == '__main__':
    d = DataSetsVocalfolds(kind="train",data_volume="source")
    from torch.utils import data

    dl = data.DataLoader(d, batch_size=4, shuffle=True, num_workers=0, drop_last=True,collate_fn=None)#
    for _, x, y, y_onehot in dl:
        print(x)
        print(y)
        pass
