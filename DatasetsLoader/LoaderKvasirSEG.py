#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   LoaderKvasirSEG.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-08-31 19:46   tango      1.0         None
"""
import os
import cv2
import  numpy as np
from PIL import  Image
from einops import rearrange
from torch.utils import data
from torchvision import transforms

BasePath=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
imageDataPath = os.path.join(BasePath,"SegDatasets/Kvasir-SEG/images")
masksDataPath = os.path.join(BasePath,"SegDatasets/Kvasir-SEG/masks")
## 总数据集1000张  分为800:200
allImageData= os.listdir(imageDataPath)
trainDataIndex,testDataIndex= data.dataset.random_split(allImageData,[800,200])



class DataSetsKvasirSEG(data.Dataset):
    def __init__(self,kind,image_size=512,num_classes=1+1):
        """
        800:200   resize 512*512
        :param data:
        """
        super(DataSetsKvasirSEG, self).__init__()
        if kind=="train":
            data= [allImageData[i] for i in trainDataIndex.indices]
            self.X = [ os.path.join(imageDataPath,i) for i in data]
            self.Y = [ os.path.join(masksDataPath,i) for i in data]
        else:
            data = [allImageData[i] for i in testDataIndex.indices]
            self.X = [os.path.join(imageDataPath, i) for i in data]
            self.Y = [os.path.join(masksDataPath, i) for i in data]
        self.image_size=image_size
        self.num_classes=num_classes
        self.kind=kind

    def __len__(self):
        return len(self.X)


    def __getitem__(self, index):
        """
        训练集和测试集都使用它
        :param item:
        :return:
        """
        x= Image.open(self.X[index])
        x= transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ]
            )(x)
        y= cv2.imread(self.Y[index],cv2.IMREAD_GRAYSCALE)
        y= cv2.resize(y,(self.image_size,self.image_size))
        y[y<=100]=0  #背景类
        y[y>100]=1   #病灶类
        y_onehot=np.eye(self.num_classes)[y.reshape([-1])]
        y_onehot = rearrange(y_onehot,"(h w) c->c h w",h=self.image_size)
        return self.X[index],x,y,y_onehot

def getKvasirSEGLoader(kind,**kwargs):
    d = DataSetsKvasirSEG(kind=kind)
    return data.DataLoader(d,**kwargs)

if __name__ == '__main__':
    d=DataSetsKvasirSEG(kind="train")
    from torch.utils import data
    dl=data.DataLoader(d,batch_size=2,shuffle=True,num_workers=0,drop_last=True)
    for x,y,*_ in dl:
        print(x)
        print(y)
        pass
