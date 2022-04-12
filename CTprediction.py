#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   prediction.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-09-10 08:51   tango      1.0         None
"""
from utils import setup_seed

setup_seed(seed=666)

import torch
import os
import time
from config import parse_args
import matplotlib.pyplot as plt
from DatasetsLoader import GetLoader
from utils import getModel, collater, toColor
from criterion import *
from utils.pathManager import PathManager
from einops import rearrange

from PIL import Image
import numpy as np

basePath = os.path.dirname(os.path.abspath(__file__))
smooth = 1e-6
sp_models = [ 'caranet',"pvt2"]
model_conf={
    "unet":{
        "batch_size":4,
        "image_size" : 1024,
        "weight":"09_26_15_14_Epoch61_TLoss_0.31978_Dice_0.48348.pth",
    },
    "deeplabv3_plus": {
        "batch_size": 16,
        "image_size": 1024,
        "weight": "10_13_19_37_Epoch51_TLoss_0.14974_Dice_0.68825.pth",
    },
    "pranet": {
        "batch_size": 1,
        "image_size": 512,
        "weight": "12_07_23_18_Epoch105_TLoss_0.00499_Dice_0.68333.pth",
    },
    "dunet": {
        "batch_size": 4,
        "image_size": 512,
        "weight": "10_30_12_47_Epoch68_TLoss_0.22801_Dice_0.60848",
    },
    "hardnet": {
        "batch_size": 4,
        "image_size": 1024,
        "weight": "10_27_15_27_Epoch66_TLoss_0.18261_Dice_0.64097.pth",
    },
    "transunet": {
        "batch_size": 4,
        "image_size": 1024,
        "weight": "11_05_09_27_Epoch495_TLoss_0.10163_Dice_0.66553.pth",
    },
    "setr_PUP": {
        "batch_size": 3,
        "image_size": 512,
        "weight": "11_15_08_45_Epoch310_TLoss_575378.39338_Dice_0.15229.pth",
    },
    "res_varra": {
        "batch_size": 4,
        "image_size": 768,
        "weight": "12_03_22_46_Epoch199_TLoss_0.12133_Dice_0.65673.pth",
    },
    "pvt_varra": {
        "batch_size": 4,
        "image_size": 768,
        "weight": "12_05_23_09_Epoch200_TLoss_0.75784_Dice_0.29507.pth",
    },
}
params = {
    "whichData": "CT",
    "weight": "10_13_19_37_Epoch51_TLoss_0.14974_Dice_0.68825.pth",
    "whichModel": "pranet",
    # "collate_fn":512,
    "inch": 3,
    "num_classes": 2,
    "isLogits": True,
    "savestep": 42,
    "isplot": False,
    "comment": "pranet CT "
}
params.update(model_conf[params["whichModel"]])
args = parse_args()
pm = PathManager(params["whichData"], params["whichModel"], isTbDir=False)
weight = os.path.join(pm.weight_dir_prefix, params.get("weight"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = getModel(params.get("whichModel"), params.get("inch"), params.get("num_classes"), backbone=args.backbone,
                 backbone_pretrained=args.backbone_pretrained, vit_name=args.vit_name, n_skip=args.n_skip,
                 image_size=params["image_size"],kind ="test")
model = model.to(device)
model.load_state_dict(torch.load(weight, map_location=device)["model"], strict=False)
model.eval()
batch_size=8
testdir="H:\weidong\my-seg-pro\SegDatasets\CT\seg\\vali"
savedir="H:\weidong\my-seg-pro\SegDatasets\CT\seg\\ctvis"
import SimpleITK as sitk
import nrrd
for f in os.listdir(testdir):
    data=np.load(os.path.join(testdir,f))
    data=data.transpose(2,0,1)
    data=np.expand_dims(data,1)
    result=[]
    step=[i for i in range(data.shape[0]) ]
    for batch in step[::batch_size]:
        x=torch.Tensor(data[batch:batch+batch_size]).to(device)
        logits = model(x)
        if params.get('isLogits'):
            logits = torch.softmax(logits, 1)
        mask = torch.argmax(logits, 1)
        result.append(mask.cpu().data.numpy())
    result=np.concatenate(result,0)

    saveout = sitk.GetImageFromArray(result)
    sitk.WriteImage(saveout, os.path.join(savedir,f"{f}.nii.gz"))
    print(f)

print("OVER!")
