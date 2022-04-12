#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   old_vs_new.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-11-11 09:47   tango      1.0         None
"""

import numpy as np
import os
import cv2
import torch
from einops import rearrange
from PIL import Image
from utils import toColor
# p_old="/Users/dongzai/Desktop/res50+DRP"
p_old="/Users/dongzai/Desktop/res50"
p_new="/Users/dongzai/PycharmProjects/MySegPro/SegDatasets/Throat/segVal/mask"

pre_mask=[]
target_mask=[]
steps_iou = []
steps_dice = []
smooth=1e-6


def one_hot_result(label, label_values=[[0], [1], [2]]):
    semantic_map = []
    for color in label_values:
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=1)
    return semantic_map

def MulticlassIoU_fn(inputs, targets,logits=True,smooth=1e-6):
    if logits:
        inputs=torch.softmax(inputs,1)
    inputs = inputs.cpu().detach()
    targets = targets.cpu().float().numpy()
    inputs = torch.argmax(inputs, dim=1)  # shape=(16, 224, 224)
    inputs = torch.unsqueeze(inputs, dim=1)  # shape=(16, 1, 224, 224)
    inputs = inputs.numpy()  # 取值为 numpy
    N = targets.shape[0]  # batch size
    categories = targets.shape[1]  # channels

    label_values = np.arange(categories)  # [0, 1, 2]
    # 独热编码，shape=(16, 1, 224, 224)
    inputs = one_hot_result(inputs, label_values).astype(np.float)

    # iou = np.zeros(categories)  # [0., 0., 0.]
    iou = []  # [0., 0., 0.]
    for input_, target_ in zip(inputs, targets):  # n c h w
        # flatten image size，(3, 50176)
        iflat = input_.reshape(categories, -1)   # c h*w
        tflat = target_.reshape(categories, -1)
        has_obj=np.sum(tflat,axis=1)==0
        # 交集，shape=(3, 50176)
        intersection = iflat * tflat   # c h*w
        intersection = np.sum(intersection, axis=1)  # c
        union = np.sum(iflat, axis=1) + np.sum(tflat, axis=1) #c
        # 计算 iou，交集/两个合起来后的面积
        iou_image = intersection / (union - intersection+smooth) # c
        iou_image[np.isnan(iou_image) * (union==0)] = 1.0
        iou_image[np.isnan(iou_image) * (union!=0)] = 0.0
        # iou_image=iou_image + (has_obj * smooth)
        iou.append(iou_image)
    iou=np.array(iou)
    iou = np.sum(iou,axis=0) / (np.sum(iou!=0,axis=0)+smooth)
    return iou

def MulticlassSoftDice(y_pred, y_true, epsilon=1e-6, logits=True):
    '''

    '''
    if len(y_pred.shape) == 3 and logits:
        y_pred = torch.softmax(y_pred, 0)
    if len(y_pred.shape) == 4 and logits:
        y_pred = torch.softmax(y_pred, 1)
    numerator = 2. * torch.sum(y_pred * y_true,axis=[2,3])
    # denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    denominator = torch.sum(y_pred + y_true,axis=[2,3])
    softdice=(numerator) / (denominator + epsilon)    # b c
    has_obj=torch.sum(y_true, axis=[2, 3])==0
    # softdice=softdice+(has_obj*smooth)
    return (torch.sum(softdice,axis=0) / (torch.sum(softdice!=0,axis=0)+epsilon)).cpu().detach().numpy()

def multiclass_mIou_mDice(inputs, targets,logits=True,smooth=1e-6):
    dice=MulticlassSoftDice(inputs, targets, epsilon=smooth, logits=logits)
    iou=MulticlassIoU_fn(inputs, targets,logits=logits,smooth=smooth)
    return dice,iou

# for i in os.listdir(p_old):
for i in range(159):
    img=np.load(os.path.join(p_old, f"{i}.npy"))
    s=np.argmax(img,axis=-1)
    color_mask=toColor(s,num_classes=6)
    Image.fromarray(np.uint8(color_mask)).save(os.path.join(p_old,f"{i}.jpg"))
    img=np.int8(img)
    img=cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    img=rearrange(img, "h w c->c h w")
    pre_mask.append(torch.from_numpy(img))

for i in range(159):
    y = cv2.imread(os.path.join(p_new, f"{i}.png"), cv2.IMREAD_GRAYSCALE)
    y[y == 113] = 3
    y[y == 52] = 5
    y[y == 75] = 2
    y[y == 14] = 4
    y[y == 38] = 1
    y = cv2.resize(y, (1024,1024),interpolation=cv2.INTER_NEAREST)
    y_onehot = np.eye(6)[y.reshape([-1])]
    y_onehot = rearrange(y_onehot, "(h w) c->c h w", h=1024)
    target_mask.append(torch.from_numpy(y_onehot))

for logits,y_onehot in zip(pre_mask,target_mask):
    logits=logits.unsqueeze(0)
    y_onehot=y_onehot.unsqueeze(0)
    dice, iou = multiclass_mIou_mDice(logits, y_onehot,logits=False)
    steps_iou.append(iou)
    steps_dice.append(dice)
steps_iou = np.array(steps_iou)
steps_dice = np.array(steps_dice)
miou = (np.sum(steps_iou, axis=0) / (np.sum(steps_iou != 0, axis=0) + smooth)).tolist()
mdice = (np.sum(steps_dice, axis=0) / (np.sum(steps_dice != 0, axis=0) + smooth)).tolist()

print("miou:", miou)
print("mdice:", mdice)
