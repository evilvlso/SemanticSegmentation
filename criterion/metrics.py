#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   metrics.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-09-04 20:04   tango      1.0         None
"""
import torch
import numpy as np
from torchmetrics.functional import accuracy, iou, dice_score, recall, f1, precision
from utils import preprocess
import torch.nn.functional as F
from einops import rearrange

##以下都是通过softmax的概率值取argmax计算混淆矩阵的

def MetricsAccu(inputs, target, logits=True):
    inputs, target, = preprocess(inputs, target, logits=logits)
    return accuracy(inputs, target)


#### 下面两个计算多分类不适用（某张图片不存在某类别不应该算在分母中,下面两个算在了分母中，下面两个对二分类是适用的）
# def MetricsDice(inputs,target,logits=True):
#     inputs, target,=preprocess(inputs, target, logits=logits)
#     return dice_score(inputs,target,bg=True)
#
# def MetricsIou(inputs,target,num_classes=2,logits=True):
#     inputs, target,=preprocess(inputs, target, logits=logits)
#     return iou(inputs,target,num_classes=num_classes)
# --------------------------------------------

def MetricsDice(inputs, target, logits=True):
    mdice = MulticlassSoftDice(inputs, target, epsilon=1e-6, logits=logits)
    return torch.mean(torch.from_numpy(mdice))


def MetricsIou(inputs, target, logits=True):
    miou = MulticlassIoU_fn(inputs, target, logits=logits, smooth=1e-6)
    return torch.mean(torch.from_numpy(miou))


def MetricsRecall(inputs, target, logits=True, num_classes=2):
    inputs, target, = preprocess(inputs, target, logits=logits)
    return recall(inputs, target, mdmc_average="samplewise", average='macro', num_classes=num_classes)


def MetricsPre(inputs, target, logits=True, num_classes=2):
    inputs, target, = preprocess(inputs, target, logits=logits)
    return precision(inputs, target, mdmc_average="samplewise", average='macro', num_classes=num_classes)


def MetricsF1(inputs, target, logits=True, num_classes=2):
    inputs, target, = preprocess(inputs, target, logits=logits)
    return f1(inputs, target, mdmc_average="samplewise", average='macro', num_classes=num_classes)


def getMetrics(inputs, target, num_classes):
    sacc = MetricsAccu(inputs, target)
    sdice = MetricsDice(inputs, target)
    siou = MetricsIou(inputs, target)
    srecall = MetricsRecall(inputs, target, num_classes=num_classes)
    spre = MetricsPre(inputs, target, num_classes=num_classes)
    sf1 = MetricsF1(inputs, target, num_classes=num_classes)
    return sacc, sdice, siou, srecall, spre, sf1


def one_hot_result(label, label_values=[[0], [1], [2]]):
    semantic_map = []
    for color in label_values:
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=1)
    return semantic_map


def MulticlassIoU_fn(inputs, targets, logits=True, smooth=1e-6):
    if logits:
        inputs = torch.softmax(inputs, 1)
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
        iflat = input_.reshape(categories, -1)  # c h*w
        tflat = target_.reshape(categories, -1)
        has_obj = np.sum(tflat, axis=1) > 100
        # 交集，shape=(3, 50176)
        intersection = iflat * tflat  # c h*w
        intersection = np.sum(intersection, axis=1)  # c
        union = np.sum(iflat, axis=1) + np.sum(tflat, axis=1)  # c
        # 计算 iou，交集/两个合起来后的面积
        iou_image = intersection / (union - intersection + smooth)  # c
        iou_image[np.isnan(iou_image) * (union == 0)] = 1.0
        iou_image[np.isnan(iou_image) * (union != 0)] = 0.0
        iou_image = iou_image + (has_obj * smooth)
        iou.append(iou_image)
    iou = np.array(iou)
    iou = np.sum(iou, axis=0) / (np.sum(iou != 0, axis=0) + smooth)
    return iou


def MulticlassSoftDice(y_pred, y_true, epsilon=1e-6, logits=True):
    '''

    '''
    if len(y_pred.shape) == 3 and logits:
        y_pred = torch.softmax(y_pred, 0)
    if len(y_pred.shape) == 4 and logits:
        y_pred = torch.softmax(y_pred, 1)
    numerator = 2. * torch.sum(y_pred * y_true, axis=[2, 3])
    # denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    denominator = torch.sum(y_pred + y_true, axis=[2, 3])
    softdice = (numerator) / (denominator + epsilon)  # b c
    has_obj = torch.sum(y_true, axis=[2, 3]) == 0
    # softdice = softdice + (has_obj * epsilon)
    return (torch.sum(softdice, axis=0) / (torch.sum(softdice != 0, axis=0) + epsilon)).cpu().detach().numpy()


def multiclass_mIou_mDice(inputs, targets, logits=True, smooth=1e-6):
    # dice = MulticlassSoftDice(inputs, targets, epsilon=smooth, logits=logits)
    iou = MulticlassIoU_fn(inputs, targets, logits=logits, smooth=smooth)
    dice = cal_dice(inputs, targets, logits=logits, smooth=smooth)
    return dice, iou

def cal_dice(inputs, targets, logits=True, smooth=1e-6):
    dice=[]
    batch=targets.shape[0]
    num_classes=targets.shape[1]
    if len(inputs.shape) == 4 and logits:
        inputs = torch.softmax(inputs, 1)
    masks=torch.argmax(inputs, 1)
    pre_onehot = F.one_hot(masks, num_classes)
    pre_onehot = rearrange(pre_onehot, "b h w c->b c h w")
    for i in range(batch):
        has_obj = torch.sum(targets[i], axis=[1,2]) > 100
        intersection = pre_onehot[i] * targets[i]  # c h*w
        intersection = 2. *  torch.sum(intersection, axis=[1,2])  # c
        union = torch.sum(pre_onehot[i], axis=[1,2]) + torch.sum(targets[i], axis=[1,2])  # c
        # 计算 iou，交集/两个合起来后的面积
        dice_image = intersection / (union + smooth)  # c
        dice_image=dice_image + (has_obj * smooth)
        dice.append(dice_image.cpu().numpy())
    # dice = [i for i in dice]
    dice = np.array(dice)
    dice = np.sum(dice, axis=0) / (np.sum(dice != 0, axis=0) + smooth)
    return dice

if __name__ == '__main__':
    import torch

    inputs = torch.load("/Users/dongzai/PycharmProjects/MySegPro/DatasetsLoader/DatasetsLoader/masks.pth",
                        map_location=torch.device('cpu'))
    target = torch.load("/Users/dongzai/PycharmProjects/MySegPro/criterion/target")
    target_onehot = torch.load("/Users/dongzai/PycharmProjects/MySegPro/logs/Throat-deeplabv3_plus/vy_onehot.err",
                               map_location=torch.device('cpu'))
    # print(MetricsAccu(inputs,target))
    # print(MetricsAccu(inputs,target_onehot))
    print(MetricsDice(inputs, target_onehot))
    print(cal_dice(inputs, target_onehot))
    # print(MetricsRecall(inputs,target_onehot,num_classes=6))
    # print(MetricsPre(inputs,target_onehot,num_classes=6))
    # print(MetricsF1(inputs,target_onehot,num_classes=6))
    # print(MulticlassIoU_fn(inputs,target_onehot))
    # print(MulticlassSoftDice(inputs,target_onehot))
