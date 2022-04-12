#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   loss.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-09-04 20:04   tango      1.0         None
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from torchmetrics.functional import accuracy
from utils import preprocess


def F1score(inputs, target, beta=1, smooth=1e-5, threhold=0.5, logits=True):
    n, c, h, w = inputs.size()
    nt, ct, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    if logits:
        temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    else:
        temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c)
    temp_target = target.view(n, -1, ct)
    temp_inputs = torch.gt(temp_inputs, threhold).float()  # 多分类设置固定阈值是不是不合理
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


def SoftDiceLoss(y_pred, y_true, epsilon=1e-6, logits=True):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format. |A| use square calc(likewise for set B).

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    if len(y_pred.shape) == 3 and logits:
        y_pred = torch.softmax(y_pred, 0)
    if len(y_pred.shape) == 4 and logits:
        y_pred = torch.softmax(y_pred, 1)
    numerator = 2. * torch.sum(y_pred * y_true)
    # denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    denominator = torch.sum(y_pred + y_true)
    return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))  # mean 因为有batchsize,但是mean在这里没有作用,因为里面是一个值


def mDiceLoss(inputs, target, beta=1, smooth=1e-6, logits=True):
    n, c, h, w = inputs.size()
    nt, ct, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    if logits:
        temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    else:
        temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c)
    temp_target = target.view(n, -1, ct)

    # tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1]) #和下面等效
    tp = torch.sum(temp_target * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def CELosstargetWithOneHot(inputs, target, logits=True):
    """
    1. inputs 需要softmax转换为概率
    2. target one_hot形式,n c h w
    3. nn.NLLLoss() 参数格式：inputs n c h w  target n h w
    :param inputs:
    :param target:
    :return:
    """
    n, c, h, w = inputs.size()
    nt, nc, ht, wt = target.size()
    if len(inputs.shape) == 3 and logits:
        temp_inputs = F.log_softmax(inputs, dim=0)
    elif len(inputs.shape) == 4 and logits:
        temp_inputs = F.log_softmax(inputs, dim=1)
        temp_target = torch.argmax(target, 1)
    else:
        temp_target = torch.argmax(target, 1)
        temp_inputs=inputs
    CE_loss = torch.nn.NLLLoss()(temp_inputs, temp_target.type(dtype=torch.long))
    return CE_loss


def CELosstargetWithLabel(inputs, target, logits=True):
    """
    1. inputs 需要softmax转换为概率
    2. target 需要从one_hot转为binary label
    :param inputs:
    :param target:
    :return:
    """
    n, c, h, w = inputs.size()
    # temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)  #和下一句等效
    if len(inputs.shape) == 3 and logits:
        temp_inputs = rearrange(F.log_softmax(inputs, dim=0), "c h w->(h w) c")
    if len(inputs.shape) == 4 and logits:
        temp_inputs = rearrange(F.log_softmax(inputs, dim=1), "n c h w->(n h w) c")
    temp_target = torch.flatten(target)
    CE_loss = torch.nn.NLLLoss()(temp_inputs, temp_target.type(dtype=torch.long))
    return CE_loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, logits=True):
        super(DiceLoss, self).__init__()
        self.logits = logits

    def forward(self, inputs, targets, smooth=1e-6):
        # comment out if your model contains a sigmoid or equivalent activation layer
        if len(inputs.shape) == 3 and self.logits:
            inputs = F.softmax(inputs, dim=0)
        if len(inputs.shape) == 4 and self.logits:
            inputs = F.softmax(inputs, dim=1)
        # flatten label and prediction tensors

        inputs1 = inputs.view(-1)
        targets1 = targets.view(-1)
        intersection1 = (inputs1 * targets1).sum()
        dice = (2. * intersection1 + smooth) / (inputs1.sum() + targets1.sum() + smooth)
        # temp_inputs=rearrange(inputs, "n c h w->(n h w ) c")
        # temp_target=rearrange(targets, "n c h w->(n h w ) c")
        # intersection = (temp_inputs * temp_target).sum(0)
        # dice1 = (2. * intersection + smooth) / (temp_inputs.sum(0) + temp_target.sum(0) + smooth)
        # print(dice1)
        return 1 - torch.mean(dice)


def CEDiceLoss(inputs, target, logits=True):
    ce_loss = CELosstargetWithOneHot(inputs, target, logits=logits)
    dice_loss = SoftDiceLoss(inputs, target, logits=logits)
    return ce_loss , dice_loss


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, logits=True):
        super(DiceBCELoss, self).__init__()
        self.logits = logits

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        if self.logits:
            inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, logits=True):
        super(FocalLoss, self).__init__()
        self.logits = logits

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        if self.logits:
            if len(inputs.shape) == 3 and self.logits:
                inputs = F.softmax(inputs, dim=0)
            if len(inputs.shape) == 4 and self.logits:
                inputs = F.softmax(inputs, dim=1)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, logits=True):
        super(IoULoss, self).__init__()
        self.logits = logits

    def forward(self, inputs, targets, smooth=1e-6):
        # comment out if your model contains a sigmoid or equivalent activation layer
        if len(inputs.shape) == 3 and self.logits:
            inputs = F.softmax(inputs, dim=0)
        if len(inputs.shape) == 4 and self.logits:
            inputs = F.softmax(inputs, dim=1)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


def AccLoss(inputs, target, logits=True):
    """
    y_pre应该是softmax输出的多类的结果
    :param y_pre:
    :param y_true:
    :param device:
    :return:
    """
    inputs, target, = preprocess(inputs, target, logits=logits)
    return 1 - accuracy(inputs, target)


def showMask(data):
    """
    data是一个二维数据 array的
    :param data:
    :return:
    """
    import matplotlib.pyplot as plt

    plt.imshow(data, cmap="gray")
    plt.show()


if __name__ == '__main__':
    inputs = torch.load("/Users/dongzai/PycharmProjects/MySegPro/logs/Throat-deeplabv3_plus/vlogits.err",
                        map_location=torch.device('cpu'))
    target = torch.load("/Users/dongzai/PycharmProjects/MySegPro/criterion/target")
    target_onehot = torch.load("/Users/dongzai/PycharmProjects/MySegPro/logs/Throat-deeplabv3_plus/vy_onehot.err",
                               map_location=torch.device('cpu'))
    # ce loss
    # print(CE_Loss_targetWithLabel(inputs,target))
    # print(CELosstargetWithOneHot(inputs,target_onehot))
    # dice loss
    print(SoftDiceLoss(inputs,target_onehot))
    # print(mDiceLoss(inputs,target_onehot))
    dice=DiceLoss()
    print(dice(inputs,target_onehot))
    # acc
    # print(AccLoss(inputs,target_onehot))
    # f1
    # print(F1score(inputs,target_onehot))
    # iou
    # iou = IoULoss()
    # print(iou(inputs, target_onehot))
