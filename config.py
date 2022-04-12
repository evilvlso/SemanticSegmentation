#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   config.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-09-15 16:14   tango      1.0         None
"""
import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--early_stop",default=5)
    parser.add_argument("--epochs",default=1000)
    parser.add_argument("--loss_threshold",default=0.0025)
    parser.add_argument("--init_lr",default=4e-5)
    parser.add_argument("--pre_trained",default=False,choices=["False","True"])
    parser.add_argument("--pre_trained_pth",default=r"12_15_22_13_Epoch200_TLoss_0.37461_Dice_0.59445.pth")
    parser.add_argument("--backbone",default="resnet50")
    parser.add_argument("--backbone_pretrained",default="imagenet")
    # parser.add_argument("--vit_name",default="ViT-B_16")
    # parser.add_argument("--n_skip",default=0)
    parser.add_argument("--vit_name",default="R50-ViT-B_16")
    parser.add_argument("--n_skip",default=3)
    parser.add_argument("--patch_size",default=512)
    args=parser.parse_args()
    return args
