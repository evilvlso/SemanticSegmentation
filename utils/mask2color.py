#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   mask2color.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-11-06 10:11   tango      1.0         None
"""

import numpy as np

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