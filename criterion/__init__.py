#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   __init__.py.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-08-31 20:32   tango      1.0         None
"""
"""
pip install torchmetrics
"""

from .loss import F1score,SoftDiceLoss,CEDiceLoss,CELosstargetWithOneHot,CELosstargetWithLabel,IoULoss,AccLoss,FocalLoss
from .metrics import getMetrics,multiclass_mIou_mDice


