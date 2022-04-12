#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ac.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-12-30 17:56   tango      1.0         None
"""
from PIL import Image
import torch
from Evison import Display, show_network
from torchvision import models
from models.PVT import pvt_ra_unet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# image = Image.open('/Users/dongzai/PycharmProjects/MySegPro/SegDatasets/Throat/segVal/imgs/0558.jpg').resize((768,768))
# network=pvt_ra_unet.RAUnet(img_size=768, num_classes=6, ra=True, pvt_type="pvt_medium")
# network.load_state_dict(torch.load("/Users/dongzai/PycharmProjects/MySegPro/weights/Throat-raunet/12_16_11_18_Epoch513_TLoss_0.14389_Dice_0.70157.pth", map_location=device)["model"], strict=False)
# visualized_layer = 'up'
# display = Display(network, visualized_layer, img_size=(768, 768))  # img_size的参数指的是输入图片的大小

image = Image.open('/Users/dongzai/Desktop/cat').resize((224,224))
network=models.resnet34(pretrained=True)

show_network(network)
visualized_layer = 'layer4.2.conv2'
display = Display(network, visualized_layer, img_size=(224, 224))  # img_size的参数指的是输入图片的大小
display.save(image,target_class=None, file='cat1')