#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   testmodel.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-09-08 17:23   tango      1.0         None
"""
from torchsummary import summary
from thop import profile,clever_format
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
############
whichmodel="setr_MLA"
image_size=512
num_classes=7
############
if whichmodel=="DeepLabv3_plus":
    from models.DeepLabV3_plus import deeplabv3_plus
    model = deeplabv3_plus.DeepLabv3_plus(in_channels=3, num_classes=26, backend='resnet34', os=16,)
elif whichmodel=="unet":
    from models.Unet.Unet3 import UNet
    model = UNet(in_channels=3, num_classes=6)
elif whichmodel == "fpn":
    from models.FPN import fpn
    model = fpn.FPN(num_classes=num_classes)
elif whichmodel == "pranet":
    from models.PraNet.PraNet_Res2Net import PraNet
    model = PraNet(channel=6)
elif whichmodel == "caranet":
    from models.CaraNet.CaraNet import caranet
    model = caranet(channel=6)
elif whichmodel == "hardnet":
    from models.HarDNet.HarDMSEG import HarDMSEG
    model = HarDMSEG(channel=6)
elif whichmodel == "dunet":
    from models.DUnet.DoubleUnet import DUNet
    model = DUNet(channel=6)
elif whichmodel == "transunet":
    from models.TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
    from models.TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
    import numpy as np
    vit_name="ViT-B_16"
    vit_name="R50-ViT-B_16"
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = 6
    config_vit.n_skip = 3
    model = ViT_seg(config_vit, img_size=1024, num_classes=6)
    model.load_from(weights=np.load(config_vit.pretrained_path))
elif whichmodel == "setr_MLA":
    from models.setr import SETR
    model = SETR.SETR_MLA_L(img_size=image_size, num_classes=num_classes)
elif whichmodel == "setr_PUP":
    from models.setr import SETR
    model = SETR.SETR_PUP_L(img_size=image_size, num_classes=num_classes)
elif whichmodel == "pvt_varra":
    from models.PVT import pvt_varra
    import sys
    model=pvt_varra.RAPVT(img_size=image_size, num_classes=num_classes, pvt_type="pvt_medium")
    print("Loading PVT weight------->>>>>")
    if "win" in sys.platform:
        model.init_weights(pretrained="H:\weidong\my-seg-pro\models\PVT\pvt_medium.pth")
    else:
        model.init_weights(pretrained="/Users/dongzai/PycharmProjects/MySegPro/models/PVT/pvt_medium.pth")
elif whichmodel == "basepra_pvt":
    from models.rethink import basepra_pvt_ra
    import sys
    model=basepra_pvt_ra.BasePraPvt(img_size=image_size, channel=num_classes,deepsupervise=False)
    print("Loading PVT weight------->>>>>")
    if "win" in sys.platform:
        model.init_weights(pretrained="H:\weidong\my-seg-pro\models\PVT\pvt_medium.pth")
    else:
        model.init_weights(pretrained="/Users/dongzai/PycharmProjects/MySegPro/models/PVT/pvt_medium.pth")
elif whichmodel == "res_varra":
    from models.PVT import resnet_varra
    model=resnet_varra.RAPVT(img_size=image_size, num_classes=num_classes, ra=True)
elif whichmodel == "pvt_fpn":
    import sys
    from models.PVT import pvt_fpn
    model=pvt_fpn.PVTFPN(img_size=image_size,num_classes=num_classes)
    if "win32" in sys.platform:
        model.init_weights(pretrained="H:\weidong\my-seg-pro\models\PVT\pvt_medium.pth")
    else:
        model.init_weights(pretrained="/Users/dongzai/PycharmProjects/MySegPro/models/PVT/pvt_medium.pth")
elif whichmodel == "raunet":
    from models.PVT import pvt_ra_unet
    model=pvt_ra_unet.RAUnet(img_size=image_size, num_classes=num_classes, ra=True, pvt_type="pvt_medium")
elif whichmodel == "convraunet":
    from models.PVT import conv_pvt_ra_unet
    model=conv_pvt_ra_unet.CovRAUnet(img_size=image_size, num_classes=num_classes, ra=True, pvt_type="pvt_medium")
elif whichmodel == "swin_ra":
    import sys
    from models.PVT import swin_ra_unet
    model=swin_ra_unet.RAUnet(img_size=image_size, num_classes=num_classes, ra=True, pvt_type="pvt_medium")
    if "win32" in sys.platform:
        model.init_weights(pretrained="H:\weidong\my-seg-pro\models\swin\swin_base_patch4_window12_384_22k.pth")
    else:
        model.init_weights(pretrained="/Users/dongzai/PycharmProjects/MySegPro/models/swin/swin_base_patch4_window12_384_22k.pth")

model=model.to(device)
# print(summary(model, (3, 1024, 1024)))
input_tensor=torch.randn(4, 3, image_size, image_size).to(device)
flops, params = profile(model, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.3f")
print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))