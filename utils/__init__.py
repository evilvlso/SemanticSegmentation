#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   __init__.py.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-08-31 20:32   tango      1.0         None
"""

import torch
import numpy as np
import random
from einops import rearrange
from .mask2color import toColor
from utils import convcrf as convcrf

def setup_seed(seed=666):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess(inputs,target,logits=True):
    if len(inputs.shape) == 3 and logits:
        inputs = torch.softmax(inputs, 0)
    if len(inputs.shape) == 4 and logits:
        inputs = torch.softmax(inputs, 1)
    if len(target.shape) == 3:
        target = torch.argmax(target,0)
    if len(target.shape) == 4:
        target = torch.argmax(target,1)
    return inputs,target

def getModel(whichModel,inch,num_classes,backbone,backbone_pretrained,vit_name="",n_skip=0,image_size=1024,kind="train"):
    if whichModel=="unet":
        from models.Unet import Unet3
        return Unet3.UNet(in_channels=inch, num_classes=num_classes, filter_scale=1)
    elif whichModel=="deeplabv3_plus":
        from models.DeepLabV3_plus import deeplabv3_plus
        return deeplabv3_plus.DeepLabv3_plus(in_channels=inch, num_classes=num_classes, backend=backbone, os=16,
                                      pretrained=backbone_pretrained)
    elif whichModel == "pranet":
        from models.PraNet.PraNet_Res2Net import PraNet
        return PraNet(channel=num_classes)
    elif whichModel == "fpn":
        from models.FPN import fpn
        return fpn.FPN(num_classes=num_classes,dr=False)
    elif whichModel == "fpn_dr":
        from models.FPN import fpn
        return fpn.FPN(num_classes=num_classes,dr=True)
    elif whichModel == "caranet":
        from models.CaraNet.CaraNet import caranet
        return caranet(channel=num_classes)
    elif whichModel == "hardnet":
        from models.HarDNet.HarDMSEG import HarDMSEG
        return HarDMSEG(channel=num_classes)
    elif whichModel == "dunet":
        from models.DUnet.DoubleUnet import DUNet
        return DUNet(channel=num_classes)
    elif whichModel == "transunet":
        from models.TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
        from models.TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        import numpy as np
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = num_classes
        config_vit.n_skip = n_skip
        model = ViT_seg(config_vit, img_size=image_size, num_classes=num_classes)
        if kind == "train":
            model.load_from(weights=np.load(config_vit.pretrained_path))
        return model
    elif whichModel == "setr_MLA":
        from models.setr import SETR
        model = SETR.SETR_MLA_S(img_size=image_size, num_classes=num_classes)
        return model
    elif whichModel == "setr_PUP":
        from models.setr import SETR
        model = SETR.SETR_PUP_S(img_size=image_size, num_classes=num_classes)
        return model
    elif whichModel == "pvt_varra":
        from models.PVT import pvt_varra
        import sys
        model=pvt_varra.RAPVT(img_size=image_size, num_classes=num_classes, pvt_type="pvt_medium")
        if kind == "train":
            print("Loading PVT weight------->>>>>")
            if "win" in sys.platform :
                model.init_weights(pretrained="H:\weidong\my-seg-pro\models\PVT\pvt_medium.pth")
            else:
                model.init_weights(pretrained="/Users/dongzai/PycharmProjects/MySegPro/models/PVT/pvt_medium.pth")
        return model
    elif whichModel == "basepra_pvt":
        from models.rethink import basepra_pvt_ra
        import sys
        model=basepra_pvt_ra.BasePraPvt(img_size=image_size, channel=num_classes)
        if kind == "train":
            print("Loading PVT weight------->>>>>")
            if "win" in sys.platform:
                model.init_weights(pretrained="H:\weidong\my-seg-pro\models\PVT\pvt_medium.pth")
            else:
                model.init_weights(pretrained="/Users/dongzai/PycharmProjects/MySegPro/models/PVT/pvt_medium.pth")
        return model
    elif whichModel == "res_varra":
        from models.PVT import resnet_varra
        model=resnet_varra.RAPVT(img_size=image_size, num_classes=num_classes, ra=True)
        return model
    elif whichModel == "pvt_fpn":
        from models.PVT import pvt_fpn
        import sys
        model = pvt_fpn.PVTFPN(img_size=image_size, num_classes=num_classes)
        if kind == "train":
            print("Loading PVT weight------->>>>>")
            if "win" in sys.platform:
                model.init_weights(pretrained="H:\weidong\my-seg-pro\models\PVT\pvt_medium.pth")
            else:
                model.init_weights(pretrained="/Users/dongzai/PycharmProjects/MySegPro/models/PVT/pvt_medium.pth")
        return model
    elif whichModel == "raunet":
        from models.PVT import pvt_ra_unet
        import sys
        model=pvt_ra_unet.RAUnet(img_size=image_size, num_classes=num_classes, ra=True, pvt_type="pvt_medium")
        if kind == "train":
            print("Loading PVT weight------->>>>>")
            if "win" in sys.platform:
                model.init_weights(pretrained="H:\weidong\my-seg-pro\models\PVT\pvt_medium.pth")
            else:
                model.init_weights(pretrained="/Users/dongzai/PycharmProjects/MySegPro/models/PVT/pvt_medium.pth")
        return model
    elif whichModel == "convraunet":
        from models.PVT import conv_pvt_ra_unet
        import sys
        model=conv_pvt_ra_unet.CovRAUnet(img_size=image_size, num_classes=num_classes, ra=True, pvt_type="pvt_medium")
        if kind == "train":
            print("Loading PVT weight------->>>>>")
            if "win" in sys.platform:
                model.init_weights(pretrained="H:\weidong\my-seg-pro\models\PVT\pvt_medium.pth")
            else:
                model.init_weights(pretrained="/Users/dongzai/PycharmProjects/MySegPro/models/PVT/pvt_medium.pth")
        return model
    elif whichModel == "swin_ra":
        import sys
        from models.PVT import swin_ra_unet
        model=swin_ra_unet.RAUnet(img_size=image_size, num_classes=num_classes, ra=True, pvt_type="pvt_medium")
        if kind == "train":
            print("Loading swin weight------->>>>>")
            if "win" in sys.platform:
                model.init_weights(pretrained="H:\weidong\my-seg-pro\models\swin\swin_base_patch4_window12_384_22k.pth")
            else:
                model.init_weights(
                    pretrained="/Users/dongzai/PycharmProjects/MySegPro/models/swin/swin_base_patch4_window12_384_22k.pth")
        return model

def disGray(mp:str):
    import matplotlib.pyplot as plt
    import cv2
    img=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
    plt.imshow(img,cmap="gray")
    plt.show()

class collater():
    def __init__(self,patch_size=512):
        self.patch_size = patch_size

    def __call__(self, data):
        """
        reconver: rearrange(a, "(bs h w) c ph pw->bs c (h ph) (w pw)", h=9/patch_size, w=9/patch_size)
        :param data:
        :return:
        """
        patch_size=self.patch_size
        name=[i[0] for i in data]
        x=[]
        y=[]
        y_onehot=[]
        for i in data:
            for j in i:
                if not isinstance(j,str) and len(j.shape) ==2:
                    y.append(rearrange(j, "(h ph) (w pw)->(h w) ph pw", pw=patch_size, ph=patch_size))
                elif not isinstance(j,str) and len(j.shape) ==3 and j.shape[0]==6:
                    y_onehot.append(rearrange(j, "c (h ph) (w pw)->(h w) c ph pw", pw=patch_size, ph=patch_size))
                elif not isinstance(j, str) and len(j.shape) == 3 and j.shape[0] == 3:
                    x.append(rearrange(j, "c (h ph) (w pw)->(h w) c ph pw", pw=patch_size, ph=patch_size))
        x=np.concatenate(x,axis=0)
        y=np.concatenate(y,axis=0)
        y_onehot=np.concatenate(y_onehot,axis=0)
        return name,torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(y_onehot)

def do_crf_inference(image, unary,ConvCRF_iter,use_gpu):

    # get basic hyperparameters
    num_classes = unary.shape[0]
    shape = image.shape[1:]
    config = convcrf.default_conf
    config['filter_size'] = 7
    config['pyinn'] = False

    # if args.normalize:
    #     # Warning, applying image normalization affects CRF computation.
    #     # The parameter 'col_feats::schan' needs to be adapted.
    #
    #     # Normalize image range
    #     #     This changes the image features and influences CRF output
    #     image = image / 255
    #     # mean substraction
    #     #    CRF is invariant to mean subtraction, output is NOT affected
    #     image = image - 0.5
    #     # std normalization
    #     #       Affect CRF computation
    #     image = image / 0.3
    #
    #     # schan = 0.1 is a good starting value for normalized images.
    #     # The relation is f_i = image * schan
    #     config['col_feats']['schan'] = 0.1

    # make input pytorch compatible
    # image = image.transpose(2, 0, 1)  # shape: [3, hight, width]
    # # Add batch dimension to image: [1, 3, height, width]
    # image = image.reshape([1, 3, shape[0], shape[1]])
    # img_var = Variable(torch.Tensor(image))
    #
    # unary = unary.transpose(2, 0, 1)  # shape: [3, hight, width]
    # # Add batch dimension to unary: [1, 21, height, width]
    # unary = unary.reshape([1, num_classes, shape[0], shape[1]])
    # unary_var = Variable(torch.Tensor(unary))
    img_var = image.unsqueeze(0)
    unary_var = unary.unsqueeze(0)
    ##
    # Create CRF module
    use_gpu= not "cpu" in use_gpu.type
    gausscrf = convcrf.GaussCRF(conf=config, shape=shape, nclasses=num_classes,
                                use_gpu=use_gpu)

    # Perform CRF inference
    prediction = gausscrf.forward(unary=unary_var, img=img_var,num_iter=ConvCRF_iter)
    prediction = torch.argmax(prediction,1)
    return prediction[0].data.cpu().numpy()

def do_crf_evalu(image, unary,ConvCRF_iter,use_gpu):

    # get basic hyperparameters
    num_classes = unary.shape[1]
    shape = image.shape[2:]
    config = convcrf.default_conf
    config['filter_size'] = 7
    config['pyinn'] = False
    config['final_softmax'] = True
    config['logsoftmax'] = True

    # if args.normalize:
    #     # Warning, applying image normalization affects CRF computation.
    #     # The parameter 'col_feats::schan' needs to be adapted.
    #
    #     # Normalize image range
    #     #     This changes the image features and influences CRF output
    #     image = image / 255
    #     # mean substraction
    #     #    CRF is invariant to mean subtraction, output is NOT affected
    #     image = image - 0.5
    #     # std normalization
    #     #       Affect CRF computation
    #     image = image / 0.3
    #
    #     # schan = 0.1 is a good starting value for normalized images.
    #     # The relation is f_i = image * schan
    #     config['col_feats']['schan'] = 0.1

    # make input pytorch compatible
    # image = image.transpose(2, 0, 1)  # shape: [3, hight, width]
    # # Add batch dimension to image: [1, 3, height, width]
    # image = image.reshape([1, 3, shape[0], shape[1]])
    # img_var = Variable(torch.Tensor(image))
    #
    # unary = unary.transpose(2, 0, 1)  # shape: [3, hight, width]
    # # Add batch dimension to unary: [1, 21, height, width]
    # unary = unary.reshape([1, num_classes, shape[0], shape[1]])
    # unary_var = Variable(torch.Tensor(unary))
    img_var = image
    unary_var = unary
    ##
    # Create CRF module
    # use_gpu= not "cpu" in use_gpu.type
    use_gpu= False
    gausscrf = convcrf.GaussCRF(conf=config, shape=shape, nclasses=num_classes,
                                use_gpu=use_gpu)

    # Perform CRF inference
    prediction = gausscrf.forward(unary=unary_var, img=img_var,num_iter=ConvCRF_iter)
    return prediction
