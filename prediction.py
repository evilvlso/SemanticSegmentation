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
import torch.nn.functional as F
from DatasetsLoader import GetLoader
from utils import getModel, collater, toColor, do_crf_inference, do_crf_evalu
from criterion import *
from utils.pathManager import PathManager
from einops import rearrange

basePath = os.path.dirname(os.path.abspath(__file__))
smooth = 1e-6
sp_models = [ 'caranet',"pvt2"]
model_conf={"Throat":{
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
        "batch_size": 3,
        "image_size": 1024,
        "weight": "10_21_17_23_Epoch47_TLoss_0.87383_Dice_0.66054.pth",
    },
    "dunet": {
        "batch_size": 4,
        "image_size": 512,
        "weight": "10_30_12_47_Epoch68_TLoss_0.22801_Dice_0.60848",
    },
    "hardnet": {
        "batch_size": 4,
        "image_size": 1024,
        "weight": "10_28_09_06_Epoch60_TLoss_0.23543_Dice_0.65253.pth",
    },
    "transunet": {
        "batch_size": 4,
        "image_size": 768,
        "weight": "12_09_21_19_Epoch787_TLoss_0.02483_Dice_0.80252.pth",
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
        "batch_size": 5,
        "image_size": 768,
        "weight": "12_17_15_14_Epoch510_TLoss_0.18845_Dice_0.63375.pth",
    },
    "basepra_pvt": {
        "batch_size": 5,
        "image_size": 768,
        "weight": "12_06_23_28_Epoch313_TLoss_0.03591_Dice_0.75663.pth",
    },
    "pvt_fpn": {
        "batch_size": 4,
        "image_size": 768,
        "weight": "12_08_20_25_Epoch277_TLoss_0.02727_Dice_0.72548.pth",
    },
    "raunet": {
        "batch_size": 5,
        "image_size": 768,
        "weight": "12_16_11_18_Epoch513_TLoss_0.14389_Dice_0.70157.pth",
    },
   "swin_ra": {
       "batch_size": 4,
       "image_size": 768,
       "weight": "12_28_20_02_Epoch302_TLoss_0.21160_Dice_0.65365.pth",
   },
    "convraunet": {
        "batch_size": 10,
        "image_size": 1024,
        "weight": "01_05_23_19_Epoch946_TLoss_0.14748_Dice_0.74341.pth",
        "mask_size":512,
    },
},
    "Vocalfolds":{
    "unet":{
            "batch_size":10,
            "image_size" : 512,
            "weight":"01_19_16_47_Epoch393_TLoss_0.10506_Dice_0.74567.pth",
        },
        "deeplabv3_plus": {
            "batch_size": 16,
            "image_size": 512,
            "weight": "",
        },
        "pranet": {
            "batch_size": 16,
            "image_size": 512,
            "weight": "01_20_21_41_Epoch1000_TLoss_0.04270_Dice_0.79905.pth",
        },
        "dunet": {
            "batch_size": 8,
            "image_size": 512,
            "weight": "01_21_17_14_Epoch819_TLoss_0.05708_Dice_0.76838.pth",
        },
        "hardnet": {
            "batch_size": 32,
            "image_size": 512,
            "weight": "01_20_09_42_Epoch1000_TLoss_0.09657_Dice_0.76190.pth",
        },
        "transunet": {
            "batch_size": 8,
            "image_size": 512,
            "weight": "01_22_21_43_Epoch904_TLoss_0.03928_Dice_0.76826.pth",
        },
        "setr_PUP": {
            "batch_size": 3,
            "image_size": 512,
            "weight": "",
        },
        "res_varra": {
            "batch_size": 4,
            "image_size": 512,
            "weight": "",
        },
        "pvt_varra": {
            "batch_size": 5,
            "image_size": 512,
            "weight": "",
        },
        "basepra_pvt": {
            "batch_size": 14,
            "image_size": 512,
            "weight": "01_23_09_42_Epoch467_TLoss_0.07104_Dice_0.77620.pth",
        },
        "setr_MLA": {
            "batch_size": 6,
            "image_size": 512,
            "weight": "02_16_11_42_Epoch1000_TLoss_0.06680_Dice_0.72600.pth",
        },
        "pvt_fpn": {
            "batch_size": 14,
            "image_size": 512,
            "weight": "01_23_15_57_Epoch358_TLoss_0.04219_Dice_0.79265.pth",
        },
        "raunet": {
            "batch_size": 14,
            "image_size": 512,
            "weight": "01_23_20_24_Epoch719_TLoss_0.07137_Dice_0.77031.pth",
        },
        "swin_ra": {
            "batch_size": 8,
            "image_size": 512,
            "weight": "01_26_19_03_Epoch563_TLoss_0.09744_Dice_0.71627.pth",
        },
        "convraunet": {
            "batch_size": 10,
            "image_size": 512,
            "weight": "01_25_17_30_Epoch692_TLoss_0.07752_Dice_0.76230.pth",
            "mask_size":512,
        },
    },
    "KavsirSEG":{"fpn": {
            "batch_size": 16,
            "image_size": 512,
            "weight": "",
        },},
    "CVC":{"fpn": {
            "batch_size": 24,
            "image_size": 384,
            "weight": "",
        },},
}
params = {
    "whichData": "Vocalfolds",
    "whichModel": "setr_MLA",
    # "collate_fn":512,
    "plot_batch_size":3,
    "inch": 3,
    "num_classes": 7,
    "isLogits": True,
    "savestep": 42,
    "isplot": True,
    "ConvCRF": False,
    "ConvCRF_iter":5,
    "comment": " Vocalfolds {} convcrf new metric has obj"
}
params["comment"] =params["comment"].format(" " if params["ConvCRF"] else "no")
params.update(model_conf[params["whichData"]][params["whichModel"]])
args = parse_args()
pm = PathManager(params["whichData"], params["whichModel"], isTbDir=False)
weight = os.path.join(pm.weight_dir_prefix, params.get("weight"))
if params.get("collate_fn", None):
    collate_fn = collater(patch_size=args.patch_size)
else:
    collate_fn = None
gl = GetLoader(**{"batch_size": params["plot_batch_size"] if params["isplot"] else params["batch_size"], "shuffle": False, "num_workers": 0, "collate_fn": collate_fn,
                  "drop_last": False})
png_prefix = "_".join(params["weight"].split("_")[:5])
genTest = gl.getLoader(params["whichData"], "test", image_size=params["image_size"],mask_size=params.get("mask_size",params["image_size"]))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = getModel(params.get("whichModel"), params.get("inch"), params.get("num_classes"), backbone=args.backbone,
                 backbone_pretrained=args.backbone_pretrained, vit_name=args.vit_name, n_skip=args.n_skip,
                 image_size=params["image_size"],kind ="test")
model = model.to(device)
model.load_state_dict(torch.load(weight, map_location=device)["model"], strict=False)
model.eval()

if params["isplot"]:
    with torch.no_grad():
        figPath = pm.figures_dir_prefix
        for step, data in enumerate(genTest, start=1):
            imgName, x, y, y_onehot = data
            x = x.to(device)
            if params['whichModel'] in sp_models:
                lateral_map_5, lateral_map_4, lateral_map_3, logits = model(x)
            else:
                logits = model(x)
            if params.get('isLogits'):
                logits = torch.softmax(logits, 1)
            mask = torch.argmax(logits, 1)
            if params.get("collate_fn", None):
                x = rearrange(x, "(bs h w) c ph pw->bs c (h ph) (w pw)", h=params["image_size"] / args.patch_size,
                              w=params["image_size"] / args.patch_size)
                y = rearrange(y, "(bs h w)  ph pw->bs  (h ph) (w pw)", h=params["image_size"] / args.patch_size,
                              w=params["image_size"] / args.patch_size)
                y_onehot = rearrange(y_onehot, "(bs h w) c ph pw->bs c (h ph) (w pw)",
                                     h=params["image_size"] / args.patch_size, w=params["image_size"] / args.patch_size)
            # TODO:上色 画图 保存
            # plt.figure(figsize=(3*400,12),dpi=100)
            columns= 4 if params["ConvCRF"] else 3
            fig, axes = plt.subplots(params["plot_batch_size"],columns, figsize=(columns * 3, params["plot_batch_size"] * 3), dpi=100)
            for index, n in enumerate(imgName):
                img = x[index].transpose(0, 1).transpose(1, 2).contiguous().data.cpu()
                # a= Image.open(imgName[index]).resize((params["image_size"],params["image_size"]))
                # b= Image.fromarray(np.uint8(toColor(mask[index],6)))
                # c= Image.blend(a,b,0.5)
                c = mask[index].data.cpu()
                axes[index, 0].imshow(img)
                axes[index, 0].axis("off")
                axes[index, 1].imshow(y[index], cmap="gray")
                axes[index, 1].axis("off")
                # axes[index,2].imshow(np.array(c))#,cmap="gray"
                # axes[index, 2].imshow(c, cmap="gray")
                # 第三列画彩图
                axes[index, 2].imshow(toColor(c,params["num_classes"]))
                axes[index, 2].axis("off")
                # 第四列
                if params["ConvCRF"]:
                    if params.get("mask_size",None) and (params.get("mask_size",None) != params.get("image_size",None)):
                        x = F.interpolate(x, scale_factor=0.5, mode="bilinear")
                    convCRF_mask=do_crf_inference(x[index].cpu(),logits[index].cpu(),ConvCRF_iter=params['ConvCRF_iter'],use_gpu=device)
                    axes[index, 3].imshow(toColor(convCRF_mask,params["num_classes"]))
                    axes[index, 3].axis("off")
            axes[0, 0].set_title("Input", fontsize=28, fontweight="bold")
            axes[0, 1].set_title("GT", fontsize=28, fontweight="bold")
            axes[0, 2].set_title("Mask", fontsize=28, fontweight="bold")
            if params["ConvCRF"]:
                axes[0, 3].set_title(f"ConvCRF/{params['ConvCRF_iter']}", fontsize=28, fontweight="bold")
            fig.suptitle("    ".join([i.split("/")[-1] for i in imgName]))
            plt.subplots_adjust(left=0.027,
                                bottom=0,
                                right=0.99,
                                top=0.93,
                                wspace=0.005,
                                hspace=0.045)
            fig.savefig(os.path.join(figPath, "{}-{}.png".format(png_prefix, str(time.time() * 1000)[:-5])))
            plt.close(fig)
            # plt.show()
            if step == params["savestep"]:
                break
            pass
else:
    from tqdm import tqdm
    import numpy as np
    import json
    state_iou=[]
    state_dice=[]
    for i in range(3):
        # 分类计算miou mdice
        with torch.no_grad():
            with tqdm(total=len(genTest), desc=f'prediction val') as pbar:
                steps_iou = []
                steps_dice = []
                for step, data in enumerate(genTest, start=0):
                    imgName, x, y, y_onehot = data
                    x = x.to(device)
                    y_onehot = y_onehot.to(device)
                    if params['whichModel'] in sp_models:
                        lateral_map_5, lateral_map_4, lateral_map_3, logits = model(x)
                    else:
                        logits = model(x)
                    if params["ConvCRF"]:
                        # logits=torch.softmax(logits, 1)
                        if params.get("mask_size",None) and (params.get("mask_size",None) != params.get("image_size",None)):
                            x=F.interpolate(x,scale_factor=0.5,mode="bilinear")
                        logits = do_crf_evalu(x.cpu(), logits.cpu(), ConvCRF_iter=params['ConvCRF_iter'],use_gpu=device)
                        logits=logits.to(device)
                        dice, iou = multiclass_mIou_mDice(logits, y_onehot,logits=False)
                    else:
                        dice, iou = multiclass_mIou_mDice(logits, y_onehot,logits=params.get('isLogits'))
                    steps_iou.append(iou)
                    steps_dice.append(dice)
                    pbar.update(1)
        steps_iou = np.array(steps_iou)
        steps_dice = np.array(steps_dice)
        # miou=(np.sum(steps_iou,axis=0) / (len(steps_iou)+smooth)).tolist()
        # mdice=(np.sum(steps_dice,axis=0) / (len(steps_dice)+smooth)).tolist()
        miou = (np.sum(steps_iou, axis=0) / (np.sum(steps_iou != 0, axis=0) + smooth)).tolist()
        mdice = (np.sum(steps_dice, axis=0) / (np.sum(steps_dice != 0, axis=0) + smooth)).tolist()
        state_iou.append(miou)
        state_dice.append(mdice)
    with open(pm.predictResultPath, "a", encoding="utf8") as f:
        json.dump({
            "miou": np.mean(np.array(state_iou),axis=0).tolist(),
            "mdice": np.mean(np.array(state_dice),axis=0).tolist(),
            "model": params["weight"],
            "datetime": pm.time_prefix,
            "type": params["comment"],
        }, f, indent=2)
print("OVER!")
