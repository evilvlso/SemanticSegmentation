#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train.py
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-08-31 20:33   tango      1.0         None
"""
from utils import setup_seed, getModel
setup_seed()

import os
import cv2
import torch
from tqdm import tqdm
from torch import optim
import torch.backends.cudnn as cudnn

from DatasetsLoader import GetLoader
from config import parse_args
from criterion import *
from utils.pathManager import PathManager
from utils.recordManager import RecordManager

basePath = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

LossFunc = {
    "F1score": F1score,  # 无用
    "SoftDiceLoss": SoftDiceLoss,
    "CEDiceLoss": CEDiceLoss,
    "CELosstargetWithOneHot": CELosstargetWithOneHot,
    "CELosstargetWithLabel": CELosstargetWithLabel,
    "AccLoss": AccLoss,
    "IoULoss": IoULoss(),
    "FocalLoss": FocalLoss(),  # 需要修改
}

def sendNotify(epoch,start_date,message=""):
    runtime=f"start date:{start_date:>15}\nover date:{time.strftime('%m-%d %H:%M'):>15}"
    message=runtime+message
    print(message)
    import requests
    requests.get(f"https://push.hellyw.com/6146dfbca4d8ef00390a6b91/TranOver-{epoch}/{message}")

def train(model,genTrain,epoch,Epoch):
    model.train()
    with tqdm(total=len(genTrain), desc=f'Training Epoch:{epoch}/{Epoch}') as pbar:
        for step, data in enumerate(genTrain, start=0):
            imgName, x, y, y_onehot = data
            x = x.to(device)
            y_onehot = y_onehot.to(device)
            optimizer.zero_grad()
            logits = model(x)

            loss = lossF(logits, y_onehot)
            rm.update_loss(loss.item())
            loss.backward()
            optimizer.step()
            # 训练指标无意义
            # sacc, sdice, siou, srecall, spre, sf1 = getMetrics(logits, y_onehot, trainInfo[d]["num_classes"])
            # rm.update_metrics(*(sacc.item(), sdice.item(), siou.item(), srecall.item(), spre.item(), sf1.item()))
            pbar.set_postfix(**{'Tloss': rm.TmLoss,
                                'lr': rm.get_lr(optimizer),
                                })
            pbar.update(1)

def eval(model,genVal,rm,epoch=0,Epoch=0):
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(genVal), desc=f'Valing Epoch:{epoch}/{Epoch}') as pbar:
            for step, data in enumerate(genVal, start=0):
                imgName, x, y, y_onehot = data
                x = x.to(device)
                y_onehot = y_onehot.to(device)
                logits = model(x)
                loss = lossF(logits, y_onehot)
                rm.update_loss(loss.item(), t=1)
                sacc, sdice, siou, srecall, spre, sf1 = getMetrics(logits, y_onehot,
                                                                   trainInfo[d]["num_classes"])

                # rm.lg.info(imgName)
                # torch.save(logits, os.path.join(pm.logs_dir_prefix, "vlogits.err"))
                # torch.save(y_onehot, os.path.join(pm.logs_dir_prefix, "vy_onehot.err"))
                rm.update_metrics(
                    *(sacc.item(), sdice.item(), siou.item(), srecall.item(), spre.item(), sf1.item()))
                pbar.set_postfix(**{'Vloss': rm.VmLoss})
                pbar.update(1)

def stop_trigger(model,optimizer,epoch,pm,rm,trigger,early_stop,lr_scheduler):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,"lr_scheduler":lr_scheduler}
    pthPath = pm.weight_path.format(epoch=epoch, TL=rm.TmLoss, dice=rm.mDice)
    # if rm.diceDiff < loss_threshold:
    #     trigger += 1
    # else:
    #     trigger = 0
    #     if epoch >27:
    #         torch.save(state, pthPath)
    if epoch > 30 and (epoch % 5==0 or epoch==499):
        torch.save(state, pthPath)
    # if trigger >= early_stop:
    #     print(f"Trigger triggered ,early stop:{trigger}")
    #     torch.save(state, pthPath)
    #     return True,trigger
    return False,trigger

if __name__ == '__main__':
    import time
    start_date=time.strftime("%m-%d %H:%M")
    args= parse_args()
    d = "KavsirSEG"
    trainInfo = {
        "KavsirSEG": {
            "dataset": "KavsirSEG",
            "batch_size": 16,
            "image_size": 512,
            "num_classes": 1 + 1,
            "criterion": "CEDiceLoss",
            "optimizer": "Adam",
            "whichModel": "transunet",
            "comment": "",
            "data_volume": "source",
        },
        "CVC": {
            "dataset": "CVC",
            "batch_size": 7,
            "image_size": 384,
            "num_classes": 1 + 1,
            "criterion": "CEDiceLoss",
            "optimizer": "Adam",
            "whichModel": "transunet",
            "comment": "transunet 16 ",
            "data_volume": "source",
        },
        "Throat": {
            "dataset": "Throat",
            "batch_size": 5,
            # "image_size": 512,
            "image_size": 1024,
            "num_classes": 1 + 5,
            "criterion": "CEDiceLoss",
            "optimizer": "Adam",
            "whichModel": "transunet",
            "comment":"transunet 16 source data large epoch 1/3 lr",
            "data_volume":"source",
        }
    }
    pm = PathManager(d,trainInfo[d]["whichModel"])
    rm = RecordManager(pm)
    trainInfo[d].update(args.__dict__)
    conf = trainInfo[d]
    print(conf)
    rm.saveConfig(conf)
    model = getModel(whichModel=trainInfo[d]["whichModel"], inch=3, num_classes=trainInfo[d]["num_classes"],backbone=args.backbone,backbone_pretrained=args.backbone_pretrained,vit_name=args.vit_name,n_skip=args.n_skip,image_size=trainInfo[d]["image_size"])
    model = model.to(device)
    gl = GetLoader(**{"batch_size": trainInfo[d]["batch_size"], "shuffle": True, "num_workers": 0,
                      "drop_last": True})
    genTrain = gl.getLoader(d, "train",data_volume=trainInfo[d]["data_volume"],image_size=trainInfo[d]["image_size"])
    genVal = gl.getLoader(d, "val",data_volume=trainInfo[d]["data_volume"],image_size=trainInfo[d]["image_size"])
    lr = args.init_lr
    Epoch = 500
    startEpoch = 0
    pre_trained = args.pre_trained
    pre_trained_pth = args.pre_trained_pth
    dataset = trainInfo[d]["dataset"]
    early_stop = args.early_stop
    loss_threshold = args.loss_threshold

    if trainInfo[d]["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    elif trainInfo[d]["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)  # 每过step_size个epoch，做一次更新
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epoch/3, eta_min=0)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.65)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                      milestones=[200, 300, 320, 340, 200], gamma=0.65)
    lossF = LossFunc[trainInfo[d]["criterion"]]

    if pre_trained:
        print('==> Resuming from checkpoint..')
        pre_trained_pth = os.path.join(pm.weight_dir_prefix,pre_trained_pth)
        checkpoint = torch.load(pre_trained_pth)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        startEpoch = checkpoint['epoch'] + 1
        lr_scheduler=checkpoint['lr_scheduler']

    trigger = 0
    pre_total_loss = 0
    for epoch in range(startEpoch, Epoch):
        try:
            train(model,genTrain,epoch,Epoch)
            eval(model,genVal,rm,epoch,Epoch)
        except Exception as e:
            sendNotify(epoch,e,start_date)
            break
        rm.update2board("LearnRate", rm.get_lr(optimizer), epoch)
        rm.update2board("TrainMeanLoss", rm.TmLoss, epoch)
        rm.update2board("ValMeanLoss", rm.VmLoss, epoch)
        rm.update2board("ValDice", rm.mDice, epoch)
        rm.update2board("ValIou", rm.mIou, epoch)
        rm.update2board("ValAcc", rm.mAcc, epoch)
        rm.update2file(epoch)

        is_stop,trigger=stop_trigger(model,optimizer,epoch,pm,rm,trigger,early_stop,lr_scheduler)
        if is_stop: break

        rm.update_preTloss()
        lr_scheduler.step()
        torch.cuda.empty_cache()
    sendNotify(epoch,start_date)