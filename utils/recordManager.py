#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   recordManager.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-09-15 17:05   tango      1.0         None
"""
import numpy as np
from loguru import logger
from tensorboardX import SummaryWriter


class RecordManager():
    def __init__(self, pathManager):
        self.pm = pathManager
        self.lg = logger
        self.writer = SummaryWriter(self.pm.tensorboad_dir, flush_secs=60)
        self.lg.remove(handler_id=None)
        self.lg.add(self.pm.log_path, enqueue=False, level="INFO", format="{message}")
        self.lg.info("epoch,Tloss,Vloss,macc,mdice,miou,recall,precision,f1")
        self.init_loss()
        self.init_metrics()

    def init_metrics(self):
        self.totalAcc = []
        self.totalDice = []
        self.totalIou = []
        self.totalRecall = []
        self.totalPre = []
        self.totalF1 = []

    def init_loss(self):
        self.totalLoss = []
        self.valToalLoss = [0]
        self.bestDice = -1
        self.bestIoU = -1

    def update_loss(self, loss, t=0):
        if t == 0:
            self.totalLoss.append(loss)
        else:
            self.valToalLoss.append(loss)

    def update_bestmetrics(self):
        if self.bestIoU < np.mean(self.totalIou):
            self.bestIoU = np.mean(self.totalIou)
        if self.bestDice < np.mean(self.totalDice):
            self.bestDice = np.mean(self.totalDice)

    def update_metrics(self, *metrics, t=0):
        self.totalAcc.append(metrics[0])
        self.totalDice.append(metrics[1])
        self.totalIou.append(metrics[2])
        self.totalRecall.append(metrics[3])
        self.totalPre.append(metrics[4])
        self.totalF1.append(metrics[5])

    # @property
    # def lossCheckPoint(self):
    #     return np.abs(self.TmLoss - self.pre_TmLoss)

    @property
    def diceCheckPoint(self):
        return self.bestDice <= np.mean(self.totalDice)

    @property
    def iouCheckPoint(self):
        return self.bestIoU <= np.mean(self.totalIou)

    @property
    def mAcc(self):
        return np.mean(self.totalAcc)

    @property
    def TmLoss(self):
        return np.mean(self.totalLoss)

    @property
    def VmLoss(self):
        return np.mean(self.valToalLoss)

    @property
    def mDice(self):
        return np.mean(self.totalDice)

    @property
    def mIou(self):
        return np.mean(self.totalIou)

    @property
    def mRecall(self):
        return np.mean(self.totalRecall)

    @property
    def mPrecision(self):
        return np.mean(self.totalPre)

    @property
    def mF1(self):
        return np.mean(self.totalF1)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def update2board(self, y_label, value, x):
        self.writer.add_scalar(y_label, value, x)

    def update2file(self, epoch):
        info = np.around(
            [epoch, self.TmLoss, self.VmLoss, self.mAcc, self.mDice, self.mIou, self.mRecall, self.mPrecision,
             self.mF1], 5).tolist()
        info = list(map(lambda x: str(x), info))
        self.lg.info(",".join(info))

    def saveConfig(self, config):
        import json
        with open(self.pm.configpath, "w", encoding="utf8") as f:
            json.dump(config, f, indent=2)
