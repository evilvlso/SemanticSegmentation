#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   pathManager.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-09-15 16:44   tango      1.0         None
"""
import os
import time

class PathManager():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self,which_data,which_model,isTbDir=True):
        self.logs_dir_prefix=os.path.join(self.root_dir,"logs",f"{which_data}-{which_model}")
        self.figures_dir_prefix=os.path.join(self.root_dir,"figures",f"{which_data}-{which_model}")
        self.weight_dir_prefix=os.path.join(self.root_dir,"weights",f"{which_data}-{which_model}")
        self.time_prefix=time.strftime("%m_%d_%H_%M")
        self.tensorboad_dir=os.path.join(self.logs_dir_prefix,self.time_prefix)
        self.predictResultPath=os.path.join(self.logs_dir_prefix,"val_miou_mdice.txt")
        self.configpath=os.path.join(self.tensorboad_dir,"config.txt")
        self.isTbDir=isTbDir
        self.init_dir()

    @property
    def log_path(self):
        return os.path.join(self.logs_dir_prefix,f"{self.time_prefix}.csv")

    @property
    def weight_path(self):
        return os.path.join(self.weight_dir_prefix,self.time_prefix+"_Epoch{epoch}_TLoss_{TL:.5f}_Dice_{dice:.5f}.pth")

    @property
    def log_path(self):
        return os.path.join(self.logs_dir_prefix,f"{self.time_prefix}.csv")

    def init_dir(self):
        self.mkdir(self.logs_dir_prefix)
        # self.touchgitignore(self.logs_dir_prefix)
        self.mkdir(self.figures_dir_prefix)
        # self.touchgitignore(self.figures_dir_prefix)
        self.mkdir(self.weight_dir_prefix)
        self.touchgitignore(self.weight_dir_prefix)
        if self.isTbDir:
            self.mkdir(self.tensorboad_dir)

    def touchgitignore(self,d):
        with open(os.path.join(d,".gitignore"),"w",encoding="utf8") as f:
            f.write("""*\n!.gitignore""")

    @staticmethod
    def mkdir(r):
        os.makedirs(r,exist_ok=True)