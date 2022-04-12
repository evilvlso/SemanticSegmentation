#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   __init__.py.py    
@Contact :   bwdtango@foxmail.com
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021-08-31 19:45   tango      1.0         None
"""
from torch.utils import data

ds = {
    "KavsirSEG",
    "CVC",
    "Throat",
    "CT",
    "Vocalfolds"
}


class GetLoader():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def getLoader(self, whichData, loaderType,data_volume="source",image_size=1024,mask_size=1024):
        assert whichData in ds, "no this data"
        if whichData == "KavsirSEG":
            from .LoaderKvasirSEG import DataSetsKvasirSEG
            d = DataSetsKvasirSEG(kind=loaderType)
        elif whichData == "CVC":
            from .LoadCVC import DataSetsCVC
            d = DataSetsCVC(kind=loaderType)
        elif whichData == "CT":
            from .LoadCT import DataSetsCT
            d = DataSetsCT(kind=loaderType)
        elif whichData == "Vocalfolds":
            from .LoadVocalfolds import DataSetsVocalfolds
            d = DataSetsVocalfolds(kind=loaderType)
        elif whichData == "Throat":
            from .LoadThroat import DataSetsThroat
            d = DataSetsThroat(kind=loaderType,data_volume=data_volume,image_size=image_size,mask_size=mask_size)
        return data.DataLoader(d, **self.kwargs)
