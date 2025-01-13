# 模型总体结构代码

import torch.nn as nn
import torch.nn.functional as F     # 函数模块
from .net import *

class DeGenerater(nn.Module):
    def __init__(self):
        super(DeGenerater,self).__init__()
        self.feature_map=[]
        self.backbone=Backbone_D()
        self.neck=Neck_D()

    def forward(self,x):
        self.feature_map=self.backbone(x)
        out_map=self.neck(self.feature_map)
        # 归一化到[0,1]
        output_min = out_map.min()
        output_max = out_map.max()
        out_map = (out_map - output_min) / (output_max - output_min)
        return out_map