# 生成器完整代码

import torch 
import torch.nn as nn
import torch.nn.functional as F     # 函数模块
from net import *

# 前向扩散，引入噪声+下采样
class Forward_Diffusion:
    def __init__(self,beta_start,beta_end,timestep=1200,downstep=200) :
        self.beta=torch.linespace(beta_start,beta_end,timestep)     # 创建噪声强度向量
        self.timestep=timestep                                      # 时间步数
        self.downstep=downstep
        self.noise_x=[]                                             # 含有噪声的特征图
        self.noise_down_x=[]                                        # 含有噪声、经过下采样的特征图
        self.down_x=[]                                              # 经过下采样的特征图

    def add_noise_and_downsample(self,x):           # 加入噪声、一定步长下采样
        noise= torch.rand_like(x)                   # 创造噪声张量
        alpha= torch.cumprod(1 -  self.beta, dim=0)
        for i in range(self.timestep):
            x = x * alpha[i].sqrt() + noise * (1 - alpha[i]).sqrt()
            if i % self.downstep==0:
                self.noise_x.append(x)                   # 引入噪声+下采样
        for i in range (0,7):
            self.noise_down_x.append(F.interpolate(self.noise_x[i], scale_factor=(512-64*i)/512, mode='bilinear', align_corners=False))
        return self.noise_down_x

    def downsample(self,x):                         # 下采样
        for i in range(0,7):
            self.down_x.append(F.interpolate(x, scale_factor=(512-64*i)/512, mode='bilinear', align_corners=False))
        return self.down_x

class Back_Diffusion_G(nn.Module):
    def __init__(self):
        super(Back_Diffusion_G,self).__init__()
        self.feature_map=[]
        # 第一部分 128->192
        self.backbone_1=Backbone_G(128)
        self.neck_1=Neck_G(128)
        # 第一部分 192->256
        self.backbone_2=Backbone_G(192)
        self.neck_2=Neck_G(192)
        # 第一部分 256->320
        self.backbone_3=Backbone_G(256)
        self.neck_3=Neck_G(256)
        # 第一部分 320->384
        self.backbone_4=Backbone_G(320)
        self.neck_4=Neck_G(320)
        # 第一部分 384->448
        self.backbone_5=Backbone_G(384)
        self.neck_5=Neck_G(384)
        # 第一部分 448->512
        self.backbone_6=Backbone_G(448)
        self.neck_6=Neck_G(448)

    def forward(self,x):
        out_1=self.backbone_1(x)
        out_1=self.neck_1(out_1)
        out_2=self.backbone_2(out_1)
        out_2=self.neck_2(out_2)
        out_3=self.backbone_3(out_2)
        out_3=self.neck_3(out_3)
        out_4=self.backbone_4(out_3)
        out_4=self.neck_4(out_4)
        out_5=self.backbone_5(out_4)
        out_5=self.neck_5(out_5)
        out_6=self.backbone_6(out_5)
        out_6=self.neck_6(out_6)
        for i in range(0,6):
            self.feature_map[i]=locals()[f'out_{i+1}']
        return self.feature_map