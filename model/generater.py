# 生成器完整代码

import torch 
import torch.nn as nn
import torch.nn.functional as F     # 函数模块

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
        self.Backbone1=nn.Sequential(






        )
        pass
    
    def Conv_128_292():     # 第一个卷积、上采样层：128->292

        pass



