import torch 
import torch.nn
import torch.nn.functional as F

# 前向扩散，引入噪声+下采样
class Forward_Diffusion:
    def __init__(self,beta_start,beta_end,timestep,downstep) :
        self.beta=torch.linespace(beta_start,beta_end,timestep)     # 创建噪声强度向量
        self.timestep=timestep                                      # 时间步数
        self.downstep=downstep
    def add_noise_and_downsample(self,x):           # 加入噪声、一定步长下采样
        noise= torch.rand_like(x)                   # 创造噪声张量
        alpha= torch.cumprod(1 -  self.beta, dim=0)
        noise_x=[]                                  # 输出特征图
        for i in range(self.timestep):
            x = x * alpha[i].sqrt() + noise * (1 - alpha[i]).sqrt()
            if i % self.downstep==0:
                noise_x.append(x)                   # 引入噪声+下采样
