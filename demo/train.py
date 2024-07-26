from ..model import Data_prepare,load_data
from ..model import Forward_Diffusion,Back_Diffusion_G,Discriminator_noize_1,Discriminator_noize_2,Discriminator_texture_1,Discriminator_texture_2
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger


EPOCHES=500
BATCHSIZE=4

def train():
    # 数据集准备
    dataloader=load_data('../datasets',BATCHSIZE)
    
    # 模型定义
    F=Forward_Diffusion(1e-4,0.1)
    G=Back_Diffusion_G()
    D_noize_1=Discriminator_noize_1()
    D_noize_2=Discriminator_noize_2()
    D_texture_1=Discriminator_texture_1()
    D_texture_2=Discriminator_texture_2()

    # 训练配置
    optimizer_G=optim.Adam(G.parameters(),lr=0.01)
    optimizer_DN1=optim.Adam(D_noize_1.parameters(),lr=0.01)        # 一号噪声判别器梯度优化器
    optimizer_DN2=optim.Adam(D_noize_2.parameters(),lr=0.01)        # 二号噪声判别器梯度优化器
    optimizer_DT1=optim.Adam(D_texture_1.parameters(),lr=0.01)      # 一号纹理判别器梯度优化器
    optimizer_DT2=optim.Adam(D_texture_2.parameters(),lr=0.01)      # 二号纹理判别器梯度优化器

    
    real_labels=torch.ones(BATCHSIZE,1)
    fake_labels=torch.zeros(BATCHSIZE,1)

    G.train()
    for epoch in range(0,EPOCHES):
        for batch_idex , image in enumerate(dataloader):
            # 梯度归零
            optimizer_G.zero_grad()
            optimizer_DN1.zero_grad()
            optimizer_DN2.zero_grad()
            optimizer_DT1.zero_grad()
            optimizer_DT2.zero_grad()

            x_n_d=F.add_noise_and_downsample(image) # 引入噪声+下采样
            x_ds=F.add_noise_and_downsample(image)   # 下采样

            output_g=G(x_n_d[6])

            # 一号噪声判别
            output_d_n_1=D_noize_1(output_g[2].detach(),x_n_d[3])



            output=G(image)
            





    pass

def eval():
    pass



if __name__=='__main__':
    train()
