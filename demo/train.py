from ..model import Data_prepare,load_data
from ..model import Back_Diffusion_G,Discriminator_noize_1,Discriminator_noize_2,Discriminator_texture_1,Discriminator_texture_2
import torch.nn as nn
import torch.optim as optim
from loguru import logger

def train():
    # 数据集准备
    dataloader=load_data('../datasets',4)
    
    # 模型定义
    G=Back_Diffusion_G()
    D_noize_1=Discriminator_noize_1()
    D_noize_2=Discriminator_noize_2()
    D_texture_1=Discriminator_texture_1()
    D_texture_2=Discriminator_texture_2()

    # 训练配置
    optimizer_G=optim.Adam(G.parameters(),lr=0.01)
    optimizer_DN1=optim.Adam(D_noize_1.parameters(),lr=0.01)
    optimizer_DN2=optim.Adam(D_noize_2.parameters(),lr=0.01)
    optimizer_DT1=optim.Adam(D_texture_1.parameters(),lr=0.01)
    optimizer_DT2=optim.Adam(D_texture_2.parameters(),lr=0.01)
    epoch=500

    for i in epoch:
        G.train()





    pass

def eval():
    pass



if __name__=='__main__':
    train()