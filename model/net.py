import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone,self).__init__()
        self.relu=nn.ReLU()     # 激活函数
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1)
        self.conv2=
        self.conv3=
        self.bn1=
        self.conv4=
        self.bn2=
        self.conv5=
        self.bn3=
        self.conv6=
        self.bn4=

    def forward(self):
        pass

class Resn(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(Resn,self).__init__()
        self.conv2=
        self.conv3=
        self.bn1=
        self.conv4=
        self.bn2=
        self.conv5=
        self.bn3=
        self.conv6=
        self.bn4=


