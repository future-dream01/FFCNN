import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self,oringin_size):
        super(Backbone,self).__init__()
        self.feture_map=[]
        self.relu=nn.ReLU()     # 激活函数
        # 第一小层 尺寸：oringin_size->oringin_size+16 ; 通道数：1->128
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1)
        self.conv3=nn.Conv2d(in_channels=8,out_channels=32,kernel_size=3,stride=1)
        self.bn1=nn.BatchNorm2d(32)
        self.se1=SEModule(32)
        self.res1=Resn(32,64)
        self.upconv1=Upsample_Conv(64,128,oringin_size,oringin_size+16)

        # 第二小层 尺寸：oringin_size+16 -> oringin_size+32 ; 通道数：128->1024
        self.conv4=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1)
        self.bn2=nn.BatchNorm2d(256)
        self.conv5=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1)
        self.bn3=nn.BatchNorm2d(512)
        self.conv6=nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1)
        self.bn4=nn.BatchNorm2d(1024)
        self.se2=SEModule(1024)
        self.res2=Resn(1024,1024)
        self.upconv2=Upsample_Conv(1024,1024,oringin_size+16,oringin_size+32)

        # 第三小层 尺寸oringin_size+32 -> oringin_size+48 ； 通道数：1024->512
        self.conv7=nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1)
        self.bn5=nn.BatchNorm2d(1024)
        self.conv8=nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3,stride=1)
        self.bn6=nn.BatchNorm2d(512)
        self.conv9=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1)
        self.bn7=nn.BatchNorm2d(512)
        self.se3=SEModule(512)
        self.res3=Resn(512,512)
        self.upconv3=Upsample_Conv(512,512,oringin_size+32,oringin_size+48)

        # 第四小层 尺寸oringin_size+48 -> oringin_size+64 ； 通道数：512->128
        self.conv10=nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1)
        self.bn8=nn.BatchNorm2d(512)
        self.conv11=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1)
        self.bn9=nn.BatchNorm2d(256)
        self.conv12=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1)
        self.bn10=nn.BatchNorm2d(256)
        self.se4=SEModule(256)
        self.res4=Resn(256,128)
        self.upconv4=Upsample_Conv(128,128,oringin_size+48,oringin_size+64)

    def forward(self,x):
        # 第一层
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.se1(x)
        x=self.relu(x)
        x=self.res1(x)
        x=self.relu(x)
        x=self.upconv1(x)
        # 第二层
        x=self.conv4(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.conv5(x)
        x=self.bn3(x)
        x=self.relu(x)
        x=self.conv6(x)
        x=self.bn4(x)
        x=self.relu(x)
        x=self.se2(x)
        x=self.relu(x)
        x=self.res2(x)
        x=self.relu(x)
        x=self.upconv2(x)
        out_2=x
        self.feture_map.append(out_2) # 160*160 1024
        # 第三层
        x=self.conv7(x)
        x=self.bn5(x)
        x=self.relu(x)
        x=self.conv8(x)
        x=self.bn6(x)
        x=self.relu(x)
        x=self.conv9(x)
        x=self.bn7(x)
        x=self.relu(x)
        x=self.se3(x)
        x=self.relu(x)
        x=self.res3(x)
        x=self.relu(x)
        x=self.upconv3(x)
        out_3=x
        self.feture_map.append(out_3)   # 176*176 512
        # 第四层
        x=self.conv10(x)
        x=self.bn8(x)
        x=self.relu(x)
        x=self.conv11(x)
        x=self.bn9(x)
        x=self.relu(x)
        x=self.conv12(x)
        x=self.bn10(x)
        x=self.relu(x)
        x=self.se4(x)
        x=self.relu(x)
        x=self.res4(x)
        x=self.relu(x)
        x=self.upconv4(x)
        out_4=x
        self.feture_map.append(out_4)   # 192*192 128
        return self.feture_map

# 特征融合，将本层走过的特征图通道数合成 1 个
class Neck(nn.Module):
    def __init__(self,oringin_size,target_size):
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1)
        self.conv2=nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1)
        self.conv3=nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1)
        self.bn1=nn.BatchNorm2d(64)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,stride=1)
        self.upconv1=Upsample_Conv(in_channels=1024,out_channels=512,input_size=oringin_size+32,output_size=oringin_size+48)
        self.upconv2=Upsample_Conv(in_channels=256,out_channels=64,input_size=oringin_size+48,output_size=oringin_size+64)

    def forward(self,x):
        out1=self.upconv1(x[0])     # x[0]上采样
        out2=torch.cat((out1,x[1]),dim=1)   # x[0]和 x[1]通道融合
        out2=self.conv1(out2)       # 融合后卷积调整通道数->64
        out3=self.upconv2(out2)     # 融合后上采样->192*192
        out4=self.conv2[x[2]]       # x[2]卷积调整通道数->64
        out4=torch.cat((out3,x[2]),dim=1) 
        out4=self.conv3(out4)       # 全部融合后调整通道数->64
        out4=self.bn1               
        out4=self.relu(out4)
        out4=self.conv4(out4)       # 全部融合后调整通道数->1
        return out4                 # 最终输出的上采样之后的图像

# 指定分辨率上采样卷积层
class Upsample_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, output_size):
        super(Upsample_Conv, self).__init__()
        kernel_size, stride, padding, output_padding = self.calculate_convtranspose2d_params(input_size, output_size)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)

    def calculate_convtranspose2d_params(self,input_size, output_size):
        for stride in range(1, output_size - input_size + 2):
            for kernel_size in range(2, output_size - input_size + 3):
                for padding in range(kernel_size):
                    output_padding = output_size - ((input_size - 1) * stride - 2 * padding + kernel_size)
                    if 0 <= output_padding < stride:
                        return kernel_size, stride, padding, output_padding
        raise ValueError(f"Cannot find suitable parameters for input size {input_size} and output size {output_size}")

    def forward(self, x):
        return self.conv_transpose(x)

# 残差块
class Resn(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(Resn,self).__init__()
        self.relu=nn.ReLU()     # 激活函数
        # 分支1
        self.conv1=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=3,stride=1)  
        # 分支2
        self.conv2=nn.Conv2d(in_channels=input_channel,out_channels=input_channel+64,kernel_size=3,stride=1)
        self.conv3=nn.Conv2d(in_channels=input_channel+64,out_channels=input_channel+128,kernel_size=3,stride=1)
        self.bn1=nn.BatchNorm2d(input_channel+128)
        self.conv4=nn.Conv2d(in_channels=input_channel+128,out_channels=input_channel+256,kernel_size=3,stride=1)
        self.bn2=nn.BatchNorm2d(input_channel+256)
        self.conv5=nn.Conv2d(in_channels=input_channel+256,out_channels=input_channel+128,kernel_size=3,stride=1)
        self.bn3=nn.BatchNorm2d(input_channel+128)
        self.conv6=nn.Conv2d(in_channels=input_channel+128,out_channels=input_channel+64,kernel_size=3,stride=1)
        self.bn4=nn.BatchNorm2d(input_channel+64)
        self.conv7=nn.Conv2d(in_channels=input_channel+64,out_channels=output_channel,kernel_size=3,stride=1)
        self.bn5=nn.BatchNorm2d(output_channel)
    def forward(self,x):
        base=self.conv1(x)
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv4(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.conv5(x)
        x=self.bn3(x)
        x=self.relu(x)
        x=self.conv6(x)
        x=self.bn4(x)
        x=self.relu(x)
        x=self.conv7(x)
        x=self.bn5(x)
        x=self.relu(x)
        out=x+base
        out=self.relu(out)
        return out

# 通道注意力模块
class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
