import torch
import torch.nn as nn
import torch.nn.functional as F

# 退化器骨架
class Backbone_D(nn.Module):
    def __init__(self):
        super(Backbone_D,self).__init__()
        self.feture_map=[]              
        self.relu=nn.ReLU()     # 激活函数
        # 第一小层 尺寸：128->512 通道数：1->64
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(32)
        #self.se1=SEModule(32)
        self.res1=Resn(32,64)
        self.upconv1=Upsample_Conv(64,64,512,512)

        # 第二小层 512->128 ; 通道数：64->128
        self.conv4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.conv5=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(256)
        self.conv6=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn4=nn.BatchNorm2d(512)
        #self.se2=SEModule(512)
        self.res2=Resn(512,256)
        self.downconv1=CustomDownsample(256,128,(128,128))

        # 第三小层 尺寸：128->64 ； 通道数：128->256
        self.conv7=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn5=nn.BatchNorm2d(256)
        self.conv8=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn6=nn.BatchNorm2d(512)
        self.conv9=nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1)
        self.bn7=nn.BatchNorm2d(1024)
        #self.se3=SEModule(1024)
        self.res3=Resn(1024,512)
        self.downconv2=CustomDownsample(512,256,(64,64))

        # 第四小层 尺寸：64->16 ； 通道数：256->512
        self.conv10=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.bn8=nn.BatchNorm2d(512)
        self.conv11=nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1)
        self.bn9=nn.BatchNorm2d(1024)
        self.conv12=nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1)
        self.bn10=nn.BatchNorm2d(1024)
        #self.se4=SEModule(1024)
        self.res4=Resn(1024,512)
        self.downconv3=CustomDownsample(512,512,(16,16))

    def forward(self,x):
        # 第一层
        self.feture_map=[]      # 注意一定一定要在forward函数里面将其重置 否则
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.bn1(x)
        x=self.relu(x)
        #x=self.se1(x)
        x=self.relu(x)
        x=self.res1(x)
        x=self.upconv1(x)
        x=self.relu(x)
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
        #x=self.se2(x)
        x=self.relu(x)
        x=self.res2(x)
        x=self.downconv1(x)
        x=self.relu(x)
        out_2=x
        self.feture_map.append(out_2) # 128*128 128
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
        #x=self.se3(x)
        x=self.relu(x)
        x=self.res3(x)
        x=self.downconv2(x)
        x=self.relu(x)
        out_3=x
        self.feture_map.append(out_3)   # 64*64 256
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
        #x=self.se4(x)
        x=self.relu(x)
        x=self.res4(x)
        x=self.downconv3(x)
        x=self.relu(x)
        out_4=x
        self.feture_map.append(out_4)   # 16*16 512
        return self.feture_map

# 退化器特征融合，特征图通道数合成 1 个,尺寸变为128*128
class Neck_D(nn.Module):
    def __init__(self):
        super(Neck_D,self).__init__()
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels=512,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.downconv1=CustomDownsample(512,128,(64,64))
        self.upconv1=Upsample_Conv(512,256,64,64)
        self.upconv2=Upsample_Conv(128,64,128,128)
        self.sig=nn.Sigmoid()

    def forward(self,x):  # x[(128*128 128),(64*64 256),(16*16 512)]
        out1=self.upconv1(x[2])     # x[2]上采样
        out2=torch.cat((out1,x[1]),dim=1)   # x[2]和 x[1]通道融合,融合之后：64*64 512
        out2=self.downconv1(out2)       # 融合后卷积调整通道数->128 调整后：64*64 128
        out3=self.upconv2(out2)     # 融合后上采样->128*128 64
        out4=self.conv2(x[0])       # x[0]卷积调整通道数->64
        out4=torch.cat((out3,out4),dim=1) # 融合后->128*128 128
        out4=self.conv3(out4)       # 全部融合后调整通道数->64
        out4=self.bn1(out4)               
        out4=self.relu(out4)
        out4=self.conv4(out4)       # 全部融合后调整通道数->1
        out4=self.sig(out4)
        return out4                 # 最终输出退化之后的图像


# 指定分辨率上采样卷积层
class Upsample_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, target_height, target_width, kernel_size=3, stride=1, padding=1):
        super(Upsample_Conv, self).__init__()
        self.upsample = nn.Upsample(size=(target_height, target_width), mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # 上采样到指定的目标分辨率
        x = self.upsample(x)
        # 通过卷积层
        x = self.conv(x)
        return x

class CustomDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, output_size):
        super(CustomDownsample, self).__init__()
        self.upsample = nn.Upsample(size=output_size, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 使用上采样层进行插值
        x = self.upsample(x)
        # 通过卷积层
        x = self.conv(x)
        return x
# 残差模块
class Resn(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(Resn,self).__init__()
        self.relu=nn.ReLU()     # 激活函数
        # 分支1
        self.conv1=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=3,stride=1,padding=1)  
        # 分支2
        self.conv2=nn.Conv2d(in_channels=input_channel,out_channels=input_channel+64,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=input_channel+64,out_channels=input_channel+128,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(input_channel+128)
        self.conv4=nn.Conv2d(in_channels=input_channel+128,out_channels=input_channel+256,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(input_channel+256)
        self.conv5=nn.Conv2d(in_channels=input_channel+256,out_channels=input_channel+128,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(input_channel+128)
        self.conv6=nn.Conv2d(in_channels=input_channel+128,out_channels=input_channel+64,kernel_size=3,stride=1,padding=1)
        self.bn4=nn.BatchNorm2d(input_channel+64)
        self.conv7=nn.Conv2d(in_channels=input_channel+64,out_channels=output_channel,kernel_size=3,stride=1,padding=1)
        self.bn5=nn.BatchNorm2d(output_channel)
    def forward(self,x):
        base=self.conv1(x)
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

