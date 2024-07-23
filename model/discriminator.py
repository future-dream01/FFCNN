from net import *

# 一号噪声判别器
class Discriminator_noize_1():
    def __init__(self):
        super(Discriminator_noize_1,self).__init__()
        # 噪声判别器
        self.backbone_noize_1=Backbone_D(320)
        self.neck_noize_1=Neck_D()
        self.head_noize_1=Head_D(320)
    def forward(self,output,noise_down_x):
        out=[]
        # 生成器生成图像进入判别器
        x=self.backbone_noize_1(output)
        x=self.neck_noize_1(x)
        x=self.head_noize_1(x)
        # 噪声图像进入判别器
        y=self.backbone_noize_1(noise_down_x)
        y=self.neck_noize_1(y)
        y=self.head_noize_1(y)
        out.append(x)
        out.append(y)
        return out
    
# 二号噪声判别器
class Discriminator_noize_2():
    def __init__(self):
        super(Discriminator_noize_2,self).__init__()
        self.backbone_noize_2=Backbone_D(512)
        self.neck_noize_2=Neck_D()
        self.head_noize_2=Head_D(512)
    def forward(self,output,noise_down_x):
        out=[]
        # 生成器生成图像进入判别器
        x=self.backbone_noize_2(output)
        x=self.neck_noize_2(x)
        x=self.head_noize_2(x)
        # 噪声图像进入判别器
        y=self.backbone_noize_2(noise_down_x)
        y=self.neck_noize_2(y)
        y=self.head_noize_2(y)
        out.append(x)
        out.append(y)
        return out
    
# 一号纹理判别器    
class Discriminator_texture_1():
    def __init__(self):
        super(Discriminator_texture_1,self).__init__()
        self.backbone_texture_1=Backbone_D(320)
        self.neck_texture_1=Neck_D()
        self.head_texture_1=Head_D(320)
    def forward(self,output,down_x):
        out=[]
        # 生成器生成图像进入判别器
        x=self.backbone_texture_1(output)
        x=self.neck_texture_1(x)
        x=self.head_texture_1(x)
        # 噪声图像进入判别器
        y=self.backbone_texture_1(down_x)
        y=self.neck_texture_1(y)
        y=self.head_texture_1(y)
        out.append(x)
        out.append(y)
        return out
    
# 二号纹理判别器
class Discriminator_texture_2():
    def __init__(self):
        super(Discriminator_texture_2,self).__init__()
        self.backbone_texture_2=Backbone_D(512)
        self.neck_texture_2=Neck_D()
        self.head_texture_2=Head_D(512)
        pass
    def forward(self,output,down_x):
        out=[]
        # 生成器生成图像进入判别器
        x=self.backbone_texture_2(output)
        x=self.neck_texture_2(x)
        x=self.head_texture_2(x)
        # 噪声图像进入判别器
        y=self.backbone_texture_2(down_x)
        y=self.neck_texture_2(y)
        y=self.head_texture_2(y)
        out.append(x)
        out.append(y)
        return out