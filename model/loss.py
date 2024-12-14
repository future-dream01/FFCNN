# 接受网络输出的退化图和目标图像，计算损失值
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
# 使用预训练的 VGG19 模型作为感知损失的基础



class PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu2_2', 'relu3_3', 'relu4_3']):
        super(PerceptualLoss, self).__init__()
        
        # Load VGG19 model and modify the first layer for single-channel input
        vgg = models.vgg19(pretrained=True).features
        vgg[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Modify input channels from 3 to 1
        self.layers = layers
        self.vgg = nn.Sequential(*list(vgg.children())[:36]).eval()  # Keep layers up to relu4_3
        
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters
        
        self.layer_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 17,
            'relu4_3': 26
        }
    
    def forward(self, x, y):
        x = x.to('cuda')
        y = y.to('cuda')
        x_vgg = self.extract_features(x)
        y_vgg = self.extract_features(y)
        loss = 0.0
        for layer in self.layers:
            loss += F.mse_loss(x_vgg[layer], y_vgg[layer])
        return loss
    
    def extract_features(self, x):
        features = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            if int(name) in self.layer_mapping.values():
                layer_name = list(self.layer_mapping.keys())[list(self.layer_mapping.values()).index(int(name))]
                features[layer_name] = x
        return features

def loss_MSE(device, output, label):
    MSE = nn.MSELoss().to(device)
    loss = MSE(output, label)
    return loss

def loss_L1(device, output, label):
    L1 = nn.L1Loss().to(device)
    loss = L1(output, label)
    return loss

def loss_criterion(device, output, label):
    criterion = PerceptualLoss().to(device)
    loss = criterion(output, label)
    return loss

def loss_TOTAL(device, output, label):
    mse_loss = loss_MSE(device, output, label)
    l1_loss = loss_L1(device, output, label)
    criterion_loss=loss_criterion(device, output, label)
    total_loss = mse_loss  + l1_loss +criterion_loss
    return total_loss