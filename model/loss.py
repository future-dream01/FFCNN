# 接受网络输出的退化图和目标图像，计算损失值
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
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



# PSNR计算
def psnr(img1, img2, max_val=255.0):
    """
    计算两个灰度图像张量之间的 PSNR
    :param img1: 第一个灰度图像张量，形状为 (N, 1, H, W) 或 (1, H, W)，值范围为 0-255
    :param img2: 第二个灰度图像张量，形状与 img1 相同，值范围为 0-255
    :param max_val: 图像的最大像素值（默认 255.0）
    :return: PSNR 值
    """
    if img1.ndim == 3:  # 如果是 (1, H, W)，需要扩展为 (N, 1, H, W)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    mse = F.mse_loss(img1, img2, reduction='mean')  # 计算均方误差
    psnr = 10 * torch.log10(max_val**2 / mse)  # 根据公式计算 PSNR
    return psnr.cpu()

# SSIM计算
def ssim(img1, img2, max_val=1.0, window_size=11, sigma=1.5):
    """
    计算两个灰度图像张量之间的 SSIM
    :param img1: 第一个灰度图像张量，形状为 (N, 1, H, W) 或 (1, H, W)
    :param img2: 第二个灰度图像张量，形状与 img1 相同
    :param max_val: 图像的最大像素值（默认 1.0）
    :param window_size: 高斯窗口大小
    :param sigma: 高斯分布的标准差
    :return: SSIM 值
    """
    if img1.ndim == 3:  # 如果是 (1, H, W)，需要扩展为 (N, 1, H, W)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # 生成高斯核
    channels = 1
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel.expand(channels, 1, window_size, window_size).to(img1.device)

    # 计算均值
    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=channels)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=channels) - mu1_mu2

    # SSIM 常量
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # 计算 SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = ssim_map.mean()
    return ssim.cpu()

# MSE损失
def loss_MSE(device, output, label):
    MSE = nn.MSELoss().to(device)
    loss = MSE(output, label)
    return loss

# L1损失
def loss_L1(device, output, label):
    L1 = nn.L1Loss().to(device)
    loss = L1(output, label)
    return loss

# 感知损失
def loss_criterion(device, output, label):
    criterion = PerceptualLoss().to(device)
    loss = criterion(output, label)
    return loss

# 总损失
def loss_TOTAL(device, output, label):
    mse_loss = loss_MSE(device, output, label)
    l1_loss = loss_L1(device, output, label)
    criterion_loss=loss_criterion(device, output, label)
    total_loss =0.4* mse_loss  + 0.2*l1_loss +0.4*criterion_loss
    return total_loss