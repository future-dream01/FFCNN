import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)
from model import DeGenerater, data_prepare, loss_TOTAL, loss_graph
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from datetime import datetime
import torch
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

# 参数设定
EPOCHES = 250    # 轮次数
BATCHSIZE = 1    # 批次数
nan_loss=0
LOAD_CP=False     # 是否需要加载之前的检查点
CP_PATH= f'{project_root}/outputs/weights/12-12_18-03/163weights.pth'    # 检查点权重文件绝对路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 计算设备
current_datetime = datetime.now().strftime("%m-%d_%H-%M")               # 当前时间
log_file_path=f'{project_root}/outputs/训练与性能情况/{current_datetime}/损失日志.log'  # 训练日志文件的绝对路径
os.makedirs(f'{project_root}/outputs/训练与性能情况/{current_datetime}', exist_ok=True)    # 创建训练与性能情况文件夹
os.makedirs(f'{project_root}/outputs/weights/{current_datetime}', exist_ok=True)         # 创建权重文件夹
logger.add(log_file_path,rotation="5000 MB",level="INFO")               #创建日志文件
def train():
    # 训练配置
    global nan_loss
    dataloader = data_prepare(BATCHSIZE)        # 创建数据加载器对象
    D = DeGenerater()                           # 创建模型对象
    D.to(device)                                # 将模型转移到计算设备上
    optimizer_D = optim.Adam(D.parameters(), lr=0.00005)    # 创建梯度优化器
    start_epoch=1 
    losses = []
    if LOAD_CP:                                 # 是否加载先前的检查点文件
        D,optimizer_D,start_epoch,losses=load_checkpoint(D,optimizer_D,CP_PATH)
        min_loss=losses[start_epoch-1]
        d_epoch_num=start_epoch
        start_epoch+=1
    scaler=GradScaler()
    logger.info("开始训练")
    
    # 开始训练
    for epoch in range(int(start_epoch), EPOCHES + 1):
        D.train()                       # 将模型转换为训练模式
        loss_epoch = 0
        batches = 0  # 批次计数
        
        for images, labels in dataloader:
            # 每次迭代生成新的输入和输出
            images, labels = images.to(device), labels.to(device)
            optimizer_D.zero_grad()  # 梯度归零
            # 前向传播
            with autocast():
                output = D(images)
                loss_batch = loss_TOTAL(device, output, labels)
            if not torch.isnan(loss_batch):
                scaler.scale(loss_batch).backward()      # 反向传播
                scaler.step(optimizer_D)            # 梯度下降
                scaler.update()
                # loss_batch.backward()
                # optimizer_D.step()
                # optimizer_D.zero_grad()
                batches += 1
                loss_epoch += loss_batch.item()  # 累加损失
                logger.info(f"epoch:{epoch},batch:{batches},loss_all:{loss_batch.item()}")
            else:
                show_image=images.cpu().clone()
                show_image=show_image.squeeze(0)
                show_image=transforms.ToPILImage()(show_image)
                show_image.save(f"E:\\vscodeProject\\Githubcode\\OwnNet\\datasets\\NANpic\\{epoch}_{batches}.jpg")
                nan_loss +=1
                batches += 1
                logger.info("此批次计算出现NAN,已舍弃此损失值,不对此批次反向传播,图像已提取")
                continue
        loss_epoch = loss_epoch / (batches-nan_loss)  # 本epoch平均损失
        nan_loss=0
        losses.append(loss_epoch)
        logger.info(f"epoch:{epoch},loss_average:{loss_epoch}")
        if (epoch==start_epoch)and LOAD_CP==False:
            save_checkpoint(D,optimizer_D,epoch,losses, f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
            min_loss= loss_epoch            # 初始化最小损失
            d_epoch_num = start_epoch       # 初始化删除的轮次数
        else:
            if loss_epoch < min_loss:
                delpath=f'{project_root}/outputs/weights/{current_datetime}/{d_epoch_num}weights.pth' # 删除对应的权重
                if os.path.exists(delpath):
                    os.remove(delpath)
                    logger.info(f"删除了先前的第{d_epoch_num}轮权重")
                d_epoch_num=epoch           # 更新损失最小的轮次数
                min_loss= loss_epoch        # 更新最小损失
                epochs = range(1, len(losses) + 1)
                loss_graph(current_datetime, epochs, losses)
                logger.info(f"当前损失曲线已绘制")
                save_checkpoint(D,optimizer_D,epoch,losses, f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
                logger.info(f"保存了当前的第{d_epoch_num}轮权重")
        logger.info("当前轮次训练完成")
    logger.info("训练全部完成")
    # 作性能曲线、保存训练权重
    logger.info("训练完成")

# 保存模型权重
def save_checkpoint(model,optimizer,epoch,losses,path):
    state={
        "epoch":epoch,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "loss":losses,
    }
    torch.save(state,path)

# 获取断点的轮次数、损失值
def load_checkpoint(model,optimizer,path):
    if os.path.isfile(path):    # 判断是否存在该检查点权重文件
        checkpoint=torch.load(path) # 加载该权重文件
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch=checkpoint['epoch']
        losses=checkpoint['loss']
        logger.info(f"已成功加载检查点权重,上次结束轮次为：{start_epoch},损失值为:{losses[start_epoch-1]}")
        return model,optimizer,start_epoch,losses
    else:
        logger.info("在所给的路径下未找到对应的权重文件，请检查后重试")

if __name__ == '__main__':
    train()