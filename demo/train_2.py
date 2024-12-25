import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)
from model import DeGenerater, data_prepare,data_prepare_2, loss_TOTAL, loss_graph,psnr,ssim,allgraph
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
EPOCHES = 500    # 轮次数
BATCHSIZE = 1    # 批次数
train_nan_loss=val_nan_loss=0
LOAD_CP=True     # 是否需要加载之前的检查点
CP_PATH= f'{project_root}/outputs/weights/12-25_00-20/191weights.pth'    # 检查点权重文件绝对路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 计算设备
current_datetime = datetime.now().strftime("%m-%d_%H-%M")               # 当前时间
log_file_path=f'{project_root}/outputs/训练与性能情况/{current_datetime}/损失日志.log'  # 训练日志文件的绝对路径
os.makedirs(f'{project_root}/outputs/训练与性能情况/{current_datetime}', exist_ok=True)    # 创建训练与性能情况文件夹
os.makedirs(f'{project_root}/outputs/weights/{current_datetime}', exist_ok=True)         # 创建权重文件夹
logger.add(log_file_path,rotation="5000 MB",level="INFO")               #创建日志文件
def train():
    # 训练配置
    global train_nan_loss,val_nan_loss
    train_dataloader, val_dataloader = data_prepare_2(BATCHSIZE)        # 创建数据加载器对象
    D = DeGenerater()                           # 创建模型对象
    D.to(device)                                # 将模型转移到计算设备上
    optimizer_D = optim.Adam(D.parameters(), lr=0.00005)    # 创建梯度优化器
    start_epoch=1 
    train_losses = []
    train_psnrs = []
    train_ssims = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    power_datas = []
    if LOAD_CP:                                 # 是否加载先前的检查点文件
        D,optimizer_D,start_epoch,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,power_datas=load_checkpoint(D,optimizer_D,CP_PATH)
        max_power_data=power_datas[start_epoch-1]
        d_epoch_num=start_epoch
        start_epoch+=1
    scaler=GradScaler()
    logger.info("开始训练")

    # 开始训练
    for epoch in range(int(start_epoch), EPOCHES + 1):

        ###################### 训练集训练 #########################

        D.train()                       # 将模型转换为训练模式
        loss_epoch =psnr_epoch=ssim_epoch= 0
        train_batches = 1  # 批次计数
        logger.info(f"第{epoch}轮训练开始，训练集开始训练")
        for images, labels in train_dataloader:
            # 每次迭代生成新的输入和输出
            
            images, labels = images.to(device), labels.to(device)
            optimizer_D.zero_grad()  # 梯度归零
            # 前向传播
            with autocast():
                output = D(images)
                labels=labels/255
                # print(output.max())
                # print(output.min())
                # print((labels).max())
                # print((labels/255).max())
                #print(output.max)
                loss_batch = loss_TOTAL(device, output, labels)
                psnr_batch= psnr(output.detach(),labels)                     # 计算训练集的PSNR
                ssim_batch= ssim(output.detach(),labels)                     # 计算训练集的SSIM
                logger.info(f"epoch:{epoch},batch:{train_batches},\n loss:{loss_batch.item()} \n PSNR:{psnr_batch} \n SSIM:{ssim_batch}")
            if not torch.isnan(loss_batch):
                scaler.scale(loss_batch).backward()      # 反向传播
                scaler.step(optimizer_D)            # 梯度下降
                scaler.update()
                train_batches += 1
                loss_epoch += loss_batch.item()  # 累加损失
                psnr_epoch += psnr_batch         # 累加PSNR
                ssim_epoch += ssim_batch         # 累加SSIM
            else:
                show_image=images.cpu().clone()
                show_image=show_image.squeeze(0)
                show_image=transforms.ToPILImage()(show_image)
                show_image.save(f"E:\\vscodeProject\\Githubcode\\OwnNet\\datasets\\NANpic\\{epoch}_{train_batches}.jpg")
                train_nan_loss +=1
                train_batches += 1
                logger.info("此批次计算出现NAN,已舍弃此损失值,不对此批次反向传播,图像已提取")
                continue
        loss_epoch = loss_epoch / (train_batches-train_nan_loss)  # 本epoch平均损失
        psnr_epoch = psnr_epoch / (train_batches-train_nan_loss)  # 本epoch平均PSNR
        ssim_epoch = ssim_epoch / (train_batches-train_nan_loss)  # 本epoch平均SSIM
        train_nan_loss=0
        train_losses.append(loss_epoch)
        train_psnrs.append(psnr_epoch)
        train_ssims.append(ssim_epoch)
        logger.info(f"epoch:{epoch},\n loss_average:{loss_epoch} \n PSNR_average:{psnr_epoch} \n SSIM_average:{ssim_epoch}")
        loss_epoch =psnr_epoch=ssim_epoch= 0
        ########################## 验证集评估 #################################

        logger.info(f"第{epoch}轮训练集训练完成,开始验证集校验工作")
        D.eval()
        val_batches=1
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer_D.zero_grad()  # 梯度归零
                with autocast():
                    output = D(images)
                    labels=labels/255
                    loss_batch = loss_TOTAL(device, output, labels)
                    psnr_batch= psnr(output,labels)                     # 计算验证集的PSNR
                    ssim_batch= ssim(output,labels)                     # 计算验证集的SSIM
                    logger.info(f"epoch:{epoch},batch:{val_batches},\n loss:{loss_batch.item()} \n PSNR:{psnr_batch} \n SSIM:{ssim_batch}")
                if not torch.isnan(loss_batch):
                    val_batches += 1
                    loss_epoch += loss_batch.item()  # 累加损失
                    psnr_epoch += psnr_batch         # 累加PSNR
                    ssim_epoch += ssim_batch         # 累加SSIM
                else:
                    logger.info("损失值出现nan,已舍弃此批次验证，继续验证")
                    val_batches+=1
                    val_nan_loss+=1
        loss_epoch = loss_epoch / (val_batches-val_nan_loss)  # 本epoch平均损失
        psnr_epoch = psnr_epoch / (val_batches-val_nan_loss)  # 本epoch平均PSNR
        ssim_epoch = ssim_epoch / (val_batches-val_nan_loss)  # 本epoch平均SSIM
        val_nan_loss=0
        val_losses.append(loss_epoch)
        val_psnrs.append(psnr_epoch)
        val_ssims.append(ssim_epoch)
        power_data=(-loss_epoch*100) + (psnr_epoch) +(ssim_epoch*100)
        power_datas.append(power_data)
        logger.info(f"epoch:{epoch},\n loss_average:{loss_epoch} \n PSNR_average:{psnr_epoch} \n SSIM_average:{ssim_epoch} \n Power_data={power_data}")
        loss_epoch =psnr_epoch=ssim_epoch= 0

        ##################### 决定是否保存当前轮次 ###############################

        logger.info(f"第{epoch}轮验证集评估完成，开始判断是否保存本轮权重")
        if (epoch==start_epoch)and LOAD_CP==False:
            save_checkpoint(D,optimizer_D,epoch,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,power_datas, f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
            logger.info(f"第一轮权重已保存")
            max_power_data= power_data            # 初始化最大效果值
            d_epoch_num = start_epoch             # 初始化效果最好的轮次数
        else:
            if power_data > max_power_data:
                delpath=f'{project_root}/outputs/weights/{current_datetime}/{d_epoch_num}weights.pth' # 删除对应的权重
                if os.path.exists(delpath):
                    os.remove(delpath)
                    logger.info(f"删除了先前的第{d_epoch_num}轮权重")
                d_epoch_num=epoch                 # 更新效果最好的轮次数
                max_power_data= power_data        # 更新最好的效果值
                
                save_checkpoint(D,optimizer_D,epoch,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,power_datas, f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
                logger.info(f"保存了当前的第{d_epoch_num}轮权重,最好效果为{max_power_data}")
            else :
                logger.info("不保存此轮权重")

        ################### 绘图 ############################ 
               
        logger.info("开始绘图")
        epochs = range(1, len(power_datas) + 1)
        allgraph(current_datetime, epochs,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,power_datas)
        logger.info("绘图完成")
        logger.info(f"第{epoch}轮训练全部完成")

    logger.info("训练全部完成")

# 保存模型权重
def save_checkpoint(model,optimizer,epoch,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,power_datas,path):
    state={
        "epoch":epoch,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "train_loss":train_losses,
        "train_psnr":train_psnrs,
        "train_ssim":train_ssims,
        "val_loss":val_losses,
        "val_psnr":val_psnrs,
        "val_ssim":val_ssims,
        "power_data":power_datas
    }
    torch.save(state,path)

# 获取断点的轮次数、损失值
def load_checkpoint(model,optimizer,path):
    if os.path.isfile(path):    # 判断是否存在该检查点权重文件
        checkpoint=torch.load(path) # 加载该权重文件
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch=checkpoint['epoch']
        train_losses=checkpoint['train_loss']
        train_psnrs=checkpoint['train_psnr']
        train_ssims=checkpoint['train_ssim']
        val_losses=checkpoint['val_loss']
        val_psnrs=checkpoint['val_psnr']
        val_ssims=checkpoint['val_ssim']
        power_datas=checkpoint['power_data']

        logger.info(f"已成功加载检查点权重,上次结束轮次为：{start_epoch},训练集损失值为:{train_losses[start_epoch-1]}")
        return model,optimizer,start_epoch,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,power_datas
    else:
        logger.info("在所给的路径下未找到对应的权重文件，请检查后重试")

if __name__ == '__main__':
    train()