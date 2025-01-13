import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)
from model import DeGenerater,data_prepare, loss_TOTAL,psnr,ssim,allgraph
import torch.optim as optim
from loguru import logger
from datetime import datetime
import torch
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random

# 参数设定
EPOCHES = 300    # 轮次数
BATCHSIZE = 1    # 批次数
train_nan_loss=val_nan_loss=0
LOAD_CP=False     # 是否需要加载之前的检查点
CP_PATH= f'{project_root}/outputs/weights/01-04_13-11/97weights.pth'    # 检查点权重文件绝对路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 计算设备
current_datetime = datetime.now().strftime("%m-%d_%H-%M")               # 当前时间
log_file_path=f'{project_root}/outputs/训练与性能情况/{current_datetime}/损失日志.log'  # 训练日志文件的绝对路径
os.makedirs(f'{project_root}/outputs/训练与性能情况/{current_datetime}', exist_ok=True)    # 创建训练与性能情况文件夹
os.makedirs(f'{project_root}/outputs/weights/{current_datetime}', exist_ok=True)         # 创建权重文件夹
logger.add(log_file_path,rotation="50000 MB",level="INFO")               #创建日志文件

def train():
    # 训练配置
    global train_nan_loss,val_nan_loss
    train_dataloader, val_dataloader = data_prepare(BATCHSIZE)        # 创建数据加载器对象
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
    val_psnrs_pr=[]
    val_ssims_pr=[]
    power_datas = []

    if LOAD_CP:                                 # 是否加载先前的检查点文件
        logger.info(f"正在加载模型文件")
        D,optimizer_D,start_epoch,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,val_psnrs_pr,val_ssims_pr,power_datas=load_checkpoint(D,optimizer_D,CP_PATH)
        max_power_data=power_datas[start_epoch-1]
        d_epoch_num=start_epoch
        start_epoch+=1
        psnr_epoch_pr=val_psnrs_pr[0]
        ssim_epoch_pr=val_ssims_pr[0]
        logger.info(f"模型文件加载完成,最好powerdata指标:{max_power_data}")
    else:
        logger.info(f"正在计算验证集原始参数指标")
        val_batches_pr=0
        psnr_epoch_pr=ssim_epoch_pr=0
        for images,labels in val_dataloader:
            psnr_batch_pr = psnr(images,labels)
            ssim_batch_pr = ssim(images/255,labels/255)
            psnr_epoch_pr += psnr_batch_pr
            ssim_epoch_pr += ssim_batch_pr
            val_batches_pr+=1
        psnr_epoch_pr=psnr_epoch_pr/val_batches_pr
        ssim_epoch_pr=ssim_epoch_pr/val_batches_pr
        logger.info(f"验证集原始参数指标计算完成,原始平均PSNR:{psnr_epoch_pr},原始平均SSIM:{ssim_epoch_pr}")

    scaler=GradScaler()
    logger.info("开始训练")

    # 开始训练
    for epoch in range(int(start_epoch), EPOCHES + 1):

        ###################### 训练集训练 #########################

        D.train()                       # 将模型转换为训练模式
        loss_epoch =psnr_epoch=ssim_epoch= 0
        train_batches = 0  # 批次计数
        logger.info(f"第{epoch}轮训练开始，训练集开始训练")
        for images, labels in train_dataloader:
            # 每次迭代生成新的输入和输出
            images, labels = images.to(device), labels.to(device)
            optimizer_D.zero_grad()  # 梯度归零
            # 前向传播
            with autocast():
                output = D(images)
                labels=labels/255
                loss_batch = loss_TOTAL(device, output, labels)
                psnr_batch= psnr(output.detach()*255,labels*255)                     # 计算训练集的PSNR
                ssim_batch= ssim(output.detach(),labels)                     # 计算训练集的SSIM
                train_batches += 1
                logger.info(f"epoch:{epoch},batch:{train_batches},\n loss:{loss_batch.item()} \n PSNR:{psnr_batch} \n SSIM:{ssim_batch}")
            if not torch.isnan(loss_batch):
                scaler.scale(loss_batch).backward()      # 反向传播
                scaler.step(optimizer_D)            # 梯度下降
                scaler.update()
                loss_epoch += loss_batch.item()  # 累加损失
                psnr_epoch += psnr_batch         # 累加PSNR
                ssim_epoch += ssim_batch         # 累加SSIM
            else:
                show_image=images.cpu().clone()
                show_image=show_image.squeeze(0)
                show_image=transforms.ToPILImage()(show_image)
                show_image.save(f"E:\\vscodeProject\\Githubcode\\OwnNet\\datasets\\NANpic\\{epoch}_{train_batches}.jpg")
                train_nan_loss +=1
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
        val_batches=0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer_D.zero_grad()  # 梯度归零
                with autocast():
                    output = D(images)
                    labels=labels/255
                    loss_batch = loss_TOTAL(device, output, labels)
                    psnr_batch= psnr(output.detach()*255,labels*255)                     # 计算验证集的PSNR
                    ssim_batch= ssim(output.detach(),labels)                     # 计算验证集的SSIM
                    val_batches += 1
                    logger.info(f"epoch:{epoch},batch:{val_batches},\n loss:{loss_batch.item()} \n PSNR:{psnr_batch} \n SSIM:{ssim_batch}")

                    if val_batches==1:
                        output = output.squeeze().cpu().numpy()
                        #logger.info(f"推理结果: {output}")
                        
                        # 将输出的128x128灰度图截取图像紧贴右侧边界的121x115像素大小的区域
                        cropped_output = output[:, -121:]  # 截取右侧121列
                        cropped_output = cropped_output[6:121-6, :]  # 上下对称截取115行
                        
                        # 将结果放大到128x128
                        cropped_image = Image.fromarray((cropped_output * 255).astype(np.uint8), mode='L')
                        resized_image = cropped_image.resize((128, 128), Image.BILINEAR)
                        
                        output_image_path = f'{project_root}/outputs/训练与性能情况/{current_datetime}/第{epoch}轮结果图片.png'
                        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                        resized_image.save(output_image_path)
                if not torch.isnan(loss_batch):
                    loss_epoch += loss_batch.item()  # 累加损失
                    psnr_epoch += psnr_batch         # 累加PSNR
                    ssim_epoch += ssim_batch         # 累加SSIM
                else:
                    logger.info("损失值出现nan,已舍弃此批次验证，继续验证")
                    val_nan_loss+=1
        loss_epoch = loss_epoch / (val_batches-val_nan_loss)  # 本epoch平均损失
        psnr_epoch = psnr_epoch / (val_batches-val_nan_loss)  # 本epoch平均PSNR
        ssim_epoch = ssim_epoch / (val_batches-val_nan_loss)  # 本epoch平均SSIM
        val_nan_loss=0
        val_losses.append(loss_epoch)
        val_psnrs.append(psnr_epoch)
        val_ssims.append(ssim_epoch)
        val_psnrs_pr.append(psnr_epoch_pr)
        val_ssims_pr.append(ssim_epoch_pr)
        power_data=(-loss_epoch*90) + (psnr_epoch) +(ssim_epoch*140)
        power_datas.append(power_data)
        logger.info(f"epoch:{epoch},\n loss_average:{loss_epoch} \n PSNR_average:{psnr_epoch} \n SSIM_average:{ssim_epoch} \n Power_data={power_data}")
        loss_epoch =psnr_epoch=ssim_epoch= 0

        ##################### 决定是否保存当前轮次 ###############################

        logger.info(f"第{epoch}轮验证集评估完成，开始判断是否保存本轮权重")
        if (epoch==start_epoch)and LOAD_CP==False:
            save_checkpoint(D,optimizer_D,epoch,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,val_psnrs_pr,val_ssims_pr,power_datas, f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
            logger.info(f"第一轮权重已保存")
            max_power_data= power_data            # 初始化最大效果值
            d_epoch_num = start_epoch             # 初始化效果最好的轮次数
        if (epoch==300):
            save_checkpoint(D,optimizer_D,epoch,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,val_psnrs_pr,val_ssims_pr,power_datas, f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
            logger.info(f"最后一轮权重已保存")
        else:
            if power_data > max_power_data:
                delpath=f'{project_root}/outputs/weights/{current_datetime}/{d_epoch_num}weights.pth' # 删除对应的权重
                if os.path.exists(delpath):
                    os.remove(delpath)
                    logger.info(f"删除了先前的第{d_epoch_num}轮权重")
                d_epoch_num=epoch                 # 更新效果最好的轮次数
                max_power_data= power_data        # 更新最好的效果值
                
                save_checkpoint(D,optimizer_D,epoch,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,val_psnrs_pr,val_ssims_pr,power_datas, f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
                logger.info(f"保存了当前的第{d_epoch_num}轮权重,最好效果为{max_power_data}")
            else :
                logger.info("不保存此轮权重")

        ################### 绘图 ############################ 
               
        logger.info("开始绘图")
        epochs = range(1, len(power_datas) + 1)
        allgraph(current_datetime, epochs,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,val_psnrs_pr,val_ssims_pr,power_datas)
        logger.info("绘图完成")
        logger.info(f"第{epoch}轮训练全部完成")

    logger.info("训练全部完成")

# 保存模型权重
def save_checkpoint(model,optimizer,epoch,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,val_psnrs_pr,val_ssims_pr,power_datas,path):
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
        "val_psnr_pr":val_psnrs_pr,
        "val_ssim_pr":val_ssims_pr,
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
        val_psnrs_pr=checkpoint['val_psnr_pr']
        val_ssims_pr=checkpoint['val_ssim_pr']
        power_datas=checkpoint['power_data']

        logger.info(f"已成功加载检查点权重,上次结束轮次为：{start_epoch},训练集损失值为:{train_losses[start_epoch-1]}")
        return model,optimizer,start_epoch,train_losses,train_psnrs,train_ssims,val_losses,val_psnrs,val_ssims,val_psnrs_pr,val_ssims_pr,power_datas
    else:
        logger.info("在所给的路径下未找到对应的权重文件，请检查后重试")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(42)  # 设置随机种子
    train()