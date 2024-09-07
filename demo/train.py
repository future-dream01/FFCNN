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

# 参数设定
EPOCHES = 200
BATCHSIZE = 1
LOAD_CP=False     # 是否需要加载之前的检查点

CP_PATH= f'{project_root}/outputs/weights/09-05_21-18/3weights.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_datetime = datetime.now().strftime("%m-%d_%H-%M")
log_file_path=f'{project_root}/outputs/训练与性能情况/{current_datetime}/损失日志.txt'
os.makedirs(os.path.dirname(log_file_path),exist_ok=True)
logger.add(log_file_path,rotation="500 MB",level="INFO")    # 添加日志目标
def train():
    # 训练配置
    dataloader = data_prepare(BATCHSIZE)
    D = DeGenerater()
    D.to(device)
    optimizer_D = optim.Adam(D.parameters(), lr=0.00005)
    start_epoch=1
    if LOAD_CP:
        D,optimizer_D,start_epoch,loss=load_checkpoint(D,optimizer_D,CP_PATH)
        start_epoch+=1
    scaler=GradScaler()
    logger.info("开始训练")
    losses = []
    # 开始训练
    for epoch in range(int(start_epoch), EPOCHES + 1):
        D.train()
        loss_epoch = 0
        batches = 0  # 批次计数
        os.makedirs(f'{project_root}/outputs/训练与性能情况/{current_datetime}', exist_ok=True)
        os.makedirs(f'{project_root}/outputs/weights/{current_datetime}', exist_ok=True)
        for images, labels in dataloader:
            # 每次迭代生成新的输入和输出
            images, labels = images.to(device), labels.to(device)
            optimizer_D.zero_grad()  # 梯度归零
            # 前向传播
            with autocast():
                output = D(images)
                loss1 = loss_TOTAL(device, output, labels)
            # 反向传播，并不保留计算图
            scaler.scale(loss1).backward()
            scaler.step(optimizer_D)
            scaler.update()
            batches += 1
            loss_epoch += loss1.item()  # 累加损失
            logger.info(f"epoch:{epoch},batch:{batches},loss_all:{loss1.item()}")
        loss_epoch = loss_epoch / batches  # 本epoch平均损失
        losses.append(loss_epoch)
        logger.info(f"epoch:{epoch},loss_average:{loss_epoch}")
        if epoch==start_epoch:
            save_checkpoint(D,optimizer_D,epoch,loss_epoch, f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
            min_loss= loss_epoch            # 初始化最小损失
            d_epoch_num = start_epoch       # 初始化删除的轮次数
        else:
            if loss_epoch < min_loss:
                os.remove(f'{project_root}/outputs/weights/{current_datetime}/{d_epoch_num}weights.pth') # 删除对应的权重
                logger.info(f"删除了先前的第{d_epoch_num}轮次权重")
                d_epoch_num=epoch           # 更新删除的轮次数
                min_loss= loss_epoch        # 更新最小损失
                save_checkpoint(D,optimizer_D,epoch,loss_epoch, f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
        logger.info("当前轮次训练完成，权重已保存")
    logger.info("训练全部完成")
    # 作性能曲线、保存训练权重
    epochs = range(1, len(losses) + 1)
    loss_graph(current_datetime, epochs, losses)
    logger.info("性能曲线、训练日志、模型权重成功保存")

# 保存模型权重
def save_checkpoint(model,optimizer,epoch,loss,path):
    state={
        "epoch":epoch,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "loss":loss,
    }
    torch.save(state,path)

# 获取断点的轮次数、损失值
def load_checkpoint(model,optimizer,path):
    if os.path.isfile(path):
        checkpoint=torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch=checkpoint['epoch']
        loss=checkpoint['loss']
        logger.info(f"检查点最后一次损失值为:{loss}")
        return model,optimizer,start_epoch,loss
    else:
        logger.info("在所给的路径下未找到对应的权重文件，请检查后重试")

if __name__ == '__main__':
    train()
