import torch
from model import DeGenerater, data_prepare
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from datetime import datetime
import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)

# 参数设定
BATCHSIZE = 1

weight_PATH= f'{project_root}/outputs/weights/09-05_21-18/3weights.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_datetime = datetime.now().strftime("%m-%d_%H-%M")
log_file_path=f'{project_root}/outputs/推理结果/{current_datetime}/推理日志.txt'
os.makedirs(os.path.dirname(log_file_path),exist_ok=True)
logger.add(log_file_path,rotation="500 MB",level="INFO")    # 添加日志目标

def detect():
    # 训练配置
    dataloader = data_prepare(BATCHSIZE)
    D = DeGenerater()
    D.to(device)
    optimizer_D = optim.Adam(D.parameters(), lr=0.00005)
    D.eval()
    result=[]
    logger.info("推理开始")
    with torch.no_grad():           # 不需要梯度
        for image ,label in dataloader:
            image = image.to(device)
            output=D(image)
            output=output.squeeze().cpu().numpy()


