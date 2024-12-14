import torch
import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)
from model import DeGenerater, data_prepare
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from datetime import datetime

import numpy as np
from PIL import Image


# 参数设定
BATCHSIZE = 1
weight_name="12-14_01-50/237weights"
weight_PATH=f'{project_root}/outputs/weights/{weight_name}.pth' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_datetime = datetime.now().strftime("%m-%d_%H-%M")
log_file_path=f'{project_root}/outputs/推理结果/{current_datetime}/推理日志.txt'
os.makedirs(os.path.dirname(log_file_path),exist_ok=True)
logger.add(log_file_path,rotation="500 MB",level="INFO")    # 添加日志目标

def detect():
    # 训练配置
    dataloader = data_prepare(BATCHSIZE)
    checkpoint = torch.load(weight_PATH, map_location=device)
    D = DeGenerater()
    D.load_state_dict(checkpoint['model_state_dict'])
    D.to(device)
    D.eval()
    result=[]
    logger.info(f"weight:{weight_name}")
    logger.info("推理开始")
    with torch.no_grad():           # 不需要梯度
        for image ,label in dataloader:
            image = label.to(device)
            output=D(image)
            output=output.squeeze().cpu().numpy()
            result.append(output)
            #logger.info(f"推理结果: {output}")
            output_image = Image.fromarray((output * 255).astype(np.uint8), mode='L')
            output_image_path = f'{project_root}/outputs/推理结果/{current_datetime}/推理结果_{len(result)}.png'
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            output_image.save(output_image_path)
            logger.info(f"推理结果图像已保存: {output_image_path}")

if __name__=="__main__":
    detect()
