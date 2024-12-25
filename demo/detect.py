import torch
import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)
from model import DeGenerater, data_prepare_2
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from datetime import datetime

import numpy as np
from PIL import Image


# 参数设定
BATCHSIZE = 1
weight_name="12-25_13-20/315weights"
weight_PATH=f'{project_root}/outputs/weights/{weight_name}.pth' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_datetime = datetime.now().strftime("%m-%d_%H-%M")
log_file_path=f'{project_root}/outputs/推理结果/{current_datetime}/推理日志.txt'
os.makedirs(os.path.dirname(log_file_path),exist_ok=True)
logger.add(log_file_path,rotation="500 MB",level="INFO")    # 添加日志目标

def detect():
    # 训练配置
    train_dataloader, val_dataloader = data_prepare_2(BATCHSIZE)
    checkpoint = torch.load(weight_PATH, map_location=device)
    D = DeGenerater()
    D.load_state_dict(checkpoint['model_state_dict'])
    D.to(device)
    D.eval()
    result = []
    logger.info(f"weight:{weight_name}")
    logger.info("推理开始")
    with torch.no_grad():  # 不需要梯度
        for image, label in val_dataloader:
            image = label.to(device)
            output = D(image)
            output = output.squeeze().cpu().numpy()
            #logger.info(f"推理结果: {output}")
            result.append(output)
            
            # 将输出的128x128灰度图截取图像紧贴右侧边界的121x115像素大小的区域
            cropped_output = output[:, -121:]  # 截取右侧121列
            cropped_output = cropped_output[6:121-6, :]  # 上下对称截取115行
            
            # 将结果放大到128x128
            cropped_image = Image.fromarray((cropped_output * 255).astype(np.uint8), mode='L')
            resized_image = cropped_image.resize((128, 128), Image.BILINEAR)
            
            output_image_path = f'{project_root}/outputs/推理结果/{current_datetime}/推理结果_{len(result)}.png'
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            resized_image.save(output_image_path)
            logger.info(f"推理结果图像已保存: {output_image_path}")

if __name__=="__main__":
    detect()
