import os
import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image

# 创建自己的数据集类，用于统一数据集图片尺寸
class Data_prepare(Dataset):
    def __init__(self,img_dir,):
        super(Data_prepare,self).__init__()
        self.tranform=transforms.Compose([transforms.Resize((512*512),interpolation=Image.BICUBIC),transforms.ToTensor(),])
        self.img_names = [img for img in os.listdir(img_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])    # 该图片的路径
        image = Image.open(img_path).convert("L")                       # 确保是灰度格式
        image=self.tranform(image)                                      # 将图片大小转换为 512*512，统一尺寸
        return image

def load_data(path,batch_size):
    dataset=Data_prepare(path)      # 数据集
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True, num_workers=4)
    return dataloader








