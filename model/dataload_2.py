# 训练数据集准备
import torch 
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, sys

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 自定义数据集创建器
class Train_Dataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = os.path.abspath(images_dir)
        self.labels_dir = os.path.abspath(labels_dir)

        # 获取图片文件名列表，以便__len__()方法可以获得数据集大小
        self.image_filenames = sorted(os.listdir(self.images_dir))
        self.label_filenames = sorted(os.listdir(self.labels_dir))

        # 定义转换操作
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):     # 原Dataset类的内置方法
        # 构建图像和标签的绝对路径
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        label_path = os.path.join(self.labels_dir, self.label_filenames[idx])

        # 打开指定图像和标签图片，转为灰度后返回image和label对象并赋予image和label
        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("L")

        # 转换为张量
        image = self.to_tensor_no_normalize(image)       # 讲图片数据转换为张量数据，形状是[channel，height，width]
        label = self.to_tensor_no_normalize(label)

        return image, label
    
    @staticmethod
    def to_tensor_no_normalize(pic):
        if isinstance(pic, np.ndarray):     # 是numpy张量
            img = torch.from_numpy(pic)     # 转换为torch张量
        else:
            # 如果是 PIL 图像，转换为 NumPy 数组，然后转换为张量
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))    # 
            nchannel = 1 if pic.mode == 'L' else len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            img = img.permute(2, 0, 1).contiguous()
        return img.float()


class Val_Dataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = os.path.abspath(images_dir)
        self.labels_dir = os.path.abspath(labels_dir)

        # 获取图片文件名列表，以便__len__()方法可以获得数据集大小
        self.image_filenames = sorted(os.listdir(self.images_dir))
        self.label_filenames = sorted(os.listdir(self.labels_dir))

        # 定义转换操作
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):     # 原Dataset类的内置方法
        # 构建图像和标签的绝对路径
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        label_path = os.path.join(self.labels_dir, self.label_filenames[idx])

        # 打开指定图像和标签图片，转为灰度后返回image和label对象并赋予image和label
        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("L")

        # 转换为张量
        image = self.to_tensor_no_normalize(image)       # 讲图片数据转换为张量数据，形状是[channel，height，width]
        label = self.to_tensor_no_normalize(label)

        return image, label
    
    @staticmethod
    def to_tensor_no_normalize(pic):
        if isinstance(pic, np.ndarray):     # 是numpy张量
            img = torch.from_numpy(pic)     # 转换为torch张量
        else:
            # 如果是 PIL 图像，转换为 NumPy 数组，然后转换为张量
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))    # 
            nchannel = 1 if pic.mode == 'L' else len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            img = img.permute(2, 0, 1).contiguous()
        return img.float()
    


def data_prepare_2(batchsize):
    # 使用绝对路径
    train_images_dir = os.path.join(project_root, "datasets/1")
    train_labels_dir = os.path.join(project_root, "datasets/2")

    val_image_dir= os.path.join(project_root, "datasets/3")
    val_lable_dir= os.path.join(project_root, "datasets/4")
    
    train_set = Train_Dataset(train_images_dir, train_labels_dir)
    val_set = Val_Dataset(val_image_dir, val_lable_dir)

    train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    val_dataloader=DataLoader(val_set, batch_size=batchsize, shuffle=True)
    return train_dataloader,val_dataloader