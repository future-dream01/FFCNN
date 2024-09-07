# 训练数据集准备
import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, sys

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)

class ImageLabelDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = os.path.abspath(images_dir)
        self.labels_dir = os.path.abspath(labels_dir)

        # 获取图片文件名列表
        self.image_filenames = sorted(os.listdir(self.images_dir))
        self.label_filenames = sorted(os.listdir(self.labels_dir))

        # 定义转换操作
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # 构建图像和标签的完整路径
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        label_path = os.path.join(self.labels_dir, self.label_filenames[idx])

        # 打开图像和标签
        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("L")

        # 转换为张量
        image = self.transform(image)
        label = self.transform(label)

        return image, label
    
def data_prepare(batchsize):
    # 使用绝对路径
    images_dir = os.path.join(project_root, "datasets/images")
    labels_dir = os.path.join(project_root, "datasets/labels")
    
    train_set = ImageLabelDataset(images_dir, labels_dir)
    dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    return dataloader