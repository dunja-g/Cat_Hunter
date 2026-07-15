import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CatNonCatDataset(Dataset):
    def __init__(self, csv_file, transform=None, sample_fraction=1.0, seed=42):
        """
        csv_file: 数据划分 CSV 文件路径
        transform: 图像数据变换 (包含大小调整、增强、归一化等)
        sample_fraction: 随机抽样的比例 (0.0 到 1.0)。若小于 1.0，则仅使用部分数据，便于快速调试。
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # 如果需要子采样，主要用于 CPU 快速开发和测试
        if sample_fraction < 1.0:
            self.df = self.df.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["label"]
        
        # 读取图像并转换为 RGB 模式
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # 容错处理：如果图像损坏，返回一张纯黑背景图
            print(f"[WARNING] Failed to load image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(is_train=True):
    """
    获取数据处理流（包括缩放、旋转、水平翻转、亮度调整和归一化）
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            # 亮度微调 (brightness=0.2 相当于在 [0.8, 1.2] 之间随机调整)
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),  # ToTensor 会自动把像素归一化到 [0.0, 1.0]
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
