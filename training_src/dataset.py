import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CatNonCatDataset(Dataset):
    def __init__(self, csv_file, transform=None, sample_fraction=1.0, seed=42):
        """
        csv_file: path to the split CSV file
        transform: image transforms (resize, augmentation, normalisation, ...)
        sample_fraction: fraction of rows to sample (0.0 to 1.0). Below 1.0 only part of the data is used, for quick debugging.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Optional subsampling, mainly for quick development and testing on CPU
        if sample_fraction < 1.0:
            self.df = self.df.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["label"]
        
        # Load the image and convert it to RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # If the image is corrupt, fall back to a black image instead of crashing
            print(f"[WARNING] Failed to load image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(is_train=True):
    """
    Return the image transform pipeline (resize, random affine, horizontal flip, colour jitter, normalisation).
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),  # ToTensor scales pixel values to [0.0, 1.0]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
