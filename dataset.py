import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class SkinCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # 获取所有图片路径和标签
        self.images = []
        self.labels = []
        
        # 加载良性图片
        benign_dir = os.path.join(root_dir, 'benign_segmented')
        benign_images = [(os.path.join(benign_dir, img), 0) for img in os.listdir(benign_dir)]
        
        # 加载恶性图片
        malignant_dir = os.path.join(root_dir, 'malignant_segmented')
        malignant_images = [(os.path.join(malignant_dir, img), 1) for img in os.listdir(malignant_dir)]
        
        # 合并数据
        all_images = benign_images + malignant_images
        np.random.shuffle(all_images)
        
        # 划分训练集和测试集
        split = int(len(all_images) * 0.8)
        if train:
            all_images = all_images[:split]
        else:
            all_images = all_images[split:]
            
        self.images = [x[0] for x in all_images]
        self.labels = [x[1] for x in all_images]
        
        # 默认转换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    创建训练集和测试集的数据加载器
    """
    train_dataset = SkinCancerDataset(data_dir, train=True)
    test_dataset = SkinCancerDataset(data_dir, train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader 