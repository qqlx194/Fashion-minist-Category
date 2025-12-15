import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import DATA_CONFIG, CLASS_NAMES

def get_data_loaders(batch_size=None, transform=None):
    """
    获取Fashion-MNIST数据加载器
    """
    if batch_size is None:
        batch_size = DATA_CONFIG['batch_size']
    
    if transform is None:
        # 训练集增强
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # 测试集不增强
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = transform
        test_transform = transform
    
    # 加载数据集
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=DATA_CONFIG['shuffle'],
        num_workers=DATA_CONFIG.get('num_workers', 0)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=DATA_CONFIG.get('num_workers', 0)
    )
    
    return train_loader, test_loader

def get_class_names():
    """获取类别名称"""
    return CLASS_NAMES