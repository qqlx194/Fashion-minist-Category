import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet import FashionResNet18
from data.data_loader import get_data_loaders
from training.trainer import ModelTrainer
from config import device, TRAIN_CONFIG

def main():
    print(f"Using device: {device}")
    
    # 1. 准备数据
    print("Loading data...")
    train_loader, test_loader = get_data_loaders()
    
    # 2. 初始化模型
    print("Initializing ResNet-18 model...")
    model = FashionResNet18().to(device)
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 初始化训练器
    # 使用 TensorBoard 记录日志，日志目录为 runs/resnet18_experiment
    trainer = ModelTrainer(
        model=model,
        device=device,
        optimizer_name="ResNet18_Adam",
        log_dir="runs/resnet18_experiment"
    )
    
    # 5. 开始训练
    epochs = TRAIN_CONFIG['epochs']
    print(f"Starting training for {epochs} epochs...")
    
    trainer.train(train_loader, test_loader, optimizer, criterion, epochs)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
