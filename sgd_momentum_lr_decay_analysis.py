import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from config import device, TRAIN_CONFIG, OPTIMIZERS_CONFIG, TENSORBOARD_CONFIG
from data.data_loader import get_data_loaders
from training.trainer import ModelTrainer
from optimizer_5layer_non_sgd_normalize05 import CNN5Layers
from resultShow import analyze_per_class_performance


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def main():
    """使用 SGD(带动量) + 学习率衰减训练 5 层 CNN，并做按类别分析。"""
    set_seed(TRAIN_CONFIG["random_seed"])
    epochs = TRAIN_CONFIG["epochs"]

    # 1. 数据预处理：Normalize_0.5
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_loader, test_loader = get_data_loaders(transform=transform)

    # 2. 模型
    model = CNN5Layers().to(device)

    # 3. 损失函数和优化器（使用 config.py 中 SGD 的默认参数，包括动量）
    criterion = nn.CrossEntropyLoss()
    sgd_cfg = OPTIMIZERS_CONFIG["SGD"]
    optimizer = optim.SGD(model.parameters(), **sgd_cfg["params"])

    # 4. 学习率调度器（每 10 个 epoch 将学习率衰减为原来的 0.1 倍）
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # 5. 训练
    log_dir = os.path.join(
        TENSORBOARD_CONFIG["log_dir"], "sgd_momentum_lr_decay_cnn5_norm05"
    )
    trainer = ModelTrainer(
        model,
        device,
        optimizer_name="SGD_momentum_lr_decay_CNN5_Norm05",
        use_tensorboard=True,
        log_dir=log_dir,
    )

    print("开始使用 SGD(带动量) + 学习率衰减 训练 CNN5Layers ...")
    result = trainer.train(
        train_loader,
        test_loader,
        optimizer,
        criterion,
        epochs,
        scheduler=scheduler,
    )

    print(
        f"训练结束! 最终测试准确率: {result['final_test_accuracy']:.2f}% | 用时: {result['training_time']:.2f}s"
    )

    # 6. 按类别可视化分类情况，找出最难分类的类别
    print("\n开始按类别分析分类性能，并绘制混淆矩阵...")
    analyze_per_class_performance(model, test_loader, device, top_k_worst=3)


if __name__ == "__main__":
    main()
