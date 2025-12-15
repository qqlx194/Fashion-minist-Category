import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms

from config import device, TRAIN_CONFIG, OPTIMIZERS_CONFIG
from data.data_loader import get_data_loaders
from training.trainer import ModelTrainer


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


class CNN5Layers(nn.Module):
    """5 层卷积 CNN，用于本次实验

    结构：
    - Block1: conv1, conv2, pool  (28 -> 14)
    - Block2: conv3, conv4, pool  (14 -> 7)
    - Block3: conv5               (7 -> 7)
    - FC: 64 * 7 * 7 -> 128 -> 10
    """

    def __init__(self):
        super(CNN5Layers, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 28 -> 14

        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # 14 -> 7

        # Block 3
        x = F.relu(self.conv5(x))  # 7 -> 7

        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_optimizer(optimizer_name, params, model_parameters):
    """根据配置创建优化器（不修改 config 中的默认学习率等参数）"""
    if optimizer_name == "SGD":
        return optim.SGD(model_parameters, **params)
    if optimizer_name == "Adam":
        return optim.Adam(model_parameters, **params)
    if optimizer_name == "RMSprop":
        return optim.RMSprop(model_parameters, **params)
    if optimizer_name == "Adagrad":
        return optim.Adagrad(model_parameters, **params)
    if optimizer_name == "AdamW":
        return optim.AdamW(model_parameters, **params)
    raise ValueError(f"不支持的优化器: {optimizer_name}")


def run_experiment(optimizer_tag, optimizer_config, train_loader, test_loader, epochs):
    print(f"\n===== 实验：CNN5Layers + {optimizer_tag}（epochs = {epochs}）=====")

    model = CNN5Layers().to(device)
    criterion = nn.CrossEntropyLoss()

    base_name = optimizer_config["optimizer"]
    params = optimizer_config["params"]
    optimizer = create_optimizer(base_name, params, model.parameters())

    from config import TENSORBOARD_CONFIG
    import os

    log_dir = os.path.join(
        TENSORBOARD_CONFIG["log_dir"], f"cnn5_norm05_{optimizer_tag}"
    )

    trainer = ModelTrainer(
        model,
        device,
        optimizer_name=f"CNN5_{optimizer_tag}",
        use_tensorboard=True,
        log_dir=log_dir,
    )

    result = trainer.train(train_loader, test_loader, optimizer, criterion, epochs)

    print(
        f"[CNN5Layers + {optimizer_tag}] 最终测试准确率: {result['final_test_accuracy']:.2f}%"
    )
    print(
        f"[CNN5Layers + {optimizer_tag}] 训练耗时: {result['training_time']:.2f} 秒"
    )

    return result


def main():
    set_seed(TRAIN_CONFIG["random_seed"])
    epochs = TRAIN_CONFIG["epochs"]

    # 选择除 "SGD" 之外的所有优化器（包含 SGD_no_momentum）
    optimizers_to_compare = {
        name: cfg
        for name, cfg in OPTIMIZERS_CONFIG.items()
        if name != "SGD"
    }

    print("将比较的优化器:", list(optimizers_to_compare.keys()))

    # 统一使用同一套预处理：ToTensor + Normalize(0.5, 0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # 只构造一次 DataLoader，供所有优化器共享
    train_loader, test_loader = get_data_loaders(transform=transform)

    results = {}
    for opt_name, opt_config in optimizers_to_compare.items():
        results[opt_name] = run_experiment(
            opt_name, opt_config, train_loader, test_loader, epochs
        )

    # 打印汇总表
    print("\n" + "=" * 70)
    print(f"{'优化器':<12} | {'最终测试准确率':<16} | {'训练时间':<10}")
    print("-" * 70)
    for opt_name, r in results.items():
        print(
            f"{opt_name:<12} | {r['final_test_accuracy']:>15.2f}% | {r['training_time']:>8.2f}s"
        )

    # 简单画一个柱状图对比不同优化器的最终准确率
    labels = list(results.keys())
    accs = [results[name]["final_test_accuracy"] for name in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, accs, color="skyblue")
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Optimizer")
    plt.title("CNN5Layers on Fashion-MNIST (Adam/Adagrad: NoNormalize)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
