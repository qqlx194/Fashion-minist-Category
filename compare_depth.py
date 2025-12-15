import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from config import device, TRAIN_CONFIG, OPTIMIZERS_CONFIG
from data.data_loader import get_data_loaders
from training.trainer import ModelTrainer


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 1. 定义不同深度的 CNN 模型
# -----------------------------------------------------------------------------

class CNN2Layers(nn.Module):
    """2层卷积 (SimpleCNN)"""
    def __init__(self):
        super(CNN2Layers, self).__init__()
        # 28x28 -> 14x14 -> 7x7
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN3Layers(nn.Module):
    """3层卷积"""
    def __init__(self):
        super(CNN3Layers, self).__init__()
        # 28x28 -> 14x14 -> 7x7 -> 3x3
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 经过3次池化: 28->14->7->3 (7/2=3)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN4Layers(nn.Module):
    """4层卷积版本的 FashionCNN"""
    def __init__(self):
        super(CNN4Layers, self).__init__()
        # 28x28 -> conv1 -> 28x28 -> pool -> 14x14
        # 14x14 -> conv2 -> 14x14 -> conv3 -> 14x14 -> pool -> 7x7
        # 7x7  -> conv4 -> 7x7
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1: conv1 + pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Block 2: conv2 + conv3 + pool
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Block 3: conv4
        x = F.relu(self.conv4(x))

        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN10Layers(nn.Module):
    """10层卷积 (深度过深，可能导致梯度消失或过拟合)"""
    def __init__(self):
        super(CNN10Layers, self).__init__()
        
        # 为了堆叠10层且不让尺寸变成0，我们只在部分层做池化，或者用 padding 保持尺寸
        # 这里采用：前2层池化一次，中间再池化一次，后面不再池化，靠 padding=1 维持尺寸
        
        self.features = nn.Sequential(
            # Block 1: 2层卷积 + 1次池化 (28->14)
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 2层卷积 + 1次池化 (14->7)
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 2层卷积 (7->7)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            
            # Block 4: 2层卷积 (7->7)
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            
            # Block 5: 2层卷积 (7->7)
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        
        # 最终尺寸 7x7，通道 64
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------------------------------------------------------
# 2. 实验运行逻辑
# -----------------------------------------------------------------------------

def run_experiment(model_class, tag, epochs, optimizer_name, optimizer_config):
    print(f"\n===== 实验：{tag} - {optimizer_name}（epochs = {epochs}）=====")
    
    # 统一使用默认预处理
    train_loader, test_loader = get_data_loaders()

    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 根据配置创建优化器（支持所有在 OPTIMIZERS_CONFIG 中定义的优化器）
    params = optimizer_config['params']
    if optimizer_name in ['SGD', 'SGD_no_momentum']:
        # SGD_no_momentum 也是用 SGD，只是 momentum 配置为 0
        optimizer = optim.SGD(model.parameters(), **params)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), **params)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), **params)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), **params)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), **params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # 使用不同的 log_dir 来区分实验
    log_dir = f"runs/depth_experiment/{tag}_{optimizer_name}"
    
    trainer = ModelTrainer(
        model, 
        device, 
        optimizer_name=f"{tag}_{optimizer_name}", 
        use_tensorboard=True,
        log_dir=log_dir
    )
    result = trainer.train(train_loader, test_loader, optimizer, criterion, epochs)

    print(f"[{tag} - {optimizer_name}] 最终测试准确率: {result['final_test_accuracy']:.2f}%")
    print(f"[{tag} - {optimizer_name}] 训练耗时: {result['training_time']:.2f} 秒")
    
    return result


def main():
    set_seed(TRAIN_CONFIG["random_seed"])
    # 为了快速演示，这里可以减少 epoch 数，或者使用 TRAIN_CONFIG["epochs"]
    epochs = 10 

    # 定义要对比的模型（3 层 vs 4 层 FashionCNN 风格）
    models_config = {
        "FashionCNN_3_Layers": CNN3Layers,
        "FashionCNN_4_Layers": CNN4Layers,
    }
    
    # 定义要对比的优化器：使用 config.py 中的所有优化器配置
    # 这里假设你已经在主实验中调好了每个优化器的最佳学习率，并更新到了 OPTIMIZERS_CONFIG
    optimizers_to_compare = OPTIMIZERS_CONFIG

    results = {}

    for model_tag, model_cls in models_config.items():
        results[model_tag] = {}
        for opt_name, opt_config in optimizers_to_compare.items():
            res = run_experiment(model_cls, model_tag, epochs, opt_name, opt_config)
            results[model_tag][opt_name] = res

    # 打印对比表
    print("\n" + "="*80)
    print(f"{'模型深度':<15} | {'优化器':<10} | {'测试准确率':<15} | {'训练时间':<10}")
    print("-" * 80)
    for model_tag in models_config.keys():
        for opt_name in optimizers_to_compare.keys():
            r = results[model_tag][opt_name]
            print(f"{model_tag:<15} | {opt_name:<10} | {r['final_test_accuracy']:>14.2f}% | {r['training_time']:>8.2f}s")

    # 画分组柱状图
    plot_depth_optimizer_comparison(results, models_config.keys(), optimizers_to_compare.keys())

def plot_depth_optimizer_comparison(results, model_tags, opt_names):
    model_tags = list(model_tags)
    opt_names = list(opt_names)
    
    n_models = len(model_tags)
    n_opts = len(opt_names)
    
    # 设置柱状图宽度
    bar_width = 0.35
    index = np.arange(n_models)
    
    plt.figure(figsize=(10, 6))
    
    for i, opt_name in enumerate(opt_names):
        accuracies = [results[tag][opt_name]['final_test_accuracy'] for tag in model_tags]
        plt.bar(index + i * bar_width, accuracies, bar_width, label=opt_name)
        
        # 在柱子上显示数值
        for j, v in enumerate(accuracies):
            plt.text(index[j] + i * bar_width, v + 1, f"{v:.1f}%", ha='center', fontsize=9)

    plt.xlabel('Network Depth')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Effect of Network Depth & Optimizer on Accuracy')
    plt.xticks(index + bar_width / 2 * (n_opts - 1), model_tags)
    plt.legend()
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
