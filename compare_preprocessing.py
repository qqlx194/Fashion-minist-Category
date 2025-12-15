import os
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from config import device, TRAIN_CONFIG, TENSORBOARD_CONFIG, OPTIMIZERS_CONFIG
from data.data_loader import get_data_loaders
from models.cnn_model import SimpleCNN
from training.trainer import ModelTrainer
from utils.tensorboard_logger import TensorBoardLogger


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def create_optimizer(optimizer_name, params, model_parameters):
    """根据名称和参数创建优化器"""
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


def run_experiment(transform, preprocess_tag, optimizer_tag, optimizer_config, epochs):
    """使用给定的 transform 和优化器训练一次模型，并返回结果"""
    print(f"\n===== 实验：{preprocess_tag} + {optimizer_tag}（epochs = {epochs}）=====")

    train_loader, test_loader = get_data_loaders(transform=transform)

    # 使用 SimpleCNN 作为对比预处理效果的模型
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # optimizer_config 中保存了 optimizer 名称和具体参数
    base_name = optimizer_config["optimizer"]
    params = optimizer_config["params"]
    optimizer = create_optimizer(base_name, params, model.parameters())

    log_dir = os.path.join(
        TENSORBOARD_CONFIG["log_dir"], f"preprocess_{preprocess_tag}_{optimizer_tag}"
    )
    trainer = ModelTrainer(
        model,
        device,
        optimizer_name=f"{preprocess_tag}_{optimizer_tag}",
        use_tensorboard=True,
        log_dir=log_dir,
    )
    result = trainer.train(train_loader, test_loader, optimizer, criterion, epochs)

    print(
        f"[{preprocess_tag} + {optimizer_tag}] 最终测试准确率: {result['final_test_accuracy']:.2f}%"
    )
    print(
        f"[{preprocess_tag} + {optimizer_tag}] 训练耗时: {result['training_time']:.2f} 秒"
    )

    return result


def main():
    set_seed(TRAIN_CONFIG["random_seed"])
    epochs = TRAIN_CONFIG["epochs"]
    # 定义多种预处理方案: 名称 -> (描述, transform)
    preprocess_configs = {
        "NoNormalize": (
            "ToTensor",
            transforms.Compose([
                transforms.ToTensor(),
            ]),
        ),
        "Normalize_0.5": (
            "ToTensor + Normalize(0.5,0.5)",
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        ),
        "CropFlip_Norm": (
            "RandomCrop + HFlip + Norm",
            transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        ),
        "Affine_Norm": (
            "RandomAffine + Norm",
            transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        ),
        "Erase_Norm": (
            "Norm + RandomErasing",
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.RandomErasing(p=0.2),
            ]),
        ),
    }

    results = {}

    # 依次运行所有预处理方案 + 所有优化器
    for preprocess_tag, (desc, tfm) in preprocess_configs.items():
        results[preprocess_tag] = {}
        for optimizer_tag, optimizer_config in OPTIMIZERS_CONFIG.items():
            results[preprocess_tag][optimizer_tag] = run_experiment(
                tfm, preprocess_tag, optimizer_tag, optimizer_config, epochs
            )

    # 打印对比表
    print("\n预处理方案\t优化器\t\t预处理描述\t\t\t测试集准确率 (%)\t训练时间 (s)")
    for preprocess_tag, (desc, _) in preprocess_configs.items():
        for optimizer_tag in OPTIMIZERS_CONFIG.keys():
            r = results[preprocess_tag][optimizer_tag]
            print(
                f"{preprocess_tag}\t{optimizer_tag}\t{desc}\t{r['final_test_accuracy']:.2f}\t\t{r['training_time']:.2f}"
            )

    # 画测试集准确率柱状图（每个柱对应一个 预处理+优化器 组合）
    labels = []
    accuracies = []
    for preprocess_tag in preprocess_configs.keys():
        for optimizer_tag in OPTIMIZERS_CONFIG.keys():
            labels.append(f"{preprocess_tag}\n{optimizer_tag}")
            accuracies.append(
                results[preprocess_tag][optimizer_tag]["final_test_accuracy"]
            )

    plt.figure(figsize=(10, 6))
    plt.bar(labels, accuracies, color="skyblue")
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Preprocessing Scheme + Optimizer")
    plt.title(
        "Effect of Different Preprocessing and Optimizers on Test Accuracy (FashionCNN)"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()