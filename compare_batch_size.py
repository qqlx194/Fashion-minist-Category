import torch
import numpy as np
import matplotlib.pyplot as plt
from config import device, TRAIN_CONFIG, OPTIMIZERS_CONFIG, DATA_CONFIG
from data.data_loader import get_data_loaders
from training.optimizer_comparison import OptimizerComparator
from models.cnn_model import SimpleCNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders_with_batch_size(batch_size):
    """获取指定batch size的数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=DATA_CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=DATA_CONFIG['num_workers'])
    
    return train_loader, test_loader

def compute_convergence_epoch(mean_test_accuracies, threshold_ratio=0.9):
    """根据测试集准确率曲线，计算达到最终准确率一定比例所需的 epoch 数。

    默认使用最终准确率的 90% 作为收敛阈值，返回第一个达到该阈值的 epoch（1-based）。
    如果始终未达到，则返回最后一个 epoch。
    """
    final_acc = mean_test_accuracies[-1]
    threshold = final_acc * threshold_ratio
    for idx, acc in enumerate(mean_test_accuracies):
        if acc >= threshold:
            return idx + 1  # epoch 从 1 开始计
    return len(mean_test_accuracies)


def main():
    # 设置随机种子
    torch.manual_seed(TRAIN_CONFIG['random_seed'])
    np.random.seed(TRAIN_CONFIG['random_seed'])

    print("Fashion-MNIST - Batch Size 敏感性分析 (SGD vs Adam)")
    print(f"使用设备: {device}")
    print("=" * 60)

    # 1. 固定模型和 epoch 数
    # 模型在 OptimizerComparator 内部固定为 FashionCNN
    fixed_epochs = TRAIN_CONFIG['epochs']  # 与主实验保持一致

    # 2. 定义要比较的 Batch Sizes
    batch_sizes = [32, 128, 1024]

    # 仅比较 SGD 和 Adam 两个优化器
    target_optimizers = {
        'SGD': OPTIMIZERS_CONFIG['SGD'],
        'Adam': OPTIMIZERS_CONFIG['Adam']
    }

    print(f"比较优化器: {list(target_optimizers.keys())}")
    print(f"比较 Batch Sizes: {batch_sizes}")
    print(f"每个组合的训练 epoch 数: {fixed_epochs}")

    # 每个 batch size 独立创建 comparator，避免结果互相覆盖
    all_results = {}

    for bs in batch_sizes:
        print(f"\n>>> 正在测试 Batch Size = {bs} ...")
        train_loader, test_loader = get_data_loaders_with_batch_size(bs)

        # 使用 SimpleCNN 作为比较用的模型
        comparator = OptimizerComparator(device, model_class=SimpleCNN, use_tensorboard=True)

        # 2）对 SGD 和 Adam，在当前 batch size 下自动调参学习率
        #    tune_lr=True 会调用内部的 _find_best_lr，为每个优化器在该 batch 上做网格搜索
        results = comparator.compare_optimizers(
            train_loader,
            test_loader,
            optimizers_config=target_optimizers,
            epochs=fixed_epochs,
            seeds=[42],          # 固定一个种子，专注于 batch size 与优化器的交互
            tune_lr=True,
            lr_tune_epochs=5     # 每个候选 lr 调参 5 个 epoch
        )

        # 计算并附加收敛速度指标（达到最终准确率 90% 所需 epoch）
        for opt_name, res in results.items():
            mean_test_acc = res['mean_test_accuracies']
            conv_epoch = compute_convergence_epoch(mean_test_acc, threshold_ratio=0.9)
            res['convergence_epoch_90pct'] = conv_epoch

        all_results[bs] = results

    # 3）打印摘要：最终准确率 + 收敛速度
    print("\n" + "=" * 60)
    print("Batch Size 敏感性分析摘要 (SGD vs Adam)")
    print("=" * 60)
    print(f"{'Batch Size':<12} | {'优化器':<10} | {'测试准确率':<15} | {'收敛Epoch(90%)':<15} | {'训练时间':<10}")
    print("-" * 80)

    for bs in batch_sizes:
        for opt_name, res in all_results[bs].items():
            print(
                f"{bs:<12} | {opt_name:<10} | "
                f"{res['final_test_accuracy']:>14.2f}% | "
                f"{res['convergence_epoch_90pct']:>14d} | "
                f"{res['mean_training_time']:>8.2f}秒"
            )

    # 4）绘制 Batch Size -> 最终准确率 / 收敛速度 曲线
    plot_batch_size_comparison(all_results, batch_sizes)

def plot_batch_size_comparison(all_results, batch_sizes):
    """绘制 Batch Size 对比图：

    左图：Batch Size -> 最终测试准确率
    右图：Batch Size -> 收敛速度（达到最终准确率 90% 所需 epoch）
    """
    plt.figure(figsize=(12, 5))

    # 1. 最终准确率对比
    plt.subplot(1, 2, 1)
    for opt_name in ['SGD', 'Adam']:
        accs = [all_results[bs][opt_name]['final_test_accuracy'] for bs in batch_sizes]
        plt.plot(batch_sizes, accs, marker='o', label=opt_name)

    plt.xscale('log')
    plt.xlabel('Batch Size (log scale)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Batch Size vs Final Test Accuracy')
    plt.legend()
    plt.grid(True)

    # 2. 收敛速度对比（越低表示越快收敛）
    plt.subplot(1, 2, 2)
    for opt_name in ['SGD', 'Adam']:
        conv_epochs = [all_results[bs][opt_name]['convergence_epoch_90pct'] for bs in batch_sizes]
        plt.plot(batch_sizes, conv_epochs, marker='s', linestyle='--', label=opt_name)

    plt.xscale('log')
    plt.xlabel('Batch Size (log scale)')
    plt.ylabel('Epochs to reach 90% of final acc')
    plt.title('Batch Size vs Convergence Speed (lower is better)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('batch_size_comparison.png')
    print("\n图表已保存至 batch_size_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
