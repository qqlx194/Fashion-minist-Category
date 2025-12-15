import torch
import numpy as np
from config import device, TRAIN_CONFIG, OPTIMIZERS_CONFIG
from data.data_loader import get_data_loaders
from training.optimizer_comparison import OptimizerComparator
from utils.visualization import plot_optimizer_comparison

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Fashion-MNIST - 快速寻找最佳学习率")
    print(f"使用设备: {device}")
    print("=" * 60)
    
    # 加载数据
    print("加载数据...")
    train_loader, test_loader = get_data_loaders()
    
    # 比较优化器
    print("开始搜索最佳学习率...")
    comparator = OptimizerComparator(device)
    
    # 关键配置：
    # tune_lr=True: 开启网格搜索
    # lr_tune_epochs=5: 每个候选 LR 跑 5 个 epoch
    # epochs=1: 选出最佳 LR 后，只跑 1 个 epoch 验证一下即可，不用跑完 30 轮
    results = comparator.compare_optimizers(
        train_loader, test_loader,
        optimizers_config=OPTIMIZERS_CONFIG,
        tune_lr=True,       
        lr_tune_epochs=5,
        epochs=1,
        seeds=[42] # 只用一个种子快速验证
    )
    
    # 显示结果摘要
    print("\n" + "="*60)
    print("学习率搜索结果摘要")
    print("="*60)
    print(f"{'优化器':<15} | {'最佳学习率':<15} | {'验证准确率 (1 epoch)':<20}")
    print("-" * 60)
    
    for opt_name, res in results.items():
        best_lr = res['best_lr']
        final_acc = res['final_test_accuracy']
        print(f"{opt_name:<15} | {best_lr:<15} | {final_acc:>18.2f}%")

if __name__ == "__main__":
    main()
