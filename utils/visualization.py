import matplotlib.pyplot as plt
import numpy as np
from data.data_loader import get_data_loaders, get_class_names

def show_samples(train_loader, num_samples=10):
    """显示训练样本"""
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    class_names = get_class_names()
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(min(num_samples, 10)):
        ax = axes[i//5, i%5]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'{class_names[labels[i]]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_optimizer_comparison(results):
    """绘制优化器比较结果 (支持误差带)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 辅助绘图函数
    def plot_with_std(ax, mean_data, std_data, label, epochs):
        x = range(1, epochs + 1)
        line, = ax.plot(x, mean_data, label=label, linewidth=2)
        ax.fill_between(x, mean_data - std_data, mean_data + std_data, alpha=0.2, color=line.get_color())

    # 获取epochs数
    first_result = next(iter(results.values()))
    epochs = len(first_result['mean_train_losses'])

    # 绘制训练损失
    for opt_name, result in results.items():
        plot_with_std(ax1, result['mean_train_losses'], result['std_train_losses'], opt_name, epochs)
    ax1.set_title('训练损失 (Mean ± Std)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制测试准确率
    for opt_name, result in results.items():
        plot_with_std(ax2, result['mean_test_accuracies'], result['std_test_accuracies'], opt_name, epochs)
    ax2.set_title('测试准确率 (Mean ± Std)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 绘制训练准确率
    for opt_name, result in results.items():
        plot_with_std(ax3, result['mean_train_accuracies'], result['std_train_accuracies'], opt_name, epochs)
    ax3.set_title('训练准确率 (Mean ± Std)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 绘制最终性能比较
    optimizers = list(results.keys())
    final_accuracies = [results[opt]['final_test_accuracy'] for opt in optimizers]
    final_std = [results[opt]['final_test_accuracy_std'] for opt in optimizers]
    training_times = [results[opt]['mean_training_time'] for opt in optimizers]
    
    x = np.arange(len(optimizers))
    width = 0.35
    
    # 绘制带误差棒的柱状图
    ax4.bar(x - width/2, final_accuracies, width, yerr=final_std, capsize=5, label='最终测试准确率', alpha=0.8)
    ax4.bar(x + width/2, training_times, width, label='平均训练时间(秒)', alpha=0.8)
    ax4.set_title('最终性能和训练时间比较')
    ax4.set_xlabel('优化器')
    ax4.set_ylabel('值')
    ax4.set_xticks(x)
    ax4.set_xticklabels(optimizers)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion Matrix'):
    """绘制混淆矩阵（仅依赖 matplotlib，不使用 seaborn）"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar(im)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha='center',
                va='center',
                color='white' if cm[i, j] > thresh else 'black',
            )

    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.show()