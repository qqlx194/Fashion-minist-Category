import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from config import TENSORBOARD_CONFIG, CLASS_NAMES

class TensorBoardLogger:
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = TENSORBOARD_CONFIG['log_dir']
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=TENSORBOARD_CONFIG['flush_secs']
        )
        self.step = 0
        
    def log_scalar(self, tag, value, step=None):
        """记录标量值"""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
        
    def log_metrics(self, metrics_dict, step=None):
        """记录多个指标"""
        if step is None:
            step = self.step
        for tag, value in metrics_dict.items():
            self.writer.add_scalar(tag, value, step)
            
    def log_model_graph(self, model, input_tensor):
        """记录模型计算图"""
        self.writer.add_graph(model, input_tensor)
        
    def log_histograms(self, model, step=None):
        """记录模型参数的直方图"""
        if step is None:
            step = self.step
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'parameters/{name}', param, step)
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)
                
    def log_images(self, tag, images, step=None, nrow=8):
        """记录图像批次"""
        if step is None:
            step = self.step
        img_grid = make_grid(images, nrow=nrow, normalize=True)
        self.writer.add_image(tag, img_grid, step)
        
    def log_learning_rates(self, optimizer, step=None):
        """记录学习率"""
        if step is None:
            step = self.step
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'learning_rate/group_{i}', lr, step)
            
    def log_confusion_matrix(self, cm, class_names=None, step=None):
        """记录混淆矩阵"""
        if step is None:
            step = self.step
        if class_names is None:
            class_names = CLASS_NAMES
            
        fig = self._plot_confusion_matrix(cm, class_names)
        self.writer.add_figure('confusion_matrix', fig, step)
        plt.close(fig)
        
    def _plot_confusion_matrix(self, cm, class_names):
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # 设置刻度标签
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')

        # 旋转刻度标签并设置对齐方式
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # 在每个单元格中显示数值
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return fig
        
    def log_optimizer_comparison(self, results, step=None):
        """记录优化器比较结果"""
        if step is None:
            step = self.step
            
        # 记录最终准确率比较
        optimizers = list(results.keys())
        final_accuracies = [results[opt]['final_test_accuracy'] for opt in optimizers]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(optimizers, final_accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet'])
        ax.set_title('Optimizer Comparison - Final Test Accuracy')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Optimizer')
        
        # 在柱状图上添加数值标签
        for bar, accuracy in zip(bars, final_accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{accuracy:.2f}%', ha='center', va='bottom')
        
        self.writer.add_figure('optimizer_comparison/final_accuracy', fig, step)
        plt.close(fig)
        
    def log_training_curves(self, results, step=None):
        """记录训练曲线比较"""
        if step is None:
            step = self.step
            
        # 训练损失曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for opt_name, result in results.items():
            ax1.plot(result['train_losses'], label=opt_name, linewidth=2)
            ax2.plot(result['test_accuracies'], label=opt_name, linewidth=2)
            
        ax1.set_title('Training Loss Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Test Accuracy Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self.writer.add_figure('optimizer_comparison/training_curves', fig, step)
        plt.close(fig)
        
    def increment_step(self):
        """增加步数计数器"""
        self.step += 1
        
    def close(self):
        """关闭写入器"""
        self.writer.close()