import torch
import numpy as np
import numpy as np

from data.data_loader import get_class_names
from utils.visualization import plot_confusion_matrix

def analyze_per_class_performance(model, test_loader, device, top_k_worst=3):
    """
    展示分类中哪些类别分类最差，并输出每个类别的分类结果。
    - 计算混淆矩阵
    - 计算每类准确率
    - 找出若干个最差类别
    - 画归一化混淆矩阵
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算混淆矩阵（无需依赖 sklearn）
    num_classes = len(get_class_names())
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
    class_names = np.array(get_class_names())

    # 每类准确率：该类被预测正确的样本数 / 该类总样本数
    per_class_correct = np.diag(cm)
    per_class_total = cm.sum(axis=1)
    per_class_acc = per_class_correct / per_class_total

    print("\n每个类别的分类结果（按准确率从低到高排序）:")
    sorted_idx = np.argsort(per_class_acc)  # 从最差到最好
    for idx in sorted_idx:
        print(
            f"{class_names[idx]:<12} | "
            f"样本数: {per_class_total[idx]:>4d} | "
            f"准确率: {per_class_acc[idx] * 100:6.2f}% "
            f"(正确 {per_class_correct[idx]:>4d})"
        )

    # 显示若干个分类最差的类别
    print("\n分类最差的类别（Top-K）:")
    for i in range(min(top_k_worst, len(class_names))):
        idx = sorted_idx[i]
        print(
            f"#{i+1}: {class_names[idx]} "
            f"- 准确率 {per_class_acc[idx] * 100:.2f}% "
            f"(正确 {per_class_correct[idx]}/{per_class_total[idx]})"
        )

    # 画归一化混淆矩阵
    print("\n绘制归一化混淆矩阵...")
    plot_confusion_matrix(
        cm,
        class_names.tolist(),
        normalize=True,
        title="Fashion-MNIST 每类归一化混淆矩阵"
    )