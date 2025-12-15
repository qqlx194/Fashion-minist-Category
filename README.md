# Fashion-MNIST 深度学习对比实验

这是一个基于 PyTorch 的 Fashion-MNIST 图像分类实验项目，用于系统对比：

- 不同优化器（SGD、SGD 无动量、Adam、RMSprop、Adagrad、AdamW）；
- 不同数据预处理 / 增强策略；
- 不同网络深度（3/4/5/6 层 CNN）；
- 不同 batch size；
- 以及 ResNet18 与自建 CNN 的表现差异。

整个项目通过 TensorBoard 记录训练曲线和实验日志，配合多个脚本和 Notebook 进行可视化与分析。

## 项目结构概览

```text
Fashion-minist/
├── config.py                     # 全局配置：设备、batch size、epochs、优化器超参数
├── main.ipynb                    # 主 Notebook：优化器对比入口 + 汇总运行其它实验脚本
├── compare_preprocessing.py      # 预处理 × 优化器 对比（基于 SimpleCNN）
├── compare_batch_size.py         # Batch Size 对 SGD / Adam 的影响（基于 SimpleCNN）
├── compare_depth3-4.py           # 3 层 vs 4 层 CNN 深度对比（多优化器）
├── compare_depth5-6.py           # 5 层 vs 6 层 CNN 深度对比（多优化器）
├── optimizer_5layer_non_sgd_normalize05.py # 5 层 CNN + 多优化器对比（统一预处理）
├── sgd_momentum_lr_decay_analysis.py       # 5 层 CNN + SGD(动量) + 学习率衰减分析
├── train_resnet.py               # ResNet18 基线与改进实验
├── data/
│   ├── data_loader.py            # 默认数据加载与预处理（带几何增强的训练集）
│   └── FashionMNIST/             # 原始 Fashion-MNIST 数据（自动下载）
├── models/
│   ├── cnn_model.py              # SimpleCNN / FashionCNN 等基础 CNN 模型
│   └── resnet.py                 # ResNet18 模型定义
├── training/
│   ├── trainer.py                # 通用训练器（单模型训练 + 指标统计）
│   └── optimizer_comparison.py   # 多优化器对比与学习率搜索逻辑
├── utils/
│   ├── metrics.py                # 精度等评价指标
│   ├── tensorboard_logger.py     # TensorBoard 日志封装
│   └── visualization.py          # 曲线 / 混淆矩阵等可视化
└── runs/                         # TensorBoard 日志输出目录
        ├── optimizer_experiment/     # 优化器对比
        ├── batch_size_experiment/    # batch size 实验
        ├── depth_experiment/         # 网络深度实验
        └── fashion_mnist_experiment/ # 预处理与其它实验
```

## 数据与预处理

默认的数据加载由 `data/data_loader.py` 完成：

- 训练集预处理（默认 `get_data_loaders()` 场景）：
    - `RandomHorizontalFlip()`
    - `RandomCrop(28, padding=4)`
    - `RandomRotation(10)`
    - `ToTensor()`
    - `Normalize((0.5,), (0.5,))`
- 测试集预处理：
    - `ToTensor()`
    - `Normalize((0.5,), (0.5,))`

部分实验脚本会显式传入自定义 `transform`，例如：

- `compare_preprocessing.py`：显式定义多种预处理方案（纯 ToTensor、Normalize、随机裁剪 + 翻转、RandomAffine、RandomErasing 等），用于系统对比；
- `compare_batch_size.py`：使用 `ToTensor()` + 标准 Normalize（0.1307, 0.3081）测试 batch size 对 SGD/Adam 的影响；
- `optimizer_5layer_non_sgd_normalize05.py`：统一使用 `ToTensor() + Normalize((0.5,), (0.5,))`。

## 模型说明

- **SimpleCNN**（models/cnn_model.py）
    - 2 个卷积层 + 2 次池化，特征图尺寸 `32×7×7`，FC: `32×7×7 → 64 → 10`；
    - 结构轻量，训练快，适合作为对比实验的基础模型。

- **FashionCNN**（models/cnn_model.py）
    - 2 个卷积层（通道 32/64）+ 2 次池化，特征图尺寸 `64×7×7`；
    - FC: `3136 → 128 → 10`，中间带 Dropout(0.25/0.5)，容量更大、正则更强，适合作为主结果模型。

- **多层 CNN（3/4/5/6 层）**
    - 在 `compare_depth3-4.py` 和 `compare_depth5-6.py` 中定义，主要通过堆叠更多卷积层（通道保持在 32/64）
        来研究“网络深度”对性能和训练时间的影响。

- **ResNet18**（models/resnet.py）
    - 标准 ResNet18 适配到 1 通道输入 / 10 类输出，用于与自建 CNN 的性能对比。

## 实验脚本与内容

### 1. 优化器对比（main.ipynb + training/optimizer_comparison.py）

- 模型：默认使用 `FashionCNN`；
- 优化器：`SGD`、`SGD_no_momentum`、`Adam`、`RMSprop`、`Adagrad`、`AdamW`；
- 机制：
    - 多随机种子运行，统计平均曲线与方差；
    - 可选学习率网格搜索（`tune_lr=True`）为每个优化器寻找较优学习率；
    - 通过 TensorBoard 记录 train/test accuracy 与 loss 曲线。

### 2. 预处理 × 优化器 对比（compare_preprocessing.py）

- 模型：`SimpleCNN`；
- 预处理方案：
    - `NoNormalize`：`ToTensor()`；
    - `Normalize_0.5`：`ToTensor + Normalize((0.5,), (0.5,))`；
    - `CropFlip_Norm`：随机裁剪 + 水平翻转 + Normalize；
    - `Affine_Norm`：RandomAffine（旋转 + 平移）+ Normalize；
    - `Erase_Norm`：Normalize + RandomErasing；
- 对每种预处理，遍历所有优化器，比较最终测试准确率与训练时间。

### 3. Batch Size 实验（compare_batch_size.py）

- 模型：`SimpleCNN`；
- 优化器：`SGD` 与 `Adam`；
- Batch sizes：`32, 128, 1024`；
- 指标：
    - 最终测试准确率；
    - 收敛速度：达到最终准确率 90% 所需 epoch 数；
    - 训练时间；
- 生成 `batch_size_comparison.png` 可视化准确率与收敛速度随 batch size 的变化。

### 4. 网络深度对比（compare_depth3-4.py, compare_depth5-6.py）

- 模型组合：
    - 3 层 vs 4 层 CNN；
    - 5 层 vs 6 层 CNN；
- 优化器：使用 `OPTIMIZERS_CONFIG` 中所有优化器；
- 比较不同深度 + 优化器组合下的测试准确率与训练时间，并绘制分组柱状图。

### 5. 5 层 CNN 多优化器实验（optimizer_5layer_non_sgd_normalize05.py）

- 模型：`CNN5Layers`；
- 预处理：`ToTensor + Normalize((0.5,), (0.5,))`（统一）；
- 优化器：除 `SGD` 外的所有配置（含 `SGD_no_momentum`）；
- 输出每个优化器的最终测试准确率、训练时间，并绘制柱状图对比。

### 6. SGD(动量) + 学习率衰减分析（sgd_momentum_lr_decay_analysis.py）

- 模型：`CNN5Layers`；
- 优化器：`SGD`（带动量）；
- 学习率调度：`StepLR(step_size=10, gamma=0.1)`；
- 重点观察：
    - 学习率分段衰减对收敛曲线的影响；
    - 最终测试准确率与 per-class 表现（结合混淆矩阵可视化）。

### 7. ResNet18 实验（train_resnet.py）

- 模型：`ResNet18`；
- 对比：
    - 基线配置 vs 改进配置（如更强的数据增强、正则化、学习率策略等）；
- 日志与结果记录在 `runs/resnet18_*` 及对应报告文件中。

## 运行方式

### 1）直接用 Notebook

推荐打开 `main.ipynb`，顺序运行：

1. 主函数 `main()`：完成一次优化器对比主实验；
2. 依次运行后续单元：
     - `%run compare_preprocessing.py`
     - `%run compare_depth3-4.py`
     - `%run compare_batch_size.py`
     - `%run compare_depth5-6.py`
     - `%run optimizer_5layer_non_sgd_normalize05.py`
     - `%run sgd_momentum_lr_decay_analysis.py`

Notebook 末尾还包含：

- GPU / CPU 环境检查代码；
- 从测试集中各取一张样本图像，展示 10 个类别的示例图。

### 2）命令行运行单个脚本

在项目根目录下：

```bash
# 预处理 × 优化器 对比（SimpleCNN）
python compare_preprocessing.py

# Batch Size 对比（SimpleCNN + SGD/Adam）
python compare_batch_size.py

# 深度对比（3/4 层）
python compare_depth3-4.py

# 深度对比（5/6 层）
python compare_depth5-6.py

# 5 层 CNN 多优化器
python optimizer_5layer_non_sgd_normalize05.py

# SGD(动量) + 学习率衰减分析
python sgd_momentum_lr_decay_analysis.py

# ResNet18 实验
python train_resnet.py
```

## TensorBoard 可视化

所有实验会将标量指标和部分可视化结果记录到 `runs/` 目录。使用 TensorBoard 查看：

```bash
tensorboard --logdir=runs
```

在浏览器中打开提示的地址，即可查看：

- 不同优化器的 train/test accuracy / loss 曲线；
- 不同 batch size、不同网络深度的对比曲线；
- 某些实验的混淆矩阵与样本可视化（若启用）。

---

本 README 已根据当前代码结构和实验脚本更新，便于课程报告撰写和后续扩展实验使用。
