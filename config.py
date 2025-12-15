import torch

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据配置
DATA_CONFIG = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 2  # Windows下建议设为0，避免多进程死锁
}

# 训练配置
TRAIN_CONFIG = {
    'epochs': 30,
    'random_seed': 42
}

# 优化器配置
OPTIMIZERS_CONFIG = {
    # SGD（带动量）
    'SGD': {
        'optimizer': 'SGD',
        'params': {'lr': 0.1, 'momentum': 0.9}
    },
    # SGD（不带动量，用于对比）
    'SGD_no_momentum': {
        'optimizer': 'SGD',
        'params': {'lr': 0.1, 'momentum': 0.0}
    },
    'Adam': {
        'optimizer': 'Adam',
        'params': {'lr': 0.001}
    },
    'RMSprop': {
        'optimizer': 'RMSprop',
        'params': {'lr': 0.001}
    },
    'Adagrad': {
        'optimizer': 'Adagrad',
        'params': {'lr': 0.01}
    },
    'AdamW': {
        'optimizer': 'AdamW',
        'params': {'lr': 0.001}
    }
}

# Fashion-MNIST 类别名称
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# TensorBoard 配置
TENSORBOARD_CONFIG = {
    'log_dir': 'runs/fashion_mnist_experiment',
    'write_images': True,
    'flush_secs': 10
}