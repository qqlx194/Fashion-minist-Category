import torch
import torch.nn as nn
from torchvision.models import resnet18

class FashionResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(FashionResNet18, self).__init__()
        # 加载 resnet18
        # 如果 torchvision 版本较新，建议使用 weights 参数，这里为了兼容性使用 pretrained 参数（旧版）或不传
        # 为了避免版本问题，这里默认不使用预训练权重，因为我们要修改第一层结构
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = resnet18(weights=weights)
        except ImportError:
            self.model = resnet18(pretrained=pretrained)
        
        # 修改第一层卷积层以适应 Fashion-MNIST (1通道, 28x28)
        # 原版 ResNet18: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 对于 28x28 的图像，7x7 stride 2 会导致信息丢失过快，且尺寸缩减太快
        # 修改为: 3x3 kernel, stride 1, padding 1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 移除第一层的 maxpool
        # 原版 ResNet18: nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 28x28 经过 conv1(stride 1) -> 28x28
        # 如果保留 maxpool(stride 2) -> 14x14
        # 如果不移除，feature map 会变小，但 ResNet18 应该也能工作。
        # 不过对于 CIFAR/Fashion-MNIST，通常移除 maxpool 以获得更好性能。
        self.model.maxpool = nn.Identity()
        
        # 修改全连接层以适应类别数，并增加 Dropout
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
