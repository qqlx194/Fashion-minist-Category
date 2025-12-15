#!/bin/bash
set -e

echo "开始安装 Anaconda..."
# 清理旧的安装（如果有）
rm -rf ~/anaconda3

# 安装 Anaconda
bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p $HOME/anaconda3

echo "初始化 Conda..."
# 初始化 conda
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
$HOME/anaconda3/bin/conda init bash

echo "创建环境 fashion-mnist..."
# 创建环境
conda create -n fashion-mnist python=3.10 -y

echo "激活环境并安装依赖..."
# 激活环境
conda activate fashion-mnist

# 使用 pip 安装 PyTorch (CPU版) 和其他依赖
# 使用 pip 安装通常比 conda 更快且不容易出现依赖冲突
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib tensorboard ipykernel

echo "========================================================"
echo "安装完成！"
echo "请执行以下操作以开始使用："
echo "1. 输入 'bash' 进入 bash shell (如果还没在 bash 中)"
echo "2. 运行 'source ~/.bashrc'"
echo "3. 运行 'conda activate fashion-mnist'"
echo "========================================================"
