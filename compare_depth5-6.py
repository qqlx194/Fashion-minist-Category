import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from config import device, TRAIN_CONFIG, OPTIMIZERS_CONFIG
from data.data_loader import get_data_loaders
from training.trainer import ModelTrainer


def set_seed(seed: int = 42):
	torch.manual_seed(seed)
	if device.type == "cuda":
		torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 1. 定义 5 / 6 层卷积的 CNN 模型
# -----------------------------------------------------------------------------


class CNN5Layers(nn.Module):
	"""5 层卷积 CNN

	结构设计：
	- Block1: conv1, conv2, pool  (28 -> 14)
	- Block2: conv3, conv4, pool  (14 -> 7)
	- Block3: conv5               (7 -> 7)
	- FC: 64 * 7 * 7 -> 128 -> 10
	"""

	def __init__(self):
		super(CNN5Layers, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.conv5 = nn.Conv2d(64, 64, 3, padding=1)

		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 7 * 7, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		# Block 1
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool(x)  # 28 -> 14

		# Block 2
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool(x)  # 14 -> 7

		# Block 3
		x = F.relu(self.conv5(x))  # 7 -> 7

		x = x.view(-1, 64 * 7 * 7)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class CNN6Layers(nn.Module):
	"""6 层卷积 CNN

	结构设计：
	- Block1: conv1, conv2, pool  (28 -> 14)
	- Block2: conv3, conv4, pool  (14 -> 7)
	- Block3: conv5, conv6        (7 -> 7)
	- FC: 64 * 7 * 7 -> 128 -> 10
	"""

	def __init__(self):
		super(CNN6Layers, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
		self.conv6 = nn.Conv2d(64, 64, 3, padding=1)

		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 7 * 7, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		# Block 1
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.pool(x)  # 28 -> 14

		# Block 2
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.pool(x)  # 14 -> 7

		# Block 3
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))  # 7 -> 7

		x = x.view(-1, 64 * 7 * 7)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


# -----------------------------------------------------------------------------
# 2. 实验运行逻辑
# -----------------------------------------------------------------------------


def run_experiment(model_class, tag, epochs, optimizer_name, optimizer_config):
	print(f"\n===== 实验：{tag} - {optimizer_name}（epochs = {epochs}）=====")

	# 统一使用默认预处理
	train_loader, test_loader = get_data_loaders()

	model = model_class().to(device)
	criterion = nn.CrossEntropyLoss()

	# 根据配置创建优化器（支持所有在 OPTIMIZERS_CONFIG 中定义的优化器）
	params = optimizer_config["params"]
	if optimizer_name in ["SGD", "SGD_no_momentum"]:
		optimizer = optim.SGD(model.parameters(), **params)
	elif optimizer_name == "Adam":
		optimizer = optim.Adam(model.parameters(), **params)
	elif optimizer_name == "RMSprop":
		optimizer = optim.RMSprop(model.parameters(), **params)
	elif optimizer_name == "Adagrad":
		optimizer = optim.Adagrad(model.parameters(), **params)
	elif optimizer_name == "AdamW":
		optimizer = optim.AdamW(model.parameters(), **params)
	else:
		raise ValueError(f"Unsupported optimizer: {optimizer_name}")

	# 使用不同的 log_dir 来区分实验
	log_dir = f"runs/depth_experiment/FashionCNN_5_6_Layers/{tag}_{optimizer_name}"

	trainer = ModelTrainer(
		model,
		device,
		optimizer_name=f"{tag}_{optimizer_name}",
		use_tensorboard=True,
		log_dir=log_dir,
	)
	result = trainer.train(train_loader, test_loader, optimizer, criterion, epochs)

	print(
		f"[{tag} - {optimizer_name}] 最终测试准确率: {result['final_test_accuracy']:.2f}%"
	)
	print(f"[{tag} - {optimizer_name}] 训练耗时: {result['training_time']:.2f} 秒")

	return result


def main():
	set_seed(TRAIN_CONFIG["random_seed"])
	# 可以使用 TRAIN_CONFIG["epochs"]，这里保持与其他深度实验一致
	epochs = 10

	# 定义要对比的模型（5 层 vs 6 层 FashionCNN 风格）
	models_config = {
		"FashionCNN_5_Layers": CNN5Layers,
		"FashionCNN_6_Layers": CNN6Layers,
	}

	# 使用 config.py 中的所有优化器配置
	optimizers_to_compare = OPTIMIZERS_CONFIG

	results = {}

	for model_tag, model_cls in models_config.items():
		results[model_tag] = {}
		for opt_name, opt_config in optimizers_to_compare.items():
			res = run_experiment(model_cls, model_tag, epochs, opt_name, opt_config)
			results[model_tag][opt_name] = res

	# 打印对比表
	print("\n" + "=" * 80)
	print(f"{'模型深度':<20} | {'优化器':<12} | {'测试准确率':<15} | {'训练时间':<10}")
	print("-" * 80)
	for model_tag in models_config.keys():
		for opt_name in optimizers_to_compare.keys():
			r = results[model_tag][opt_name]
			print(
				f"{model_tag:<20} | {opt_name:<12} | {r['final_test_accuracy']:>14.2f}% | {r['training_time']:>8.2f}s"
			)

	# 画分组柱状图
	plot_depth_optimizer_comparison(
		results, models_config.keys(), optimizers_to_compare.keys()
	)


def plot_depth_optimizer_comparison(results, model_tags, opt_names):
	model_tags = list(model_tags)
	opt_names = list(opt_names)

	n_models = len(model_tags)
	n_opts = len(opt_names)

	# 设置柱状图宽度
	bar_width = 0.8 / max(n_opts, 1)
	index = np.arange(n_models)

	plt.figure(figsize=(10, 6))

	for i, opt_name in enumerate(opt_names):
		accuracies = [
			results[tag][opt_name]["final_test_accuracy"] for tag in model_tags
		]
		plt.bar(index + i * bar_width, accuracies, bar_width, label=opt_name)

		# 在柱子上显示数值
		for j, v in enumerate(accuracies):
			plt.text(
				index[j] + i * bar_width,
				v + 1,
				f"{v:.1f}%",
				ha="center",
				fontsize=9,
			)

	plt.xlabel("Network Depth (5 vs 6 Conv Layers)")
	plt.ylabel("Test Accuracy (%)")
	plt.title("Effect of Network Depth (5/6 Conv Layers) & Optimizer on Accuracy")
	plt.xticks(index + bar_width * (n_opts - 1) / 2, model_tags)
	plt.legend()
	plt.ylim(0, 100)
	plt.grid(axis="y", linestyle="--", alpha=0.7)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()

