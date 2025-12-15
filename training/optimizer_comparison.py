import torch
import torch.optim as optim
import numpy as np
from models.cnn_model import FashionCNN
from training.trainer import ModelTrainer
from config import OPTIMIZERS_CONFIG, TRAIN_CONFIG
from utils.tensorboard_logger import TensorBoardLogger

class OptimizerComparator:
    def __init__(self, device, model_class=FashionCNN, use_tensorboard=True):
        self.device = device
        self.model_class = model_class
        self.results = {}
        self.use_tensorboard = use_tensorboard
        
        # 创建TensorBoard记录器
        if use_tensorboard:
            self.logger = TensorBoardLogger()
        else:
            self.logger = None
        
    def create_optimizer(self, optimizer_config, model_parameters):
        """根据配置创建优化器"""
        optimizer_name = optimizer_config['optimizer']
        params = optimizer_config['params']
        
        if optimizer_name == 'SGD':
            return optim.SGD(model_parameters, **params)
        elif optimizer_name == 'Adam':
            return optim.Adam(model_parameters, **params)
        elif optimizer_name == 'RMSprop':
            return optim.RMSprop(model_parameters, **params)
        elif optimizer_name == 'Adagrad':
            return optim.Adagrad(model_parameters, **params)
        elif optimizer_name == 'AdamW':
            return optim.AdamW(model_parameters, **params)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    def compare_optimizers(self, train_loader, test_loader, optimizers_config=None, epochs=None, seeds=[42, 100, 2023], tune_lr=False, lr_tune_epochs=5):
        """比较不同优化器的性能 (支持多次运行取平均，支持超参数搜索)

        Args:
            train_loader, test_loader: 数据加载器
            optimizers_config: 优化器配置字典
            epochs: 正式训练的 epoch 数
            seeds: 不同随机种子列表
            tune_lr: 是否先做学习率网格搜索
            lr_tune_epochs: 每个候选学习率用于调参的 epoch 数（默认 5，比 3 更稳一点）
        """
        if optimizers_config is None:
            optimizers_config = OPTIMIZERS_CONFIG
            
        if epochs is None:
            epochs = TRAIN_CONFIG['epochs']
            
        # 默认的学习率搜索空间
        lr_search_space = {
            'SGD': [0.1, 0.01, 0.001],
            'SGD_no_momentum': [0.1, 0.01, 0.001],
            'Adam': [0.001, 0.0001, 0.00001],
            'RMSprop': [0.01, 0.001, 0.0001],
            'Adagrad': [0.1, 0.01, 0.001],
            'AdamW': [0.01, 0.001, 0.0001]
        }
        
        for opt_name, config in optimizers_config.items():
            print(f"\n{'='*50}")
            
            # 如果开启了调参
            current_config = config.copy()
            current_config['params'] = config['params'].copy() # 深拷贝参数
            
            if tune_lr:
                print(f"正在为 {opt_name} 搜索最佳学习率...")
                best_lr = self._find_best_lr(opt_name, current_config, train_loader, test_loader, lr_search_space, lr_tune_epochs)
                current_config['params']['lr'] = best_lr
                print(f"使用最佳学习率: {best_lr} 进行正式实验")
            
            print(f"训练 {opt_name} 优化器 (共 {len(seeds)} 次实验)...")
            print(f"{'='*50}")
            
            # 存储多次运行的结果
            all_train_losses = []
            all_train_accuracies = []
            all_test_accuracies = []
            all_training_times = []
            
            for seed in seeds:
                print(f"  > 运行 Seed: {seed}")
                # 设置随机种子
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                
                # 为每个优化器创建新的模型实例
                model = self.model_class().to(self.device)
                optimizer = self.create_optimizer(current_config, model.parameters())
                criterion = torch.nn.CrossEntropyLoss()
                
                # 创建训练器并训练
                # 修改：记录每次运行的完整曲线到 TensorBoard
                # 使用子目录结构 runs/optimizer_experiment/{opt_name}/seed_{seed}
                # Trainer 内部现在使用统一的 'Loss/train', 'Accuracy/train' 等标签
                if self.use_tensorboard:
                    log_dir = f"runs/optimizer_experiment/{opt_name}/seed_{seed}"
                    trainer = ModelTrainer(
                        model, 
                        self.device, 
                        optimizer_name=opt_name, 
                        use_tensorboard=True, 
                        log_dir=log_dir
                    )
                else:
                    trainer = ModelTrainer(model, self.device, f"{opt_name}_seed{seed}", use_tensorboard=False)
                
                result = trainer.train(train_loader, test_loader, optimizer, criterion, epochs)
                
                all_train_losses.append(result['train_losses'])
                all_train_accuracies.append(result['train_accuracies'])
                all_test_accuracies.append(result['test_accuracies'])
                all_training_times.append(result['training_time'])
            
            # 计算统计数据
            mean_train_losses = np.mean(all_train_losses, axis=0)
            std_train_losses = np.std(all_train_losses, axis=0)
            
            mean_train_accuracies = np.mean(all_train_accuracies, axis=0)
            std_train_accuracies = np.std(all_train_accuracies, axis=0)
            
            mean_test_accuracies = np.mean(all_test_accuracies, axis=0)
            std_test_accuracies = np.std(all_test_accuracies, axis=0)
            
            mean_training_time = np.mean(all_training_times)
            std_training_time = np.std(all_training_times)
            
            # --- TensorBoard Logging ---
            # 之前我们记录的是平均曲线。现在我们已经记录了所有独立的 Seed 曲线。
            # TensorBoard 可以通过 Group By 功能自动聚合这些曲线。
            # 因此，这里不再需要手动记录平均曲线，以免数据重复或混淆。
            # ---------------------------

            self.results[opt_name] = {
                'mean_train_losses': mean_train_losses,
                'std_train_losses': std_train_losses,
                'mean_train_accuracies': mean_train_accuracies,
                'std_train_accuracies': std_train_accuracies,
                'mean_test_accuracies': mean_test_accuracies,
                'std_test_accuracies': std_test_accuracies,
                'mean_training_time': mean_training_time,
                'std_training_time': std_training_time,
                'final_test_accuracy': mean_test_accuracies[-1],
                'final_test_accuracy_std': std_test_accuracies[-1],
                'best_lr': current_config['params']['lr'] # 记录使用的最佳学习率
            }
            
        return self.results

    def _find_best_lr(self, opt_name, base_config, train_loader, test_loader, search_space, lr_tune_epochs):
        """简单的网格搜索寻找最佳学习率

        Args:
            lr_tune_epochs: 每个候选学习率用于调参的 epoch 数
        """
        best_acc = 0
        best_lr = base_config['params']['lr']
        
        # 获取该优化器的搜索空间，如果没有定义则使用默认值
        lrs_to_try = search_space.get(opt_name, [base_config['params']['lr']])
        
        for lr in lrs_to_try:
            # 临时修改配置
            current_params = base_config['params'].copy()
            current_params['lr'] = lr
            
            # 快速训练几轮 (例如 lr_tune_epochs 个 epoch) 来评估收敛趋势
            # 使用固定的种子进行调参，保证公平
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                
            model = self.model_class().to(self.device)
            optimizer = self.create_optimizer({'optimizer': opt_name, 'params': current_params}, model.parameters())
            criterion = torch.nn.CrossEntropyLoss()
            
            # 这里的 log_dir 设为 None 不记录日志
            trainer = ModelTrainer(model, self.device, f"tuning_{opt_name}_{lr}", use_tensorboard=False)
            
            # 使用 lr_tune_epochs 控制调参阶段的训练轮数
            print(f"    [Tuning] 尝试 LR={lr} ...", end="", flush=True)
            result = trainer.train(train_loader, test_loader, optimizer, criterion, epochs=lr_tune_epochs)
            final_acc = result['final_test_accuracy']
            print(f" Acc={final_acc:.2f}%")
            
            if final_acc > best_acc:
                best_acc = final_acc
                best_lr = lr
                
        return best_lr
    
    def _log_confusion_matrix(self, model, test_loader):
        """计算并记录混淆矩阵"""
        if not self.logger:
            return
            
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 记录混淆矩阵
        self.logger.log_confusion_matrix(cm)
    
    def get_best_optimizer(self):
        """获取性能最好的优化器"""
        if not self.results:
            return None
            
        best_optimizer = max(self.results.items(), key=lambda x: x[1]['final_test_accuracy'])
        return best_optimizer
    
    def print_summary(self):
        """打印优化器比较摘要"""
        print("\n" + "="*60)
        print("优化器性能总结:")
        print("="*60)
        print(f"{'优化器':<10} | {'最终测试准确率':<15} | {'训练时间':<10}")
        print("-" * 60)
        
        for opt_name, result in self.results.items():
            # 使用平均训练时间作为摘要中的训练时间
            print(f"{opt_name:<10} | {result['final_test_accuracy']:>14.2f}% | {result['mean_training_time']:>8.2f}秒")
        
        best_opt = self.get_best_optimizer()
        if best_opt:
            print("-" * 60)
            print(f"最佳优化器: {best_opt[0]} "
                  f"(准确率: {best_opt[1]['final_test_accuracy']:.2f}%)")
    
    def close_logger(self):
        """关闭TensorBoard记录器"""
        if self.logger:
            self.logger.close()