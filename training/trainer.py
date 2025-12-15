import torch
import torch.nn as nn
import time
from utils.metrics import calculate_accuracy
from utils.tensorboard_logger import TensorBoardLogger

class ModelTrainer:
    def __init__(self, model, device, optimizer_name, use_tensorboard=True, log_dir=None):
        self.model = model
        self.device = device
        self.optimizer_name = optimizer_name
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
        # TensorBoard日志记录器
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.logger = TensorBoardLogger(log_dir=log_dir)
        else:
            self.logger = None
        
    def train_epoch(self, train_loader, optimizer, criterion, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 记录批次级别的指标（每100个批次记录一次）
            if self.logger and batch_idx % 100 == 0:
                batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
                self.logger.log_scalar('Loss/batch', loss.item(), self.logger.step)
                self.logger.log_scalar('Accuracy/batch', batch_accuracy, self.logger.step)
                self.logger.increment_step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def evaluate(self, test_loader):
        """在测试集上评估模型"""
        return calculate_accuracy(self.model, test_loader, self.device)
    
    def log_epoch_metrics(self, train_loss, train_accuracy, test_accuracy, epoch):
        """记录epoch级别的指标"""
        if self.logger:
            metrics = {
                'Loss/train': train_loss,
                'Accuracy/train': train_accuracy,
                'Accuracy/test': test_accuracy
            }
            self.logger.log_metrics(metrics, epoch)
            
            # 记录学习率
            # 注意：这里需要在train方法中传递optimizer
    
    def train(self, train_loader, test_loader, optimizer, criterion, epochs, scheduler=None):
        """完整的训练过程"""
        print(f"开始训练 {self.optimizer_name}，共{epochs}个epoch...")
        start_time = time.time()
        
        # 记录模型图（仅第一个批次）
        if self.logger and len(self.train_losses) == 0:
            sample_images, _ = next(iter(train_loader))
            sample_images = sample_images.to(self.device)
            self.logger.log_model_graph(self.model, sample_images)
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_loss, train_accuracy = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # 在测试集上评估
            test_accuracy = self.evaluate(test_loader)
            
            # 记录结果
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.test_accuracies.append(test_accuracy)
            
            # 记录epoch指标
            self.log_epoch_metrics(train_loss, train_accuracy, test_accuracy, epoch)
            
            # 记录学习率
            if self.logger:
                self.logger.log_learning_rates(optimizer, epoch)
            
            # 更新学习率
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}: Learning rate adjusted to {current_lr:.6f}")
            
            # 记录模型参数直方图（每5个epoch记录一次）
            if self.logger and epoch % 5 == 0:
                self.logger.log_histograms(self.model, epoch)
            
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, '
                  f'Test Acc: {test_accuracy:.2f}%')
        
        training_time = time.time() - start_time
        print(f"{self.optimizer_name} 训练完成，用时: {training_time:.2f}秒")
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'training_time': training_time,
            'final_test_accuracy': self.test_accuracies[-1]
        }