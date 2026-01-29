"""
工具函数 for 训练和评估
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(log_dir: str, log_name: str = "training.log") -> logging.Logger:
    """
    设置日志

    Args:
        log_dir: 日志目录
        log_name: 日志文件名

    Returns:
        logger 实例
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    path: str,
):
    """
    保存模型检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        loss: 当前 loss
        accuracy: 当前准确率
        path: 保存路径
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, path)
    print(f"模型已保存: {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    加载模型检查点

    Args:
        path: 检查点路径
        model: 模型实例
        optimizer: 优化器实例（可选）

    Returns:
        (epoch, loss, accuracy)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']


def save_metrics(metrics: Dict, path: str):
    """保存评估指标到 JSON"""
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"指标已保存: {path}")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 3
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """
    评估模型

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_classes: 类别数量

    Returns:
        (accuracy, metrics_per_class, confusion_matrix)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a, input_ids_b, labels = batch
            input_ids_a = input_ids_a.to(device)
            input_ids_b = input_ids_b.to(device)

            outputs = model(input_ids_a, input_ids_b)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 总体准确率
    accuracy = accuracy_score(all_labels, all_preds)

    # 每个类别的 precision, recall, f1
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 构建指标字典
    metrics_per_class = {}
    class_names = ['In_Order', 'Duplicate', 'Out_of_Order']

    for i in range(num_classes):
        metrics_per_class[f'{class_names[i]}_precision'] = float(precision[i])
        metrics_per_class[f'{class_names[i]}_recall'] = float(recall[i])
        metrics_per_class[f'{class_names[i]}_f1'] = float(f1[i])
        metrics_per_class[f'{class_names[i]}_support'] = int(support[i])

    # 宏平均
    metrics_per_class['macro_precision'] = float(np.mean(precision))
    metrics_per_class['macro_recall'] = float(np.mean(recall))
    metrics_per_class['macro_f1'] = float(np.mean(f1))

    metrics_per_class['accuracy'] = float(accuracy)

    return accuracy, metrics_per_class, cm


def plot_confusion_matrix(cm: np.ndarray, save_path: str, class_names: List[str] = None):
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        save_path: 保存路径
        class_names: 类别名称
    """
    if class_names is None:
        class_names = ['In_Order', 'Duplicate', 'Out_of_Order']

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def plot_training_history(history: Dict, save_path: str):
    """
    绘制训练历史

    Args:
        history: 训练历史 {'train_loss': [], 'val_loss': [], 'val_acc': []}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['val_acc'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练历史已保存: {save_path}")


class AverageMeter:
    """计算和存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device() -> torch.device:
    """获取训练设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device


def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: 容忍的 epoch 数
            min_delta: 最小改善幅度
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        """
        Args:
            val_score: 验证集分数（越大越好）

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop


if __name__ == "__main__":
    # 测试
    print("工具函数测试:")
    print(f"设备: {get_device()}")
    set_seed(42)

    # 测试 AverageMeter
    meter = AverageMeter()
    for i in range(1, 6):
        meter.update(i)
    print(f"AverageMeter: avg={meter.avg}, sum={meter.sum}")

    # 测试 EarlyStopping
    early_stop = EarlyStopping(patience=3)
    scores = [0.8, 0.82, 0.81, 0.80, 0.79]
    for score in scores:
        result = early_stop(score)
        print(f"Score: {score:.2f}, Early Stop: {result}")
