"""
训练脚本 for LC 索书号排序模型

使用方法:
    python train.py --data data/synthetic_pairs/lcc_training_data.csv --epochs 30 --batch-size 64
"""

import os
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from modules.logic.tokenizer import CharTokenizer
from modules.logic.dataset import LCCPairDataset, collate_fn, split_train_val
from modules.logic.comparator import SiameseBiLSTM, count_parameters
from modules.logic.utils import (
    setup_logging,
    save_checkpoint,
    save_metrics,
    evaluate_model,
    plot_training_history,
    AverageMeter,
    get_device,
    set_seed,
    EarlyStopping
)


def parse_args():
    parser = argparse.ArgumentParser(description='训练 LC 索书号排序模型')

    # 数据参数
    parser.add_argument('--data', type=str, default='data/synthetic_pairs/lcc_training_data.csv',
                        help='训练数据 CSV 路径')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--max-seq-len', type=int, default=64,
                        help='最大序列长度')

    # 模型参数
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='字符嵌入维度')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='LSTM 隐藏层维度')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='LSTM 层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 比例')
    parser.add_argument('--num-classes', type=int, default=3,
                        help='分类数量')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 其他参数
    parser.add_argument('--output-dir', type=str, default='logic/outputs',
                        help='输出目录')
    parser.add_argument('--save-every', type=int, default=5,
                        help='每多少 epoch 保存一次检查点')
    parser.add_argument('--early-stop-patience', type=int, default=5,
                        help='早停容忍 epoch 数')

    # 继续训练
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='从检查点继续训练')

    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger
) -> float:
    """训练一个 epoch"""
    model.train()

    loss_meter = AverageMeter()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, (input_ids_a, input_ids_b, labels) in enumerate(pbar):
        input_ids_a = input_ids_a.to(device)
        input_ids_b = input_ids_b.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids_a, input_ids_b)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录
        loss_meter.update(loss.item(), input_ids_a.size(0))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    logger.info(f'Epoch {epoch} - Train Loss: {loss_meter.avg:.4f}')
    return loss_meter.avg


def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger
) -> tuple:
    """验证"""
    model.eval()

    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')

    with torch.no_grad():
        for input_ids_a, input_ids_b, labels in pbar:
            input_ids_a = input_ids_a.to(device)
            input_ids_b = input_ids_b.to(device)
            labels = labels.to(device)

            outputs = model(input_ids_a, input_ids_b)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss_meter.update(loss.item(), input_ids_a.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    # 计算准确率
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()

    logger.info(f'Epoch {epoch} - Val Loss: {loss_meter.avg:.4f}, Val Acc: {accuracy:.4f}')

    return loss_meter.avg, accuracy


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = get_device()

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # 设置日志
    logger = setup_logging(output_dir)
    logger.info(f'训练配置: {args}')
    logger.info(f'输出目录: {output_dir}')

    # 加载数据
    logger.info('加载数据...')

    # 创建 tokenizer
    tokenizer = CharTokenizer(vocab=CharTokenizer.get_default_vocab())
    logger.info(f'Tokenizer: vocab_size={tokenizer.get_vocab_size()}, max_seq_len={args.max_seq_len}')

    # 划分训练集和验证集
    train_csv, val_csv = split_train_val(args.data, val_ratio=args.val_ratio, random_seed=args.seed)

    # 创建 dataset 和 dataloader
    train_dataset = LCCPairDataset(train_csv, tokenizer, max_seq_len=args.max_seq_len)
    val_dataset = LCCPairDataset(val_csv, tokenizer, max_seq_len=args.max_seq_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    logger.info(f'训练集: {len(train_dataset)} 样本')
    logger.info(f'验证集: {len(val_dataset)} 样本')
    logger.info(f'标签分布 (训练集): {train_dataset.get_label_distribution()}')

    # 创建模型
    model = SiameseBiLSTM(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=args.num_classes
    )

    # 打印模型配置
    logger.info(f'模型配置: embedding_dim={args.embedding_dim}, hidden_dim={args.hidden_dim}, num_layers={args.num_layers}')

    model = model.to(device)
    logger.info(f'模型参数量: {count_parameters(model):,}')

    # 如果指定了 checkpoint，加载模型权重
    start_epoch = 1
    if args.checkpoint:
        logger.info(f'加载检查点: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f'从 epoch {start_epoch} 继续训练')

    # 损失函数和优化器
    # 使用类别权重处理不平衡（可选）
    class_weights = train_dataset.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 如果加载了 checkpoint，也加载优化器状态
    if args.checkpoint and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info('已加载优化器状态')

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # 早停
    early_stopping = EarlyStopping(patience=args.early_stop_patience)

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    # 训练循环
    logger.info(f'开始训练... (从 epoch {start_epoch} 到 {args.epochs})')

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, logger)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, logger)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(output_dir, 'best_comparator.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)

        # 定期保存
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)

        # 早停检查
        if early_stopping(val_acc):
            logger.info(f'早停触发，停止训练 (epoch {epoch})')
            break

        epoch_time = time.time() - start_time
        logger.info(f'Epoch {epoch} 耗时: {epoch_time:.2f}s')

    # 训练完成
    logger.info('训练完成!')

    # 保存训练历史
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))

    # 在验证集上评估
    logger.info('在验证集上评估...')
    accuracy, metrics, cm = evaluate_model(model, val_loader, device, num_classes=args.num_classes)

    logger.info(f'验证集准确率: {accuracy:.4f}')
    logger.info(f'详细指标: {metrics}')

    # 保存指标
    save_metrics(metrics, os.path.join(output_dir, 'metrics.json'))

    # 绘制混淆矩阵
    from logic.utils import plot_confusion_matrix
    plot_confusion_matrix(cm, os.path.join(output_dir, 'confusion_matrix.png'))

    logger.info(f'所有结果已保存到: {output_dir}')
    print(f'\\n最佳模型准确率: {best_val_acc:.4f}')
    print(f'最佳模型路径: {os.path.join(output_dir, "best_comparator.pth")}')


if __name__ == '__main__':
    main()
