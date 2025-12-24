"""
大模型训练脚本 - 提升准确率到 85-88%

使用方法:
    python train_large.py --data lcc_training_data.csv --epochs 30 --batch-size 64
"""

import os
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.tokenizer import CharTokenizer
from models.dataset import LCCPairDataset, collate_fn, split_train_val
from models.siamese_lstm import SiameseBiLSTM, count_parameters
from models.utils import (
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
    parser = argparse.ArgumentParser(description='训练大模型 LC 索书号排序')

    parser.add_argument('--data', type=str, default='lcc_training_data.csv',
                        help='训练数据 CSV 路径')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--max-seq-len', type=int, default=64,
                        help='最大序列长度')

    # 大模型参数
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='字符嵌入维度 (默认 256)')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='LSTM 隐藏层维度 (默认 512)')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='LSTM 层数 (默认 3)')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout 比例 (默认 0.4)')
    parser.add_argument('--num-classes', type=int, default=3,
                        help='分类数量')

    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小 (大模型用 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    parser.add_argument('--output-dir', type=str, default='outputs_large',
                        help='输出目录')
    parser.add_argument('--save-every', type=int, default=5,
                        help='每多少 epoch 保存一次检查点')
    parser.add_argument('--early-stop-patience', type=int, default=7,
                        help='早停容忍 epoch 数')

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

        optimizer.zero_grad()
        outputs = model(input_ids_a, input_ids_b)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

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

    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()

    logger.info(f'Epoch {epoch} - Val Loss: {loss_meter.avg:.4f}, Val Acc: {accuracy:.4f}')

    return loss_meter.avg, accuracy


def main():
    args = parse_args()

    set_seed(args.seed)
    device = get_device()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f'大模型训练配置: {args}')
    logger.info(f'输出目录: {output_dir}')

    logger.info('加载数据...')
    tokenizer = CharTokenizer(vocab=CharTokenizer.get_default_vocab())
    logger.info(f'Tokenizer: vocab_size={tokenizer.get_vocab_size()}, max_seq_len={args.max_seq_len}')

    train_csv, val_csv = split_train_val(args.data, val_ratio=args.val_ratio, random_seed=args.seed)

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

    # 创建大模型
    model = SiameseBiLSTM(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=args.num_classes
    )

    model = model.to(device)
    logger.info(f'模型配置: embedding_dim={args.embedding_dim}, hidden_dim={args.hidden_dim}, num_layers={args.num_layers}')
    logger.info(f'模型参数量: {count_parameters(model):,}')

    class_weights = train_dataset.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    early_stopping = EarlyStopping(patience=args.early_stop_patience)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    logger.info('开始训练...')

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, logger)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, logger)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(output_dir, 'best_comparator.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)

        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)

        if early_stopping(val_acc):
            logger.info(f'早停触发，停止训练 (epoch {epoch})')
            break

        epoch_time = time.time() - start_time
        logger.info(f'Epoch {epoch} 耗时: {epoch_time:.2f}s')

    logger.info('训练完成!')

    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))

    logger.info('在验证集上评估...')
    accuracy, metrics, cm = evaluate_model(model, val_loader, device, num_classes=args.num_classes)

    logger.info(f'验证集准确率: {accuracy:.4f}')
    logger.info(f'详细指标: {metrics}')

    save_metrics(metrics, os.path.join(output_dir, 'metrics.json'))

    from models.utils import plot_confusion_matrix
    plot_confusion_matrix(cm, os.path.join(output_dir, 'confusion_matrix.png'))

    logger.info(f'所有结果已保存到: {output_dir}')
    print(f'\n最佳模型准确率: {best_val_acc:.4f}')
    print(f'最佳模型路径: {os.path.join(output_dir, "best_comparator.pth")}')


if __name__ == '__main__':
    main()
