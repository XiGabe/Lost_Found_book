"""
评估脚本 - 在测试集上评估训练好的模型

使用方法:
    python evaluate.py --checkpoint logic/outputs/run_xxx/best_comparator.pth --data logic/lcc_training_data_val.csv
"""

import os
import argparse
import json

import torch
from torch.utils.data import DataLoader

from modules.logic.tokenizer import CharTokenizer
from modules.logic.dataset import LCCPairDataset, collate_fn
from modules.logic.comparator import SiameseBiLSTM
from modules.logic.utils import (
    save_metrics,
    evaluate_model,
    plot_confusion_matrix,
    get_device
)


def parse_args():
    parser = argparse.ArgumentParser(description='评估 LC 索书号排序模型')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data', type=str, required=True,
                        help='测试数据 CSV 路径')
    parser.add_argument('--max-seq-len', type=int, default=64,
                        help='最大序列长度')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='结果输出目录（默认与 checkpoint 同目录）')

    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    加载模型

    Returns:
        (model, tokenizer, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 这里我们需要从 checkpoint 中读取模型配置
    # 为了简化，我们使用默认配置和标准 tokenizer
    tokenizer = CharTokenizer(vocab=CharTokenizer.get_default_vocab())

    # 创建模型
    model = SiameseBiLSTM(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=128,   # 默认值，实际应该从配置读取
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        num_classes=3
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, tokenizer, checkpoint


def main():
    args = parse_args()

    # 设置设备
    device = get_device()

    # 加载模型
    print(f'加载模型: {args.checkpoint}')
    model, tokenizer, checkpoint = load_model(args.checkpoint, device)

    epoch = checkpoint.get('epoch', 'unknown')
    print(f'模型来自 epoch: {epoch}')

    # 加载测试数据
    print(f'加载测试数据: {args.data}')
    dataset = LCCPairDataset(args.data, tokenizer, max_seq_len=args.max_seq_len)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f'测试集大小: {len(dataset)} 样本')
    print(f'标签分布: {dataset.get_label_distribution()}')

    # 评估
    print('\\n开始评估...')
    accuracy, metrics, cm = evaluate_model(model, dataloader, device, num_classes=3)

    # 打印结果
    print('\\n' + '=' * 50)
    print('评估结果:')
    print('=' * 50)
    print(f'总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print()

    # 打印每个类别的指标
    class_names = ['In_Order', 'Duplicate', 'Out_of_Order']
    print(f'{"类别":<15} {"Precision":<12} {"Recall":<12} {"F1":<12} {"Support":<10}')
    print('-' * 60)

    for i, name in enumerate(class_names):
        p = metrics[f'{name}_precision']
        r = metrics[f'{name}_recall']
        f = metrics[f'{name}_f1']
        s = metrics[f'{name}_support']
        print(f'{name:<15} {p:<12.4f} {r:<12.4f} {f:<12.4f} {s:<10}')

    print()
    print(f'宏平均 Precision: {metrics["macro_precision"]:.4f}')
    print(f'宏平均 Recall: {metrics["macro_recall"]:.4f}')
    print(f'宏平均 F1: {metrics["macro_f1"]:.4f}')

    # 混淆矩阵
    print('\\n混淆矩阵:')
    print(cm)

    # 保存结果
    if args.output_dir is None:
        output_dir = os.path.dirname(args.checkpoint)
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # 保存指标
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    save_metrics(metrics, metrics_path)

    # 保存混淆矩阵
    cm_path = os.path.join(output_dir, 'evaluation_confusion_matrix.png')
    plot_confusion_matrix(cm, cm_path)

    # 保存详细报告
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'模型评估报告\\n')
        f.write(f'=' * 50 + '\\n\\n')
        f.write(f'模型: {args.checkpoint}\\n')
        f.write(f'测试数据: {args.data}\\n')
        f.write(f'测试时间: {checkpoint.get("epoch", "unknown")} epoch\\n\\n')
        f.write(f'总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\\n\\n')

        f.write(f'详细指标:\\n')
        f.write(f'{"类别":<15} {"Precision":<12} {"Recall":<12} {"F1":<12} {"Support":<10}\\n')
        f.write('-' * 60 + '\\n')

        for i, name in enumerate(class_names):
            p = metrics[f'{name}_precision']
            r = metrics[f'{name}_recall']
            f = metrics[f'{name}_f1']
            s = metrics[f'{name}_support']
            f.write(f'{name:<15} {p:<12.4f} {r:<12.4f} {f:<12.4f} {s:<10}\\n')

        f.write('\\n混淆矩阵:\\n')
        f.write(str(cm))

    print(f'\\n结果已保存到: {output_dir}')


if __name__ == '__main__':
    main()
