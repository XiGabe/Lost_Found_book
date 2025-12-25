"""
推理脚本 - 对单个索书号对进行排序判断

使用方法:
    python inference.py --text-a "QA76.5 .C64" --text-b "QA76.6 .A12"
    python inference.py --checkpoint logic/outputs/run_xxx/best_comparator.pth
"""

import argparse
import sys
from pathlib import Path

import torch

from .tokenizer import CharTokenizer
from .comparator import SiameseBiLSTM


# 全局模型和tokenizer（避免重复加载）
_model = None
_tokenizer = None
_device = None


def load_model(checkpoint_path: str, device: torch.device = None):
    """
    加载模型

    Args:
        checkpoint_path: 模型检查点路径
        device: 设备（默认自动检测）

    Returns:
        (model, tokenizer, device)
    """
    global _model, _tokenizer, _device

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if _model is None:
        print(f'加载模型: {checkpoint_path}')

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 创建 tokenizer
        _tokenizer = CharTokenizer(vocab=CharTokenizer.get_default_vocab())

        # 创建模型
        _model = SiameseBiLSTM(
            vocab_size=_tokenizer.get_vocab_size(),
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            num_classes=3
        )

        # 加载权重
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model = _model.to(device)
        _model.eval()

        _device = device

        epoch = checkpoint.get('epoch', 'unknown')
        accuracy = checkpoint.get('accuracy', 'unknown')
        print(f'模型来自 epoch: {epoch}, 验证准确率: {accuracy}')

    return _model, _tokenizer, _device


def compare_lcc(
    text_a: str,
    text_b: str,
    checkpoint_path: str = None,
    model = None,
    tokenizer = None,
    device = None
) -> dict:
    """
    比较两个 LC 索书号的排序

    Args:
        text_a: 第一个索书号（OCR 字符串）
        text_b: 第二个索书号（OCR 字符串）
        checkpoint_path: 模型路径（可选）
        model: 已加载的模型（可选）
        tokenizer: 已加载的 tokenizer（可选）
        device: 设备（可选）

    Returns:
        {
            'label': int,  # 0, 1, 或 2
            'label_name': str,  # 'In_Order', 'Out_of_Order', 或 'Duplicate'
            'confidence': float,  # 置信度 (0-1)
            'probabilities': dict  # 每个类别的概率
        }
    """
    # 加载模型（如果未提供）
    if model is None:
        if checkpoint_path is None:
            raise ValueError('必须提供 checkpoint_path 或 model')
        model, tokenizer, device = load_model(checkpoint_path, device)

    # 编码输入
    ids_a = tokenizer.encode(text_a, add_padding=True)
    ids_b = tokenizer.encode(text_b, add_padding=True)

    # 转为 tensor
    input_ids_a = torch.tensor([ids_a], dtype=torch.long).to(device)
    input_ids_b = torch.tensor([ids_b], dtype=torch.long).to(device)

    # 预测
    with torch.no_grad():
        preds, probs = model.predict(input_ids_a, input_ids_b)

    # 解析结果
    pred_label = preds.item()
    probs_np = probs.cpu().numpy()[0]

    label_names = ['In_Order', 'Out_of_Order', 'Duplicate']

    return {
        'label': int(pred_label),
        'label_name': label_names[pred_label],
        'confidence': float(probs_np[pred_label]),
        'probabilities': {
            'In_Order': float(probs_np[0]),
            'Out_of_Order': float(probs_np[1]),
            'Duplicate': float(probs_np[2])
        }
    }


def batch_compare_lcc(
    pairs: list,
    checkpoint_path: str,
    batch_size: int = 64
) -> list:
    """
    批量比较多个索书号对

    Args:
        pairs: list of (text_a, text_b) tuples
        checkpoint_path: 模型路径
        batch_size: 批次大小

    Returns:
        list of 结果字典
    """
    model, tokenizer, device = load_model(checkpoint_path)

    results = []

    # 分批处理
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]

        # 编码
        batch_ids_a = []
        batch_ids_b = []

        for text_a, text_b in batch_pairs:
            ids_a = tokenizer.encode(text_a, add_padding=True)
            ids_b = tokenizer.encode(text_b, add_padding=True)
            batch_ids_a.append(ids_a)
            batch_ids_b.append(ids_b)

        # 转为 tensor
        input_ids_a = torch.tensor(batch_ids_a, dtype=torch.long).to(device)
        input_ids_b = torch.tensor(batch_ids_b, dtype=torch.long).to(device)

        # 预测
        with torch.no_grad():
            preds, probs = model.predict(input_ids_a, input_ids_b)

        # 解析结果
        for j in range(len(batch_pairs)):
            pred_label = preds[j].item()
            probs_np = probs[j].cpu().numpy()

            results.append({
                'text_a': batch_pairs[j][0],
                'text_b': batch_pairs[j][1],
                'label': int(pred_label),
                'label_name': ['In_Order', 'Out_of_Order', 'Duplicate'][pred_label],
                'confidence': float(probs_np[pred_label]),
                'probabilities': {
                    'In_Order': float(probs_np[0]),
                    'Out_of_Order': float(probs_np[1]),
                    'Duplicate': float(probs_np[2])
                }
            })

    return results


def main():
    parser = argparse.ArgumentParser(description='LC 索书号排序推理')

    parser.add_argument('--text-a', type=str, help='第一个索书号')
    parser.add_argument('--text-b', type=str, help='第二个索书号')
    parser.add_argument('--checkpoint', type=str,
                        default='weights/comparator.pth',
                        help='模型检查点路径')
    parser.add_argument('--batch-file', type=str,
                        help='批量推理文件（CSV 格式）')

    args = parser.parse_args()

    # 检查模型文件
    if not Path(args.checkpoint).exists():
        print(f'错误: 模型文件不存在: {args.checkpoint}')
        print('提示: 请先用 train.py 训练模型')
        sys.exit(1)

    # 单对推理
    if args.text_a and args.text_b:
        result = compare_lcc(args.text_a, args.text_b, args.checkpoint)

        print('\\n' + '=' * 50)
        print('LC 索书号排序判断结果')
        print('=' * 50)
        print(f'A: {args.text_a}')
        print(f'B: {args.text_b}')
        print(f'\\n预测结果: {result["label_name"]}')
        print(f'置信度: {result["confidence"]:.4f}')
        print(f'\\n各类别概率:')
        for label, prob in result['probabilities'].items():
            print(f'  {label}: {prob:.4f}')

        # 解释结果
        print('\\n解释:')
        if result['label'] == 0:
            print('  → 顺序正确 (A < B)')
        elif result['label'] == 1:
            print('  ⚠ 顺序错误 (A > B) - 需要调整位置！')
        else:
            print('  → 相同/重复 (A ≈ B)')

    # 批量推理
    elif args.batch_file:
        import pandas as pd

        df = pd.read_csv(args.batch_file)
        pairs = list(zip(df['text_a'].tolist(), df['text_b'].tolist()))

        print(f'批量推理: {len(pairs)} 对')
        results = batch_compare_lcc(pairs, args.checkpoint)

        # 保存结果
        output_file = args.batch_file.replace('.csv', '_results.csv')
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f'结果已保存: {output_file}')

        # 统计
        label_counts = results_df['label_name'].value_counts()
        print('\\n结果统计:')
        for label, count in label_counts.items():
            print(f'  {label}: {count}')

    else:
        parser.print_help()
        print('\\n示例:')
        print('  python inference.py --text-a "QA76.5 .C64" --text-b "QA76.6 .A12"')
        print('  python inference.py --batch-file data/synthetic_pairs/lcc_training_data.csv')


if __name__ == '__main__':
    main()
