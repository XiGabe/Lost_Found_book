"""
批处理性能测试 - 模拟真实书架检测场景

每批15本书，检查排序是否正确

使用方法:
    python benchmark_batch.py --data lcc_training_data.csv --batches test_batches.txt
"""

import os
import sys
import argparse
import time
import csv
from typing import List, Tuple, Dict

import torch
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from .tokenizer import CharTokenizer
from .siamese_lstm import SiameseBiLSTM


class BatchComparator:
    """批次比较器 - 处理一批书的排序检查"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.label_names = ['In_Order', 'Out_of_Order', 'Duplicate']

    def compare_pair(self, text_a: str, text_b: str) -> Tuple[int, float]:
        """
        比较一对索书号

        Returns:
            (label, confidence)
        """
        ids_a = self.tokenizer.encode(text_a, add_padding=True)
        ids_b = self.tokenizer.encode(text_b, add_padding=True)

        input_a = torch.tensor([ids_a]).to(self.device)
        input_b = torch.tensor([ids_b]).to(self.device)

        with torch.no_grad():
            output = self.model(input_a, input_b)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            confidence = probs[0][pred].item()

        return int(pred.item()), confidence

    def check_batch_ordering(self, books: List[str]) -> Dict:
        """
        检查一批书的排序

        Args:
            books: 书籍索书号列表 (15本)

        Returns:
            {
                'total_pairs': 总比较对数,
                'out_of_order': 错误对数,
                'duplicates': 重复对数,
                'is_correctly_ordered': 是否整体有序,
                'inference_time': 推理时间,
                'problem_pairs': [(i, j, text_a, text_b, label), ...]
            }
        """
        num_books = len(books)
        total_pairs = num_books * (num_books - 1) // 2

        out_of_order_count = 0
        duplicate_count = 0
        problem_pairs = []

        start_time = time.time()

        # 比较所有对
        for i in range(num_books):
            for j in range(i + 1, num_books):
                label, conf = self.compare_pair(books[i], books[j])

                if label == 1:  # Out_of_Order
                    out_of_order_count += 1
                    problem_pairs.append({
                        'pos': (i, j),
                        'book_a': books[i],
                        'book_b': books[j],
                        'reason': 'Out_of_Order',
                        'confidence': conf
                    })
                elif label == 2:  # Duplicate
                    duplicate_count += 1
                    problem_pairs.append({
                        'pos': (i, j),
                        'book_a': books[i],
                        'book_b': books[j],
                        'reason': 'Duplicate',
                        'confidence': conf
                    })

        inference_time = time.time() - start_time

        # 判断整体是否有序（允许少量错误）
        # 如果 out_of_order_count == 0，则完全有序
        # 如果有少量错误，可能只是个别问题
        is_correctly_ordered = out_of_order_count == 0

        return {
            'total_pairs': total_pairs,
            'out_of_order': out_of_order_count,
            'duplicates': duplicate_count,
            'is_correctly_ordered': is_correctly_ordered,
            'inference_time': inference_time,
            'problem_pairs': problem_pairs
        }


def load_batches_from_file(filepath: str) -> List[Dict]:
    """从文件加载批次数据"""
    batches = []
    current_batch = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # 批次标题行
            if line.startswith('# Batch'):
                if current_batch:
                    batches.append(current_batch)
                parts = line.split()
                # 格式: # Batch 0: ordered=True
                batch_id = int(parts[2].rstrip(':'))
                ordered = parts[3].split('=')[1] == 'True'
                current_batch = {
                    'batch_id': batch_id,
                    'is_ordered': ordered,
                    'books': []
                }
            # 书籍行
            elif current_batch is not None:
                current_batch['books'].append(line)

        if current_batch:
            batches.append(current_batch)

    return batches


def evaluate_on_test_set(model, tokenizer, device, test_csv: str, max_samples: int = 5000):
    """在标准测试集上评估"""
    print("\n" + "="*60)
    print("标准测试集评估")
    print("="*60)

    # 读取CSV
    test_samples = []
    with open(test_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            test_samples.append((row['text_a'], row['text_b'], int(row['label'])))

    print(f"\n评估 {len(test_samples)} 个测试样本...")

    all_preds = []
    all_labels = []
    times = []

    # 预热
    print("预热中...")
    for _ in range(10):
        for text_a, text_b, _ in test_samples[:10]:
            ids_a = tokenizer.encode(text_a, add_padding=True)
            ids_b = tokenizer.encode(text_b, add_padding=True)
            input_a = torch.tensor([ids_a]).to(device)
            input_b = torch.tensor([ids_b]).to(device)
            with torch.no_grad():
                _ = model(input_a, input_b)

    # 测试
    start_time = time.time()
    for text_a, text_b, true_label in tqdm(test_samples, desc="测试"):
        ids_a = tokenizer.encode(text_a, add_padding=True)
        ids_b = tokenizer.encode(text_b, add_padding=True)

        input_a = torch.tensor([ids_a]).to(device)
        input_b = torch.tensor([ids_b]).to(device)

        iter_start = time.time()
        with torch.no_grad():
            output = model(input_a, input_b)
            pred = torch.argmax(output, dim=1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        iter_time = (time.time() - iter_start) * 1000
        times.append(iter_time)

        all_preds.append(pred.item())
        all_labels.append(true_label)

    total_time = time.time() - start_time
    times = np.array(times)

    # 计算指标
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    accuracy = correct / len(all_preds)

    # 各类别统计
    label_counts = {0: 0, 1: 0, 2: 0}
    correct_counts = {0: 0, 1: 0, 2: 0}

    for p, l in zip(all_preds, all_labels):
        label_counts[l] += 1
        if p == l:
            correct_counts[l] += 1

    print(f"\n总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"平均推理速度: {np.mean(times):.3f} ms/pair")
    print(f"中位数推理速度: {np.median(times):.3f} ms/pair")
    print(f"吞吐量: {len(test_samples) / total_time:.1f} pairs/sec")

    print(f"\n各类别准确率:")
    label_names = ['In_Order', 'Out_of_Order', 'Duplicate']
    for i, name in enumerate(label_names):
        if label_counts[i] > 0:
            acc = correct_counts[i] / label_counts[i]
            print(f"  {name}: {acc:.4f} ({correct_counts[i]}/{label_counts[i]})")

    return {
        'accuracy': accuracy,
        'inference_time': total_time,
        'times': times
    }


def benchmark_batches(model, tokenizer, device, batches: List[Dict], max_batches: int = None):
    """批处理性能测试"""

    if max_batches:
        batches = batches[:max_batches]

    print("\n" + "="*60)
    print(f"批处理性能测试 ({len(batches)} 个批次)")
    print("="*60)

    comparator = BatchComparator(model, tokenizer, device)

    total_books = 0
    total_pairs = 0
    total_inference_time = 0

    batch_results = []

    for batch in tqdm(batches, desc="批处理测试"):
        result = comparator.check_batch_ordering(batch['books'])

        result['batch_id'] = batch['batch_id']
        result['ground_truth_ordered'] = batch['is_ordered']
        result['num_books'] = len(batch['books'])

        batch_results.append(result)

        total_books += result['num_books']
        total_pairs += result['total_pairs']
        total_inference_time += result['inference_time']

    # 统计分析
    print(f"\n{'='*60}")
    print("批处理测试结果")
    print(f"{'='*60}")

    print(f"\n总体统计:")
    print(f"  总批次数: {len(batches)}")
    print(f"  总书本数: {total_books:,}")
    print(f"  总比较次数: {total_pairs:,}")
    print(f"  总推理时间: {total_inference_time:.2f} sec")
    print(f"  平均每批时间: {total_inference_time / len(batches) * 1000:.2f} ms")
    print(f"  平均每对时间: {total_inference_time / total_pairs * 1000:.2f} ms")
    print(f"  吞吐量: {total_pairs / total_inference_time:.1f} pairs/sec")

    # 批次分类统计
    correctly_ordered = sum(1 for r in batch_results if r['is_correctly_ordered'])
    print(f"\n批次分类:")
    print(f"  有序批次: {correctly_ordered}/{len(batches)} ({correctly_ordered/len(batches)*100:.1f}%)")
    print(f"  无序批次: {len(batches) - correctly_ordered}/{len(batches)} ({(len(batches) - correctly_ordered)/len(batches)*100:.1f}%)")

    # 问题对统计
    total_out_of_order = sum(r['out_of_order'] for r in batch_results)
    total_duplicates = sum(r['duplicates'] for r in batch_results)

    print(f"\n问题对统计:")
    print(f"  顺序错误对数: {total_out_of_order}/{total_pairs} ({total_out_of_order/total_pairs*100:.2f}%)")
    print(f"  重复对数: {total_duplicates}/{total_pairs} ({total_duplicates/total_pairs*100:.2f}%)")

    # 与真实标签对比
    correct_predictions = sum(
        1 for r in batch_results
        if r['is_correctly_ordered'] == r['ground_truth_ordered']
    )

    print(f"\n批次级别准确率:")
    print(f"  准确预测批次: {correct_predictions}/{len(batches)} ({correct_predictions/len(batches)*100:.2f}%)")

    # 显示问题批次示例
    problem_batches = [r for r in batch_results if r['problem_pairs']]

    if problem_batches:
        print(f"\n问题批次示例 (前3个):")
        for i, result in enumerate(problem_batches[:3]):
            print(f"\n  批次 {result['batch_id']}:")
            print(f"    预测有序: {result['is_correctly_ordered']}")
            print(f"    实际有序: {result['ground_truth_ordered']}")
            print(f"    错误对数: {result['out_of_order']}")
            print(f"    重复对数: {result['duplicates']}")

            if result['problem_pairs']:
                print(f"    问题对示例:")
                for pair in result['problem_pairs'][:3]:
                    i, j = pair['pos']
                    print(f"      [{i+1}, {j+1}] {pair['reason']}:")
                    print(f"        A: {pair['book_a']}")
                    print(f"        B: {pair['book_b']}")
                    print(f"        置信度: {pair['confidence']:.2%}")

    return batch_results


def main():
    parser = argparse.ArgumentParser(description='批处理性能测试')
    parser.add_argument('--checkpoint', type=str,
                        default='weights/comparator.pth',
                        help='模型路径')
    parser.add_argument('--data', type=str,
                        default='lcc_training_data.csv',
                        help='测试数据 CSV')
    parser.add_argument('--batches', type=str,
                        default='test_batches.txt',
                        help='批次数据文件')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='最大测试批次数')
    parser.add_argument('--max-samples', type=int, default=5000,
                        help='最大测试样本数')
    parser.add_argument('--skip-test-set', action='store_true',
                        help='跳过标准测试集评估')
    parser.add_argument('--skip-batches', action='store_true',
                        help='跳过批处理测试')

    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    tokenizer = CharTokenizer(vocab=CharTokenizer.get_default_vocab())
    model = SiameseBiLSTM(vocab_size=tokenizer.get_vocab_size())
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"模型来自 epoch {checkpoint.get('epoch', 'N/A')}")

    # 1. 标准测试集评估
    if not args.skip_test_set:
        test_metrics = evaluate_on_test_set(model, tokenizer, device, args.data, args.max_samples)

    # 2. 批处理性能测试
    if not args.skip_batches and os.path.exists(args.batches):
        print("\n加载批次数据...")
        batches = load_batches_from_file(args.batches)
        print(f"加载了 {len(batches)} 个批次")

        batch_results = benchmark_batches(
            model, tokenizer, device, batches, args.max_batches
        )

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == '__main__':
    main()
