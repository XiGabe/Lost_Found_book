"""
Lost Book Robot - 索书号排序推理脚本 (V3.0 适配版)
功能：对两个 OCR 识别出的索书号字符串进行严格的 LCC 语义排序判断
"""

import argparse
import sys
import re
from pathlib import Path
import torch

from modules.logic.tokenizer import CharTokenizer
from modules.logic.comparator import ESIMBiLSTMComparator

# ==========================================
# 全局常量与配置
# ==========================================
MAX_SEQ_LEN = 64  
_model = None
_tokenizer = None
_device = None

def preprocess_lcc(text: str) -> str:
    """
    [极简清洗版] 为了保持与训练集（脏数据）的数据分布一致，
    移除了强行换行的逻辑，只修复 OCR 特有的断裂问题。
    """
    if not text: return ""

    # 0. 基础清洗：把多个连续的空白字符压平成一个单空格
    text = re.sub(r'\s+', ' ', text).strip()

    # 1. 修复 OCR 盲区 A：字母内部断裂 (例如 "B R 330" -> "BR 330", "P Q 2283" -> "PQ 2283")
    # 连续执行两次以应对可能的三个字母断裂情况
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', r'\1\2', text)
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', r'\1\2', text)

    # 2. 确保主类字母和数字间有空格 (例如 "PR6003" -> "PR 6003")
    text = re.sub(r'^([A-Za-z]+)(\d)', r'\1 \2', text)

    return text

def pad_sequence_ids(ids: list, max_len: int = MAX_SEQ_LEN) -> list:
    """
    序列长度对齐。确保输入 Tensor 的维度永远是 [batch, 64]。
    """
    if len(ids) < max_len:
        return ids + [0] * (max_len - len(ids))
    return ids[:max_len]

def load_model(checkpoint_path: str, device: torch.device = None):
    global _model, _tokenizer, _device

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if _model is None:
        print(f'正在加载最佳模型引擎: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 显式创建指定长度的 Tokenizer
        _tokenizer = CharTokenizer(
            vocab=CharTokenizer.get_default_vocab(), 
            max_seq_len=MAX_SEQ_LEN
        )

        # 初始化模型架构 (V4.2 ESIMBiLSTMComparator)
        _model = ESIMBiLSTMComparator(
            vocab_size=_tokenizer.get_vocab_size(),
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            num_classes=3,
        )

        _model.load_state_dict(checkpoint['model_state_dict'])
        _model = _model.to(device)
        _model.eval()
        _device = device

        print(f'模型状态：Epoch {checkpoint.get("epoch")}, 验证集准确率 {checkpoint.get("accuracy", 0):.4f}')

    return _model, _tokenizer, _device

def compare_lcc(text_a: str, text_b: str, checkpoint_path: str = None) -> dict:
    model, tokenizer, device = load_model(checkpoint_path)

    # 1. 极简预处理 (解决字母断裂，保留模型习惯的乱序格式)
    text_a_proc = preprocess_lcc(text_a)
    text_b_proc = preprocess_lcc(text_b)

    # 2. 编码并强制对齐 64 长度
    ids_a = pad_sequence_ids(tokenizer.encode(text_a_proc, add_padding=False))
    ids_b = pad_sequence_ids(tokenizer.encode(text_b_proc, add_padding=False))

    # 3. 转为 Tensor
    input_ids_a = torch.tensor([ids_a], dtype=torch.long).to(device)
    input_ids_b = torch.tensor([ids_b], dtype=torch.long).to(device)

    # 4. 推理
    with torch.no_grad():
        preds, probs = model.predict(input_ids_a, input_ids_b)

    pred_label = preds.item()
    probs_np = probs.cpu().numpy()[0]
    label_names = ['In_Order', 'Duplicate', 'Out_of_Order']

    return {
        'label': int(pred_label),
        'label_name': label_names[pred_label],
        'confidence': float(probs_np[pred_label]),
        'probabilities': {name: float(probs_np[i]) for i, name in enumerate(label_names)}
    }

def main():
    parser = argparse.ArgumentParser(description='Lost Book Robot - 索书号语义排序判断器')
    parser.add_argument('--text-a', type=str, help='左侧书籍索书号 (A)')
    parser.add_argument('--text-b', type=str, help='右侧书籍索书号 (B)')
    parser.add_argument('--checkpoint', type=str, default='weights/comparator.pth', help='模型路径')
    
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f'错误: 未找到模型文件 {args.checkpoint}')
        sys.exit(1)

    if args.text_a and args.text_b:
        res = compare_lcc(args.text_a, args.text_b, args.checkpoint)
        
        print('\n' + '='*50)
        print(f'【检测报告】')
        print(f'A: {args.text_a} (处理后: {preprocess_lcc(args.text_a)})')
        print(f'B: {args.text_b} (处理后: {preprocess_lcc(args.text_b)})')
        print('-'*50)
        print(f'判定结果: {res["label_name"]}')
        print(f'置信度: {res["confidence"]:.4f}')
        
        # 业务逻辑解释
        if res['label'] == 0:
            print('状态建议: ✓ 符合 LC 分类法，顺序正确。')
        elif res['label'] == 1:
            print('状态建议: ⚠ 检测为重复书籍或高度相似噪音，建议复核。')
        else:
            print('状态建议: 🚨 严重警报：书籍摆放错位，请立即交换 A 和 B 的位置！')
        print('='*50 + '\n')
    else:
        parser.print_help()

if __name__ == '__main__':
    main()