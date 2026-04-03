"""
字符级 Tokenizer for LC 索书号

功能：
- 构建字符表：包含所有 ASCII 可见字符 + 常见 OCR 混淆字符
- 特殊标记：<PAD>, <UNK>
- encode(): 将字符串转为 token ids
- decode(): 将 token ids 转回字符串
"""

import string
import pickle
from typing import List, Tuple, Dict
from collections import Counter


class CharTokenizer:
    """字符级 Tokenizer"""

    # 特殊标记
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"  # Start of sequence (可选)
    EOS_TOKEN = "<EOS>"  # End of sequence (可选)

    def __init__(self, vocab: Dict[str, int] = None, max_seq_len: int = 64):
        """
        Args:
            vocab: 预定义的词汇表字典 {char: idx}
            max_seq_len: 最大序列长度
        """
        self.max_seq_len = max_seq_len

        if vocab is not None:
            self.char2idx = vocab
            self.idx2char = {idx: char for char, idx in vocab.items()}
        else:
            self.char2idx = {}
            self.idx2char = {}

    @classmethod
    def from_texts(cls, texts: List[str], max_seq_len: int = 64,
                   min_freq: int = 1) -> 'CharTokenizer':
        """
        从文本列表构建词汇表

        Args:
            texts: 文本列表
            max_seq_len: 最大序列长度
            min_freq: 最小词频阈值
        """
        # 统计字符频率
        char_counter = Counter()
        for text in texts:
            char_counter.update(text)

        # 构建词汇表
        vocab = {}

        # 添加特殊标记
        special_tokens = [cls.PAD_TOKEN, cls.UNK_TOKEN]
        for idx, token in enumerate(special_tokens):
            vocab[token] = idx

        # 添加满足频率要求的字符
        idx = len(special_tokens)
        for char, freq in sorted(char_counter.items(), key=lambda x: x[1], reverse=True):
            if freq >= min_freq and char not in vocab:
                vocab[char] = idx
                idx += 1

        return cls(vocab=vocab, max_seq_len=max_seq_len)

    @classmethod
    def get_default_vocab(cls) -> Dict[str, int]:
        """
        获取默认词汇表（包含 ASCII 可见字符 + 常见 OCR 混淆字符）
        """
        # 特殊标记
        SEP_TOKEN = "<SEP>"
        vocab = {
            cls.PAD_TOKEN: 0,
            cls.UNK_TOKEN: 1,
            SEP_TOKEN: 2,
        }

        idx = 3

        # 数字
        for c in '0123456789':
            vocab[c] = idx
            idx += 1

        # 大写字母
        for c in string.ascii_uppercase:
            vocab[c] = idx
            idx += 1

        # 小写字母
        for c in string.ascii_lowercase:
            vocab[c] = idx
            idx += 1

        # 常见符号和 OCR 噪声字符
        special_chars = ' .-|,/\n\t:;\'"()[]{}!?@#$%^&*_+<=>~`'
        for c in special_chars:
            if c not in vocab:
                vocab[c] = idx
                idx += 1

        return vocab

    def encode(self, text: str, add_padding: bool = True) -> List[int]:
        """
        将字符串编码为 token ids

        Args:
            text: 输入字符串
            add_padding: 是否添加 padding

        Returns:
            token ids 列表
        """
        # 截断到最大长度
        text = text[:self.max_seq_len]

        # 编码
        ids = [self.char2idx.get(c, self.char2idx[self.UNK_TOKEN]) for c in text]

        # Padding
        if add_padding:
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.char2idx[self.PAD_TOKEN]] * pad_len
            else:
                ids = ids[:self.max_seq_len]

        return ids

    def decode(self, ids: List[int], remove_padding: bool = True) -> str:
        """
        将 token ids 解码为字符串

        Args:
            ids: token ids 列表
            remove_padding: 是否移除 padding

        Returns:
            解码后的字符串
        """
        chars = []
        for idx in ids:
            if remove_padding and idx == self.char2idx[self.PAD_TOKEN]:
                continue
            chars.append(self.idx2char.get(idx, self.UNK_TOKEN))
        return ''.join(chars)

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """
        批量编码

        Args:
            texts: 文本列表

        Returns:
            token ids 矩阵
        """
        return [self.encode(text) for text in texts]

    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.char2idx)

    def save(self, path: str):
        """保存 tokenizer 到文件"""
        with open(path, 'wb') as f:
            pickle.dump({
                'char2idx': self.char2idx,
                'max_seq_len': self.max_seq_len,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        """从文件加载 tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(vocab=data['char2idx'], max_seq_len=data['max_seq_len'])

    def __len__(self) -> int:
        return len(self.char2idx)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={len(self.char2idx)}, max_seq_len={self.max_seq_len})"


def build_tokenizer_from_csv(csv_path: str, max_seq_len: int = 64) -> CharTokenizer:
    """
    从 CSV 数据文件构建 tokenizer

    Args:
        csv_path: CSV 文件路径
        max_seq_len: 最大序列长度

    Returns:
        CharTokenizer 实例
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    texts = df['text_a'].tolist() + df['text_b'].tolist()
    return CharTokenizer.from_texts(texts, max_seq_len=max_seq_len)


if __name__ == "__main__":
    # 测试
    tokenizer = CharTokenizer(vocab=CharTokenizer.get_default_vocab())

    test_texts = [
        "QA76.5 .C64 2023",
        "OLIN|W721l\n.V30\nF56\n2010",
        "CU3926 .N703 D418 191O",
    ]

    print("Tokenizer 测试:")
    print(f"词汇表大小: {tokenizer.get_vocab_size()}")
    print(f"最大序列长度: {tokenizer.max_seq_len}")
    print()

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"原始: {text!r}")
        print(f"编码: {encoded[:20]}...")  # 只显示前 20 个
        print(f"解码: {decoded!r}")
        print()
