"""
PyTorch Dataset for LC 索书号排序

功能：
- 从 CSV 读取 (text_a, text_b, label)
- 使用 tokenizer 编码
- 返回 (input_ids_a, input_ids_b, label)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from .tokenizer import CharTokenizer


class LCCPairDataset(Dataset):
    """LC 索书号对数据集"""

    def __init__(self, csv_path: str, tokenizer: CharTokenizer, max_seq_len: int = 64):
        """
        Args:
            csv_path: CSV 文件路径
            tokenizer: CharTokenizer 实例
            max_seq_len: 最大序列长度
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 验证数据格式
        assert 'text_a' in self.df.columns, "CSV 缺少 'text_a' 列"
        assert 'text_b' in self.df.columns, "CSV 缺少 'text_b' 列"
        assert 'label' in self.df.columns, "CSV 缺少 'label' 列"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Returns:
            (input_ids_a, input_ids_b, label) 三个 tensor
        """
        row = self.df.iloc[idx]

        text_a = str(row['text_a'])
        text_b = str(row['text_b'])
        label = int(row['label'])

        # 编码
        ids_a = self.tokenizer.encode(text_a, add_padding=True)
        ids_b = self.tokenizer.encode(text_b, add_padding=True)

        # 转为 tensor
        input_ids_a = torch.tensor(ids_a, dtype=torch.long)
        input_ids_b = torch.tensor(ids_b, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return input_ids_a, input_ids_b, label

    def get_label_distribution(self) -> dict:
        """获取标签分布"""
        return self.df['label'].value_counts().to_dict()

    def class_weights(self) -> torch.Tensor:
        """
        计算类别权重（用于处理不平衡数据）

        Returns:
            每个类别的权重 tensor
        """
        label_counts = self.df['label'].value_counts().sort_index()
        total = len(self.df)
        num_classes = len(label_counts)

        weights = total / (num_classes * label_counts.values)
        return torch.FloatTensor(weights)


def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    自定义 collate_fn（其实由于 padding 已在 dataset 中完成，这里只是 stack）

    Args:
        batch: list of (input_ids_a, input_ids_b, label) tuples

    Returns:
        (batch_ids_a, batch_ids_b, batch_labels)
    """
    input_ids_a_list = []
    input_ids_b_list = []
    labels_list = []

    for ids_a, ids_b, label in batch:
        input_ids_a_list.append(ids_a)
        input_ids_b_list.append(ids_b)
        labels_list.append(label)

    batch_ids_a = torch.stack(input_ids_a_list)
    batch_ids_b = torch.stack(input_ids_b_list)
    batch_labels = torch.stack(labels_list)

    return batch_ids_a, batch_ids_b, batch_labels


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    tokenizer: CharTokenizer,
    batch_size: int = 64,
    max_seq_len: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader

    Args:
        train_csv: 训练集 CSV 路径
        val_csv: 验证集 CSV 路径
        tokenizer: Tokenizer
        batch_size: 批次大小
        max_seq_len: 最大序列长度
        num_workers: DataLoader worker 数量

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = LCCPairDataset(train_csv, tokenizer, max_seq_len)
    val_dataset = LCCPairDataset(val_csv, tokenizer, max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


def split_train_val(csv_path: str, val_ratio: float = 0.1, random_seed: int = 42) -> Tuple[str, str]:
    """
    将单个 CSV 划分为训练集和验证集

    Args:
        csv_path: 原始 CSV 路径
        val_ratio: 验证集比例
        random_seed: 随机种子

    Returns:
        (train_csv_path, val_csv_path)
    """
    import os
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)

    # 分层采样（保持标签分布）
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=df['label']
    )

    # 生成输出路径
    base_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    train_path = os.path.join(base_dir, f"{base_name}_train.csv")
    val_path = os.path.join(base_dir, f"{base_name}_val.csv")

    # 保存
    train_df.to_csv(train_path, index=False, quoting=1)  # quoting=1 保留引号
    val_df.to_csv(val_path, index=False, quoting=1)

    print(f"训练集: {len(train_df)} 样本 -> {train_path}")
    print(f"验证集: {len(val_df)} 样本 -> {val_path}")

    return train_path, val_path


if __name__ == "__main__":
    # 测试
    from .tokenizer import CharTokenizer

    tokenizer = CharTokenizer(vocab=CharTokenizer.get_default_vocab())

    # 创建测试数据集
    csv_path = "/home/lostbook/Documents/Lost_Found_book/logic/lcc_training_data.csv"

    # 划分训练集和验证集
    # train_csv, val_csv = split_train_val(csv_path)

    # 或者直接加载完整数据集测试
    dataset = LCCPairDataset(csv_path, tokenizer)

    print(f"数据集大小: {len(dataset)}")
    print(f"标签分布: {dataset.get_label_distribution()}")

    # 测试获取单个样本
    ids_a, ids_b, label = dataset[0]
    print(f"\n样本 0:")
    print(f"  input_ids_a shape: {ids_a.shape}")
    print(f"  input_ids_b shape: {ids_b.shape}")
    print(f"  label: {label}")

    # 测试 DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch_ids_a, batch_ids_b, batch_labels = next(iter(loader))

    print(f"\nBatch:")
    print(f"  batch_ids_a shape: {batch_ids_a.shape}")
    print(f"  batch_ids_b shape: {batch_ids_b.shape}")
    print(f"  batch_labels: {batch_labels}")
