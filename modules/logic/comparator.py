"""
Siamese Bi-LSTM 模型 for LC 索书号排序

架构：
- 共享 Embedding 层
- 共享 Bi-LSTM 编码器
- 拼接两个向量 + 差值特征
- MLP 分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SiameseBiLSTM(nn.Module):
    """
    Siamese Bi-LSTM 用于文本对比较

    输入：两个文本序列 (text_a, text_b)
    输出：3分类 logits [0, 1, 2]
        - 0: In_Order (A < B)
        - 1: Duplicate (A = B)
        - 2: Out_of_Order (A > B)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
        padding_idx: int = 0,
    ):
        """
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 字符嵌入维度
            hidden_dim: LSTM 隐藏层维度
            num_layers: LSTM 层数
            dropout: Dropout 比例
            num_classes: 分类数量
            padding_idx: Padding token 的索引
        """
        super(SiameseBiLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 1. 共享 Embedding 层
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        # 2. 共享 Bi-LSTM 编码器
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # 双向会拼接，所以这里是 hidden_dim // 2
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 3. Dropout
        self.dropout = nn.Dropout(dropout)

        # 4. 分类器
        # 输入: [vec_a; vec_b; |vec_a - vec_b|] (拼接 + 差值特征)
        # Mean Pooling 输出 hidden_dim (双向拼接后)，拼接后 + 差值 = hidden_dim * 3
        classifier_input_dim = hidden_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        使用共享编码器编码输入 (Mean Pooling 忽略 Padding)

        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            encoded: (batch_size, hidden_dim) - 序表示
        """
        # Embedding: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)

        # LSTM: (batch_size, seq_len, hidden_dim)
        # hidden_dim = (hidden_dim // 2) * 2 双向拼接
        lstm_out, _ = self.lstm(embedded)

        # 创建掩码，找出非 Padding 的部分
        mask = (input_ids != 0).float().unsqueeze(-1)  # (batch_size, seq_len, 1)

        # Mask padding 位置的特征
        lstm_out = lstm_out * mask

        # 计算每个样本的实际长度
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)  # (batch_size, 1, 1)

        # Mean Pooling: 对所有时间步求平均（自动忽略 padding 的 0）
        hidden = lstm_out.sum(dim=1) / lengths.squeeze(-1)  # (batch_size, hidden_dim * 2)

        return hidden

    def forward(
        self,
        input_ids_a: torch.Tensor,
        input_ids_b: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids_a: (batch_size, seq_len)
            input_ids_b: (batch_size, seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        # 编码两个输入（共享权重）
        vec_a = self._encode(input_ids_a)  # (batch_size, hidden_dim)
        vec_b = self._encode(input_ids_b)  # (batch_size, hidden_dim)

        # Dropout
        vec_a = self.dropout(vec_a)
        vec_b = self.dropout(vec_b)

        # 拼接特征: [vec_a; vec_b; |vec_a - vec_b|]
        diff = torch.abs(vec_a - vec_b)
        combined = torch.cat([vec_a, vec_b, diff], dim=1)  # (batch_size, hidden_dim * 3)

        # 分类
        logits = self.classifier(combined)  # (batch_size, num_classes)

        return logits

    def predict(
        self,
        input_ids_a: torch.Tensor,
        input_ids_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测（带概率）

        Returns:
            predictions: (batch_size,) - 预测的类别
            probabilities: (batch_size, num_classes) - 每个类别的概率
        """
        logits = self.forward(input_ids_a, input_ids_b)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs


class SiameseBiLSTMWithAttention(SiameseBiLSTM):
    """
    带注意力池化的版本（可选）
    """

    def __init__(self, *args, **kwargs):
        super(SiameseBiLSTMWithAttention, self).__init__(*args, **kwargs)

        # 注意力权重
        self.attention = nn.Linear(self.hidden_dim, 1)

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        使用注意力池化编码 (Masked Softmax 忽略 Padding)
        """
        # Embedding
        embedded = self.embedding(input_ids)

        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim)

        # 1. 计算原始 Attention Logits
        attn_logits = self.attention(lstm_out)  # (batch_size, seq_len, 1)

        # 2. 获取 Mask 并把 Padding 位置替换为负无穷
        mask = (input_ids != 0).unsqueeze(-1)   # boolean mask
        attn_logits = attn_logits.masked_fill(~mask, -1e9)  # 核心修复

        # 3. 现在的 Softmax 就只会在有效字符之间分配 100% 的权重了
        attn_weights = F.softmax(attn_logits, dim=1)  # (batch_size, seq_len, 1)

        # 加权求和
        hidden = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_dim)

        return hidden


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    vocab_size = 100
    batch_size = 4
    seq_len = 64

    model = SiameseBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        num_classes=3
    )

    print(f"模型参数量: {count_parameters(model):,}")

    # 测试前向传播
    input_ids_a = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_ids_b = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(input_ids_a, input_ids_b)

    print(f"输入 shape: {input_ids_a.shape}")
    print(f"输出 shape: {logits.shape}")

    # 测试预测
    preds, probs = model.predict(input_ids_a, input_ids_b)
    print(f"预测: {preds}")
    print(f"概率:\n{probs}")
