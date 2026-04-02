"""
Siamese Bi-LSTM 模型 for LC 索书号排序 (V3.2 架构升级版)

架构：
- 共享 Embedding 层
- 共享 Bi-LSTM 编码器
- 双重池化策略 (Mean + Max Pooling) -> 解决特征稀释问题
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
    支持双重池化 (Mean & Max Pooling) 以捕捉字符级微小变异
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
            hidden_dim // 2,  # 双向拼接后总维度为 hidden_dim
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 3. Dropout
        self.dropout = nn.Dropout(dropout)

        # 4. 分类器
        # 每个序列经过编码后：[Mean_Pool; Max_Pool] -> 维度是 hidden_dim * 2
        # Siamese Head 拼接: [vec_a; vec_b; |vec_a - vec_b|] -> 总维度 * 3
        classifier_input_dim = (hidden_dim * 2) * 3
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
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
        使用共享编码器编码输入 (双重池化)
        """
        # 1. Embedding & LSTM
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim)

        # 2. 创建掩码
        mask = (input_ids != 0).float().unsqueeze(-1)  # (batch, seq_len, 1)

        # --- Mean Pooling ---
        # 排除 Padding 影响
        masked_out = lstm_out * mask
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        mean_pool = masked_out.sum(dim=1) / lengths.squeeze(-1) # (batch, hidden_dim)

        # --- Max Pooling (核心修改) ---
        # 排除 Padding 干扰：将 Padding 位置设为极小值
        max_input = lstm_out.masked_fill(mask == 0, -1e9)
        max_pool, _ = torch.max(max_input, dim=1) # (batch, hidden_dim)

        # 3. 拼接两种池化特征
        combined = torch.cat([mean_pool, max_pool], dim=1) # (batch, hidden_dim * 2)
        return combined

    def forward(
        self,
        input_ids_a: torch.Tensor,
        input_ids_b: torch.Tensor
    ) -> torch.Tensor:
        # 编码
        vec_a = self._encode(input_ids_a)  # (batch, hidden_dim * 2)
        vec_b = self._encode(input_ids_b)

        # Dropout
        vec_a = self.dropout(vec_a)
        vec_b = self.dropout(vec_b)

        # 拼接特征: [vec_a; vec_b; |vec_a - vec_b|]
        diff = torch.abs(vec_a - vec_b)
        combined = torch.cat([vec_a, vec_b, diff], dim=1) 

        # 分类
        logits = self.classifier(combined)
        return logits

    def predict(
        self,
        input_ids_a: torch.Tensor,
        input_ids_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids_a, input_ids_b)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs


class SiameseBiLSTMWithAttention(SiameseBiLSTM):
    """
    带注意力池化的版本 (已同步升级分类头维度)
    """
    def __init__(self, *args, **kwargs):
        super(SiameseBiLSTMWithAttention, self).__init__(*args, **kwargs)
        # 注意力只需要基于单向量 hidden_dim
        self.attention = nn.Linear(self.hidden_dim, 1)
        
        # 注意：Attention 版本通常只输出单一向量，
        # 如果需要同时使用 Mean/Max/Attn，需要修改 classifier 的维度
        # 此处重写 classifier 适配 Attention 的单输出
        classifier_input_dim = self.hidden_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(kwargs.get('dropout', 0.3)),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        
        attn_logits = self.attention(lstm_out)
        mask = (input_ids != 0).unsqueeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        
        attn_weights = F.softmax(attn_logits, dim=1)
        hidden = torch.sum(attn_weights * lstm_out, dim=1)
        return hidden


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
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

    print(f"V3.2 模型参数量: {count_parameters(model):,}")
    
    input_ids_a = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_ids_b = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids_a, input_ids_b)

    print(f"输出 shape: {logits.shape}") # 应该是 [4, 3]