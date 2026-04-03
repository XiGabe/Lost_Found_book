"""
ESIM Bi-LSTM Model for LC Call Number Ordering (V6 Diff Amplifier)

Architecture:
- Twin Tower Independent Encoder (avoids Padding isolation band)
- Soft Cross-Attention alignment between A and B
- Concatenated features [x, align, x-align, x*align] for enhanced inference
- BiLSTM composition layer
- Multi-Pooling (Max + Mean) prevents tail feature loss
- V6: Diff Amplifier (2-layer MLP + GELU) to amplify tiny differences at tail
  and suppress massive prefix differences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ESIMBiLSTMComparator(nn.Module):
    """
    V6: ESIM-style Siamese Attention Network with Diff Amplifier

    The Diff Amplifier learns to suppress prefix asymmetry noise while
    amplifying the tiny differences at the tail (W vs X, v.15 vs v.16)
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
        super(ESIMBiLSTMComparator, self).__init__()

        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx

        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(dropout)

        # 2. Independent encoder (Twin Tower avoids Padding isolation)
        self.lstm_encode = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 3. Inference composer (processes Attention-composed features)
        self.lstm_compose = nn.LSTM(
            hidden_dim * 4,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 4. V6: Diff Amplifier - suppresses prefix noise, amplifies tail differences
        self.diff_amplifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )

        # 5. Fusion classifier: [v_a, v_b, amplified_diff, hadamard] = 8H
        classifier_input_dim = hidden_dim * 8
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm_encode.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for name, param in self.lstm_compose.named_parameters():
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

    def soft_attention_align(self, x1, x2, mask1, mask2):
        """
        Compute Soft Cross-Attention between A and B
        Enables L2 and M53 to align across length differences
        """
        attention_weight = torch.bmm(x1, x2.transpose(1, 2))

        mask1_float = mask1.float().unsqueeze(2)
        mask2_float = mask2.float().unsqueeze(1)

        weight1 = attention_weight.masked_fill(mask2_float == 0, -1e9)
        weight2 = attention_weight.transpose(1, 2).masked_fill(mask1_float == 0, -1e9)

        prob1 = F.softmax(weight1, dim=2)
        prob2 = F.softmax(weight2, dim=2)

        x1_align = torch.bmm(prob1, x2)
        x2_align = torch.bmm(prob2, x1)

        return x1_align, x2_align

    def pooling(self, x, mask):
        """Use both Max and Mean Pooling to preserve global and extreme features"""
        mask_expand = mask.unsqueeze(-1).expand_as(x)
        x_masked = x.masked_fill(~mask_expand, -1e9)
        v_max, _ = x_masked.max(dim=1)

        x_masked_mean = x.masked_fill(~mask_expand, 0)
        v_mean = x_masked_mean.sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1e-9)

        return torch.cat([v_max, v_mean], dim=1)

    def forward(self, input_ids_a: torch.Tensor, input_ids_b: torch.Tensor) -> torch.Tensor:
        mask_a = input_ids_a != self.padding_idx
        mask_b = input_ids_b != self.padding_idx

        # 1. Independent encoding
        emb_a = self.embed_dropout(self.embedding(input_ids_a))
        emb_b = self.embed_dropout(self.embedding(input_ids_b))

        out_a, _ = self.lstm_encode(emb_a)
        out_b, _ = self.lstm_encode(emb_b)

        # 2. Cross-attention alignment
        align_a, align_b = self.soft_attention_align(out_a, out_b, mask_a, mask_b)

        # 3. Enhanced inference feature concatenation [orig, align, diff, prod]
        comp_a = torch.cat([out_a, align_a, out_a - align_a, out_a * align_a], dim=-1)
        comp_b = torch.cat([out_b, align_b, out_b - align_b, out_b * align_b], dim=-1)

        # 4. Composition inference layer
        compose_a, _ = self.lstm_compose(comp_a)
        compose_b, _ = self.lstm_compose(comp_b)

        # 5. Dual pooling
        v_a = self.pooling(compose_a, mask_a)
        v_b = self.pooling(compose_b, mask_b)

        # 6. V6: Raw diff fed through Diff Amplifier to suppress prefix noise
        raw_diff = v_a - v_b
        amplified_diff = self.diff_amplifier(raw_diff)
        hadamard = v_a * v_b

        # 7. Classification with [v_a, v_b, amplified_diff, hadamard]
        pooled_features = torch.cat([v_a, v_b, amplified_diff, hadamard], dim=1)
        logits = self.classifier(pooled_features)
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Backward compatibility alias
InteractiveBiLSTM = ESIMBiLSTMComparator
