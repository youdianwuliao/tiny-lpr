"""
车牌识别器 — CRNN + CTC
轻量 CNN 提取特征 → BiLSTM → CTC 解码
模型 < 2M 参数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLPR(nn.Module):
    """超轻量车牌识别网络

    输入: (B, 3, 48, 160) 裁剪后的车牌图片
    输出: (B, T, 68) 每个时间步的概率分布

    字符集: 68 类（省份简写 + 字母 + 数字 + blank）
    """

    def __init__(self, num_classes: int = 68, hidden_size: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # CNN 特征提取（下采样到 H=1, W≈40）
        self.cnn = nn.Sequential(
            # Block 1: 48x160 → 24x80
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 24x80 → 12x40
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 12x40 → 6x20
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 6x20 → 3x20
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            # Block 5: 3x20 → 1x20
            nn.Conv2d(128, hidden_size, 3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),  # → (128, 1, 20)
        )

        # BiLSTM 序列建模
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )

        # 分类头
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 48, 160)
        Returns:
            logits: (B, T, num_classes)
        """
        # CNN
        x = self.cnn(x)  # (B, C, 1, W)
        x = x.squeeze(2)  # (B, C, W)
        x = x.permute(0, 2, 1)  # (B, W, C)

        # RNN
        x, _ = self.rnn(x)  # (B, W, 2*C)

        # FC
        x = self.dropout(x)
        x = self.fc(x)  # (B, W, num_classes)

        return F.log_softmax(x, dim=-1)

    @torch.no_grad()
    def decode(self, log_probs: torch.Tensor, chars: list) -> list:
        """CTC 贪心解码 → 车牌字符串列表"""
        _, max_indices = log_probs.max(dim=-1)  # (B, T)
        results = []
        for indices in max_indices:
            merged = []
            prev = -1
            for idx in indices.tolist():
                if idx != prev and idx != 0:  # 跳过 blank(0) 和重复
                    merged.append(chars[idx])
                prev = idx
            results.append(''.join(merged))
        return results


def create_recognizer(num_classes: int = 68) -> TinyLPR:
    """创建识别器实例"""
    model = TinyLPR(num_classes=num_classes)
    # 打印参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"识别器参数量: {params:,} (≈ {params/1e6:.1f}M)")
    return model


if __name__ == "__main__":
    model = create_recognizer()
    x = torch.randn(1, 3, 48, 160)
    out = model(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")