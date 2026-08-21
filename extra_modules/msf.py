import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class MSF(nn.Module):
    """
    Multi-Scale Feature Fusion (YOLO-compatible)
    Fuses features from 3 kernel scales, then projects back to c2 channels
    """

    def __init__(self, c1, c2):
        """
        c1: 输入通道数（YOLO 自动传入）
        c2: 输出通道数（由 YAML 控制）
        """
        super().__init__()

        self.b1 = Conv(c1, c2, k=3, s=1)
        self.b2 = Conv(c1, c2, k=5, s=1)
        self.b3 = Conv(c1, c2, k=3, s=1, d=2)

        self.bn = nn.BatchNorm2d(c2 * 3)
        self.act = nn.SiLU()
        self.proj = Conv(c2 * 3, c2, k=1, s=1, act=False)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.act(self.bn(y))
        return self.proj(y)