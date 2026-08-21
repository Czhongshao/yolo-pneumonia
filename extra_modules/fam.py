import torch
import torch.nn as nn
import torch.nn.functional as F


class FAM(nn.Module):
    """
    FAM-Lite (YOLO-stable)
    Frequency-Aware Module with Octave-style separation (SAFE VERSION)
    """

    def __init__(self, c1, c2, k=3, alpha=0.25):
        super().__init__()

        assert 0 < alpha < 1, "For YOLO, alpha should be small (e.g. 0.125~0.3)"

        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha

        c_low = int(c1 * alpha)
        c_high = c1 - c_low

        # 高频 / 低频分支
        self.conv_h = nn.Sequential(
            nn.Conv2d(c1, c_high, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_high),
            nn.SiLU()
        )

        self.conv_l = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(c1, c_low, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_low),
            nn.SiLU()
        )

        # 融合
        self.fuse = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

        # residual scale（防梯度爆炸）
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x

        x_h = self.conv_h(x)
        x_l = self.conv_l(x)
        x_l = F.interpolate(x_l, size=x_h.shape[2:], mode='nearest')

        out = torch.cat([x_h, x_l], dim=1)
        out = self.fuse(out)

        # residual + scale
        if identity.shape == out.shape:
            out = identity + self.gamma * out

        return out
