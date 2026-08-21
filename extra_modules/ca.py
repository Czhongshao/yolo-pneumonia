import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class CA(nn.Module):
    """
    Coordinate Attention (YOLO-compatible)
    Does NOT change channel number
    """

    def __init__(self, c1, reduction=32):
        super().__init__()

        self.c1 = c1
        c_ = max(8, c1 // reduction)

        self.conv1 = Conv(c1, c_, k=1, s=1, act=True)
        self.conv_h = Conv(c_, c1, k=1, s=1, act=False)
        self.conv_w = Conv(c_, c1, k=1, s=1, act=False)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        x_h = self.pool_h(x)                      # (n, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (n, c, w, 1)

        y = torch.cat([x_h, x_w], dim=2)          # (n, c, h+w, 1)
        y = self.conv1(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out
