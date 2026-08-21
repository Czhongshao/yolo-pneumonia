import sys
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path

##  环境初始化：确保本地 ultralytics 仓库优先于 site-packages

_repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))  # 定位 repo 根目录
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

for _mod in list(sys.modules.keys()):
    if _mod.startswith('ultralytics'):
        del sys.modules[_mod]  # 清除缓存，强制从本地 reload

import ultralytics.nn.modules.fams
import ultralytics.nn.modules.msf
import ultralytics.nn.modules.ca
from ultralytics import YOLO


def _find_target_layer(model):
    for i in range(len(model) - 1, -1, -1):  # 从后往前找第一个非 Classify 层
        if type(model[i]).__name__ not in ("Classify",):
            return i
    return len(model) - 2


class GradCAM:
    """GradCAM 可视化，支持 CA / MSF / FAMS 自定义模块。"""

    def __init__(self, model_path: str):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        raw = self.model.model
        raw = raw.float().to(self.device)
        raw.eval()
        self.model.model = raw

        self.target_idx = _find_target_layer(raw.model)  # 定位目标卷积层
        self.activations = None
        raw.model[self.target_idx].register_forward_hook(  # 注册前向钩子捕获激活
            lambda m, i, o: setattr(self, "activations", o)
        )

    def forward_only(self, img_tensor):
        """仅前向推理返回 logits，无梯度，用于预处理判断。"""
        x = img_tensor.to(self.device).float()
        with torch.no_grad():
            logits = self.model.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.detach().cpu().numpy()

    def __call__(self, img_tensor: torch.Tensor, target_class: int = 1):
        """生成 GradCAM 热力图，返回 (cam_map, logits)。"""
        self.model.model.zero_grad()
        x = img_tensor.to(self.device).float()
        x.requires_grad_(True)

        logits = self.model.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        score = logits[:, target_class].sum()
        grads = torch.autograd.grad(score, self.activations)[0]  # 目标类梯度

        weights = grads.mean(dim=(2, 3), keepdim=True)  # 全局平均池化得权重
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # 线性加权
        cam = F.relu(cam)  # 只保留正响应
        cam = F.interpolate(cam, size=(img_tensor.shape[2], img_tensor.shape[3]),
                            mode="bilinear", align_corners=False)  # 上采样到原图尺寸
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # 归一化到 [0, 1]
        return cam.squeeze().detach().cpu().numpy(), logits.detach().cpu().numpy()


def load_image(img_path: str, img_size: int = 256):
    """读取 → RGB → 缩放 → 归一化 → 4D 张量 (1, C, H, W)。"""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), img_rgb, (h, w)


def overlay_heatmap(original_img: np.ndarray, cam: np.ndarray, alpha: float = 0.4):
    """CAM 热力图叠加到原图，alpha 控制透明度。"""
    h, w = original_img.shape[:2]
    cam = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return overlay, heatmap



