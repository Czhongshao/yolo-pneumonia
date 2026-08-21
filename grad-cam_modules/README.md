# Grad-CAM 可视化模块

对 YOLOv8x-cls（及加入 MSF/CA/FAMS 的改进模型）生成的 X 光胸片分类结果进行 Grad-CAM 热力图可视化。

---

## 参考项目

### 1. Grad-CAM 原理解读

论文：*[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)*

参考实现：

- [ramprs/grad-cam](https://github.com/ramprs/grad-cam.git) — 官方实现
- [priyavrat-misra/xrays-and-gradcam](https://github.com/priyavrat-misra/xrays-and-gradcam.git) — 胸片 X 光 + Grad-CAM 实践

### 2. 借鉴内容

| 来源 | 借鉴部分 | 本项目的处理 |
|------|---------|-------------|
| ramprs/grad-cam | Grad-CAM 核心算法（前向获取特征图 → 反向传播求梯度 → 通道加权平均 → ReLU） | 用 `torch.autograd.grad` 替代手工 backward hook，避免 Ultralytics 内部 autograd Function 的视图修改冲突 |
| ramprs/grad-cam | 目标层选择策略：取最后一个卷积特征图 | 通过 `_find_target_layer` 自动寻找 backbone 末尾的非 Classify 层 |
| xrays-and-gradcam | 医学影像上的 CAM 可视化流程 | 支持 N_/P_ 前缀文件名和子目录两种数据组织方式 |
| xrays-and-gradcam | 热力图与原图的叠加融合方式 | 采用 JET 色图 + `cv2.addWeighted`，alpha=0.4 |

---

## 实现原理

### 算法流程

#### 输入

一张 $256 \times 256$ 的 RGB 胸片图像，像素值归一化到 $[0, 1]$。

#### Step 1：前向传播获取特征图

图像送入 YOLOv8x-cls 模型做前向推理。在 backbone 最后一层（即 `model.8` 或 `model.12`，该层是分类头之前的最后一个卷积特征图输出层）注册 `forward_hook`，捕获该层的输出特征图。

定义该特征图为：

$$
A \in \mathbb{R}^{C \times H \times W}
$$

其中：
- $C$ = 通道数（baseline 模型为 1280，MSF 模型为 384）
- $H$ = 特征图高度（输入 $256 \times 256$ 时通常为 $8$）
- $W$ = 特征图宽度（输入 $256 \times 256$ 时通常为 $8$）

#### Step 2：获取目标类别得分

取分类层输出的 logits 中目标类别的得分。本项目关注肺炎（PNEUMONIA）类别（class $= 1$），定义目标类别得分为：

$$
y^{c}
$$

其中 $c$ 是目标类别索引（$c = 1$ 表示 PNEUMONIA），$y^{c}$ 是模型对该类的未归一化 logit 值。

#### Step 3：反向传播求梯度

对目标类别得分 $y^{c}$ 执行反向传播，计算其对特征图 $A$ 的梯度：

$$
G = \frac{\partial y^{c}}{\partial A} \in \mathbb{R}^{C \times H \times W}
$$

其中 $G^{k}_{ij}$ 表示第 $k$ 个通道在位置 $(i, j)$ 处的梯度值。

具体实现使用 `torch.autograd.grad(y_c, A)` 而非 `register_backward_hook`，原因见下文"与参考项目的差异"部分。

#### Step 4：通道加权平均（权重计算）

对每个通道 $k$ 的梯度图做全局平均池化，得到该通道的重要性权重：

$$
w_{k} = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} G^{k}_{ij}
$$

其中 $w_{k}$ 是第 $k$ 个通道对目标类别 $c$ 的贡献权重。该权重衡量了"通道 $k$ 对模型判断为肺炎的重要程度"。

#### Step 5：加权融合 + ReLU 激活

用权重 $w_{k}$ 对所有通道的特征图 $A^{k}$ 做加权求和，再通过 ReLU 去除负响应（只保留"对肺炎有正贡献"的区域）：

$$
L^{c} = \text{ReLU}\left( \sum_{k=1}^{C} w_{k} \cdot A^{k} \right) \in \mathbb{R}^{H \times W}
$$

其中 $L^{c}$ 是生成的原始 CAM（Class Activation Map），值越大表示该区域对模型判断为肺炎的贡献越大。

ReLU 的作用：只保留正向激活（$\sum w_k \cdot A^k > 0$ 的部分），因为负值区域对应"抑制肺炎预测"的特征，与可视化目标无关。

#### Step 6：上采样到原图尺寸

将 $H \times W$ 的 CAM 上采样到原图尺寸 $256 \times 256$，使用双线性插值（bilinear interpolation）：

$$
L^{c}_{\text{up}} = \text{Upsample}(L^{c}, \text{size}=(256, 256))
$$

#### Step 7：归一化到 [0, 1]

将上采样后的 CAM 线性缩放到 $[0, 1]$ 区间：

$$
L^{c}_{\text{norm}} = \frac{L^{c}_{\text{up}} - \min(L^{c}_{\text{up}})}{\max(L^{c}_{\text{up}}) - \min(L^{c}_{\text{up}}) + \epsilon}
$$

其中 $\epsilon = 10^{-8}$ 防止除零。

#### Step 8：热力图可视化

将归一化 CAM 映射为 JET 色图（蓝色 $\rightarrow$ 绿色 $\rightarrow$ 黄色 $\rightarrow$ 红色，红色表示高激活区域），再与原图以 $0.4:0.6$ 的比例叠加：

$$
\text{Overlay} = 0.4 \times \text{Heatmap} + 0.6 \times \text{Original}
$$

#### 完整流程图示

```
输入: 256×256×3 胸片
  │
  ▼
YOLOv8x-cls 前向传播
  │
  ├── forward_hook 捕获 backbone 最后特征图 A (C×H×W)
  │
  ▼
取出 PNEUMONIA 类 logit y^c
  │
  ▼
torch.autograd.grad(y^c, A) → 梯度 G (C×H×W)
  │
  ▼
通道加权平均: w_k = mean(G_k) → 权重向量 w (C,)
  │
  ▼
加权融合 + ReLU: L^c = ReLU(Σ w_k · A^k) → CAM (H×W)
  │
  ▼
上采样 → 256×256 → 归一化 [0,1]
  │
  ▼
JET 色图映射 → 与原图 0.4:0.6 融合 → 输出 overlay
```

### 关键代码路径

```
cam-test/gradcam.py
├── GradCAM.__init__()       — 加载模型、自动选择目标层、注册 forward_hook
├── GradCAM.__call__()       — 前向 → 求导 → CAM 计算 → 上采样 → 归一化
├── load_image()              — 读图 → RGB → resize → [0,1] tensor
└── overlay_heatmap()         — CAM 数组 → JET 色图 → 与原图叠加

cam-test/run_cam.py
├── get_test_samples()        — 支持子目录和扁平目录两种数据组织
├── process_model()           — 对单个模型执行 Grad-CAM 生成
└── main()                    — 依次处理 MSF+CA+FAMS 和 Baseline 两个模型

cam-test/outputs/{model_name}/
├── {stem}_orig.png           — 原始图像
├── {stem}_overlay.png        — CAM 热力图叠加结果
└── {stem}_cam.npy            — 原始 CAM 数值
```

### 与参考项目的差异

1. **不使用注册 backward_hook**：ramprs/grad-cam 使用 `register_backward_hook`，但 Ultralytics 内部使用了自定义 autograd Function，hook 返回的梯度是 view，无法被 in-place 修改。改用 `torch.autograd.grad` 直接计算梯度。

2. **自动 target layer 选择**：通过逆向遍历 `model.model.model`，跳过 `Classify` 层，选择 backbone 最后一个特征图输出层。

3. **模型兼容**：MSF+CA+FAMS 模型的 checkpoint 保存时不包含 MSF 的 `proj` 层（旧版 Ultralytics），载入后自动注入 `nn.Identity()` 保证前向兼容。

---

## 使用方法

```bash
cd cam-test

# 默认：处理两个模型，读 testdata/
python run_cam.py

# 或通过 Python API 单图调用
python -c "
from gradcam import GradCAM, load_image, overlay_heatmap
import cv2

cam = GradCAM('path/to/model.pt')
tensor, img, _ = load_image('testdata/P_1.jpeg')
cam_map, logits = cam(tensor, target_class=1)
overlay, heatmap = overlay_heatmap(img, cam_map)
cv2.imwrite('result.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print('Prediction:', logits)
"
```

### 目录结构支持

```
# 子目录方式（自动识别）
testdata/
├── NORMAL/
│   └── *.jpeg
└── PNEUMONIA/
    └── *.jpeg

# 扁平方式（N_/P_ 前缀识别）
testdata/
├── N_1.jpeg
├── P_1.jpeg
└── ...
```
