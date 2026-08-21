<h1 align="center">YOLO-Pneumonia</h1>

<p align="center"><a href="README.md">English</a> | 简体中文</p>

<p align="center">

基于改进 <code>YOLOv8</code> 的儿童胸部X光图像肺炎辅助诊断算法，采用 <code>CLAHE</code> 图像增强算法提升对比度，在 <code>YOLOv8</code> 基础上引入多尺度特征融合模块 <code>MSF</code>、坐标注意力机制 <code>CA</code> 和频率自适应模块 <code>FAM</code>，增强模型对多尺度病灶的感知能力。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10.6-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/torch-2.5.0-orange?logo=python" alt="Torch">
  <img src="https://img.shields.io/badge/ultralytics-8.4.3-blue?logo=python" alt="Ultralytics">
  <!-- <img src="https://img.shields.io/badge/jupyter-7.4.7-brightgreen?logo=python" alt="Jupyter Notebook"> -->
  <img src="https://img.shields.io/badge/License-MIT-orange" alt="License">
</p>

---

## 方法

### CLAHE 图像增强

训练前对原始胸部 X 光图像进行 CLAHE（Contrast Limited Adaptive Histogram Equalization）预处理，限制对比度放大幅度以抑制噪声放大，在局部区域内均衡直方图，增强肺部病灶与正常组织的对比度。

### 多尺度特征融合模块（MSF）

在 YOLOv8 骨干网络的不同尺度特征层之间引入跨尺度连接与融合，聚合浅层细节与深层语义信息，提升模型对不同大小病灶的感知能力。

### 坐标注意力机制（CA）

将位置信息编码到通道注意力中，通过沿高度和宽度方向分别聚合特征，生成方向感知的注意力权重，使模型更关注病灶所在的区域。

### 频率自适应模块（FAM）

在频域中对特征图进行自适应调制，利用 FFT 将空间特征变换到频域，学习可选的频率掩码以增强或抑制特定频率成分，提升模型对纹理和边缘信息的建模能力。

### Grad-CAM 可视化

采用 Grad-CAM（Gradient-weighted Class Activation Mapping）生成类别激活热力图，叠加到原始 X 光图像上，直观展示模型决策时关注的区域，辅助验证模块的有效性。

---

## 使用说明

### 数据准备

```bash
python data_split.py --data ./data/basedata --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

按指定比例随机划分数据集为训练集、验证集和测试集，并应用 CLAHE 预处理。

### 训练

```bash
python main.py --train --name <experiment_name> --epochs <N>
```

参数说明：

- `--name`：实验名称，同时对应 `model/yaml/` 下的 YAML 配置文件
- `--epochs`：训练轮数
- `--batch`：批大小（默认 32）
- `--imgsz`：输入图像尺寸（默认 256）
- `--optimizer`：优化器（默认 SGD）
- `--lr0` / `--lrf`：初始 / 最终学习率
- `--weight_decay`：权重衰减
- `--label_smoothing`：标签平滑系数

### 评估

```bash
python main.py --eval --name <experiment_name>
```

在测试集上计算 Accuracy、Precision、Recall、Specificity、F1-score、AUC、mAP 等指标。

### Grad-CAM 热图可视化

```bash
cd cam-test && python run_cam.py
```

输入实验名称，自动加载对应权重并在 testdata 上生成 Grad-CAM 热力图。
