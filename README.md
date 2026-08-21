<h1 align="center">YOLO-Pneumonia</h1>

<p align="center">English | <a href="README.zh-Hans.md">简体中文</a></p>

---

<p align="center">

A pediatric chest X-ray assisted diagnosis algorithm for pneumonia based on an enhanced `YOLOv8` framework. The method employs the `CLAHE` image enhancement algorithm to improve contrast and incorporates a multi-scale feature fusion module (`MSF`), a coordinate attention mechanism (`CA`), and a frequency adaptive module (`FAM`) into the `YOLOv8` backbone to strengthen the model's capability in perceiving lesions of varying scales.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10.6-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/torch-2.5.0-orange?logo=python" alt="Torch">
  <img src="https://img.shields.io/badge/ultralytics-8.4.3-blue?logo=python" alt="Ultralytics">
  <!-- <img src="https://img.shields.io/badge/jupyter-7.4.7-brightgreen?logo=python" alt="Jupyter Notebook"> -->
  <img src="https://img.shields.io/badge/License-MIT-orange" alt="License">
</p>

---

## Methodology

### CLAHE Image Enhancement

Prior to training, the original chest X-ray images are preprocessed using CLAHE (Contrast Limited Adaptive Histogram Equalization). This technique restricts the amplitude of contrast amplification to suppress noise enhancement, equalizes the histogram within localized regions, and effectively improves the contrast between pulmonary lesions and normal tissues.

### Multi-Scale Feature Fusion Module (MSF)

Cross-scale connections and feature fusion are introduced among different scale feature layers within the YOLOv8 backbone network. This design aggregates fine-grained details from shallow layers and high-level semantic information from deeper layers, thereby enhancing the model's ability to detect lesions of varying sizes.

### Coordinate Attention Mechanism (CA)

Positional information is embedded into channel attention by aggregating features along the height and width dimensions separately, generating direction-aware attention weights. This enables the model to focus more precisely on the regions where lesions are located.

### Frequency Adaptive Module (FAM)

Adaptive modulation of feature maps is performed in the frequency domain. The Fast Fourier Transform (FFT) is utilized to transform spatial features into the frequency domain, where learnable frequency masks are applied to enhance or suppress specific frequency components, thereby improving the model's capacity to model texture and edge information.

### Grad-CAM Visualization

Gradient-weighted Class Activation Mapping (Grad-CAM) is adopted to generate class activation heatmaps, which are overlaid onto the original X-ray images. This provides an intuitive visualization of the regions that the model focuses on during decision-making, facilitating the validation of the proposed modules' effectiveness.

---

## Usage Instructions

### Data Preparation

```bash
python data_split.py --data ./data/basedata --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

The dataset is randomly split into training, validation, and test sets according to the specified ratios, with CLAHE preprocessing applied simultaneously.

### Training

```bash
python main.py --train --name <experiment_name> --epochs <N>
```

Parameter descriptions:

- `--name`: Experiment name, which also corresponds to the YAML configuration file under model/yaml/

- `--epochs`: Number of training epochs

- `--batch`: Batch size (default: 32)

- `--imgsz`: Input image size (default: 256)

- `--optimizer`: Optimizer (default: SGD)

- `--lr0` / `--lrf`: Initial / final learning rate

- `--weight_decay`: Weight decay coefficient

- `--label_smoothing`: Label smoothing factor

### Evaluation

```bash
python main.py --eval --name <experiment_name>
```

Performance metrics including Accuracy, Precision, Recall, Specificity, F1-score, AUC, and mAP are computed on the test set.

### Grad-CAM Heatmap Visualization

```bash
cd cam-test && python run_cam.py
```

Upon entering the experiment name, the corresponding model weights are automatically loaded, and Grad-CAM heatmaps are generated on the test data.
