import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def add_text_overlay(image, true_label=None, prob=None):
    """在图像上叠加文本标注，对标 model_test.py 的 save_prediction_result。

    以原图尺寸为基准渲染，不撑宽、不变形。

    Args:
        image:      RGB uint8 array (H, W, 3)
        true_label: "NORMAL" / "PNEUMONIA"；None 跳过 ACTUAL
        prob:       肺炎概率 (0~1)；None 跳过 PREDICTED + 概率行

    Returns:
        RGB uint8 array，shape 与 image 一致
    """
    h, w = image.shape[:2]
    dpi = 100

    ref_h = 1000
    scale = h / ref_h
    fs_hl = max(14, int(28 * scale))
    fs_prob = max(11, int(22 * scale))

    fig = plt.figure(frameon=False, figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto')

    if true_label is not None:
        ax.text(0.02, 0.96, f"ACTUAL: {true_label}",
                transform=ax.transAxes, fontsize=fs_hl, fontweight='bold',
                color=(1, 0, 0), va='top')

    if prob is not None:
        pred_label = "PNEUMONIA" if prob >= 0.75 else "NORMAL"
        pred_prob = prob * 100 if prob >= 0.75 else (1 - prob) * 100
        correct = (pred_label == true_label) if true_label is not None else True
        pred_color = (0, 0, 1) if correct else (0, 1, 0)

        ax.text(0.02, 0.88, f"PREDICTED: {pred_label} ({pred_prob:.1f}%)",
                transform=ax.transAxes, fontsize=fs_hl, fontweight='bold',
                color=pred_color, va='top')

        ax.text(0.02, 0.80,
                f"NORMAL: {(1-prob)*100:.1f}% | PNEUMONIA: {prob*100:.1f}%",
                transform=ax.transAxes, fontsize=fs_prob,
                color=(1, 1, 1), va='top')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    result = plt.imread(buf)
    if result.shape[2] == 4:
        result = result[:, :, :3]
    if result.dtype == np.float32 or result.dtype == np.float64:
        result = (result * 255).astype(np.uint8)
    return result


def save_text_overlay(image_path, true_label, prob, output_path):
    """从文件读取 → 叠加文本 → 保存 PNG。"""
    image = mpimg.imread(image_path)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    result = add_text_overlay(image, true_label, prob)
    plt.imsave(output_path, result)
