"""
评估模块
计算分类指标: Accuracy / Precision / Recall / F1 / AUC / mAP / Params / FLOPs
"""

import os
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_ap_per_class(y_true, y_score_target, pos_class):
    """COCO 101-point 插值计算单类 AP（全体样本排序，无硬阈值过滤）"""
    n_pos = int((y_true == pos_class).sum())
    if n_pos == 0:
        return 0.0

    idx = np.argsort(-y_score_target)
    tp = (y_true[idx] == pos_class).astype(float)
    fp = (y_true[idx] != pos_class).astype(float)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    rec = tp_cum / n_pos
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask_r = rec >= t
        if mask_r.any():
            ap += np.max(prec[mask_r]) / 101
    return ap


def evaluate_model(
    test_folder: str,
    model_path: str,
    class_names: list = None,
    verbose: bool = True,
):
    """
    对测试集运行完整评估

    参数
    ----
    test_folder : str
        测试集目录（含 NORMAL/ PNEUMONIA 等子目录）
    model_path : str
        训练好的模型权重路径（best.pt）
    class_names : list
        类别名称列表，默认 ['NORMAL', 'PNEUMONIA']
    verbose : bool
        是否打印详细结果

    返回值
    ------
    dict
        包含所有指标的字典
    """
    from ultralytics import YOLO

    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    if not Path(model_path).exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        return None

    model = YOLO(model_path)

    # --- 收集预测结果 ---
    y_true, y_prob_pneu = [], []
    for cls_name, cls_label in [(class_names[0], 0), (class_names[1], 1)]:
        cls_path = os.path.join(test_folder, cls_name)
        if not os.path.isdir(cls_path):
            continue
        results = model(cls_path, verbose=False)
        for r in results:
            y_true.append(cls_label)
            # 二分类: PNEUMONIA 为 class 1, 取 probs[1]
            y_prob_pneu.append(r.probs.data[1].item())

    y_true = np.array(y_true)
    y_prob_pneu = np.array(y_prob_pneu)
    y_pred = (y_prob_pneu >= 0.75).astype(int) # 分类阈值

    # --- 标准分类指标 ---
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["F1-score"] = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["AUC"] = roc_auc_score(y_true, y_prob_pneu)
    metrics["mAP"] = average_precision_score(y_true, y_prob_pneu)

    # --- Detection-style mAP (COCO 101-pt) ---
    ap_pneu = compute_ap_per_class(y_true, y_prob_pneu, pos_class=1)
    ap_norm = compute_ap_per_class(y_true, 1.0 - y_prob_pneu, pos_class=0)
    metrics["mAP_det"] = (ap_pneu + ap_norm) / 2.0

    # --- 模型参数量 & FLOPs ---
    try:
        # 通过 ultralytics 内部方法获取
        from ultralytics.utils.torch_utils import get_flops, get_num_params
        params = get_num_params(model.model)
        # 对分类模型估算 FLOPs: 输入 3x256x256
        flops = get_flops(model.model, imgsz=256)
        metrics["Params/M"] = params / 1e6
        metrics["FLOPS/G"] = flops
    except Exception:
        # 回退: 通过 thop 库直接计算
        try:
            import torch
            from thop import profile
            from copy import deepcopy
            dummy = torch.randn(1, 3, 256, 256).to(next(model.model.parameters()).device)
            flops, params = profile(deepcopy(model.model), inputs=(dummy,), verbose=False)
            metrics["Params/M"] = params / 1e6
            metrics["FLOPS/G"] = flops / 1e9
        except Exception:
            metrics["Params/M"] = 0.0
            metrics["FLOPS/G"] = 0.0

    # --- 打印结果 ---
    if verbose:
        print("=" * 55)
        print("  评估结果")
        print("=" * 55)
        for key in ["Accuracy", "Precision", "Recall", "Specificity", "F1-score", "AUC", "mAP"]:
            print(f"  {key:<12}: {metrics[key]:.4f}")
        print(f"  mAP_det     : {metrics['mAP_det']:.4f}  (COCO 101-pt)")
        print(f"  Params/M    : {metrics['Params/M']:.2f}")
        print(f"  FLOPS/G     : {metrics['FLOPS/G']:.2f}")
        print("=" * 25)
        print("min:", y_prob_pneu.min())
        print("max:", y_prob_pneu.max())
        print("mean:", y_prob_pneu.mean())
        print(np.percentile(y_prob_pneu, [5,10,25,50,75,90,95]))
        print("=" * 55)

    return metrics
