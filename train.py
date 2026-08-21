"""
训练模块
包含自定义验证器（增加 Precision / Recall / F1 / AUC / mAP）、自定义 Trainer、以及训练入口
"""

import torch
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import pandas as pd
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.models.yolo.classify.train import ClassificationTrainer


class PRFClassificationValidator(ClassificationValidator):
    """
    在 YOLOv8 分类验证器基础上，额外计算:
    Precision / Recall / F1 / AUC / mAP
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_true = []   # 收集每个 batch 的真实标签
        self.y_prob = []   # 收集每个 batch 的预测概率
        self.extra_metrics = {}  # 额外指标，不写入 results.csv（避免 plotting 越界）

    def update_metrics(self, preds, batch):
        """
        每个 batch 调用一次，收集 preds 和 targets
        """
        super().update_metrics(preds, batch)

        probs = torch.softmax(preds, dim=1)
        targets = batch["cls"].view(-1)

        self.y_true.append(targets.cpu())
        self.y_prob.append(probs.detach().cpu())

    def get_stats(self):
        """
        在所有 batch 处理完后，计算完整指标
        """
        stats = super().get_stats()

        y_true = torch.cat(self.y_true).numpy()
        y_prob = torch.cat(self.y_prob).numpy()
        y_pred = y_prob.argmax(axis=1)

        extra = {}
        extra["metrics/precision"] = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        extra["metrics/recall"] = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        extra["metrics/f1"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # AUC & mAP: binary 用正类概率，multi-class 用 ovr 加权
        try:
            if y_prob.shape[1] == 2:
                extra["metrics/auc"] = roc_auc_score(y_true, y_prob[:, 1])
                extra["metrics/mAP"] = average_precision_score(
                    y_true, y_prob[:, 1]
                )
            else:
                extra["metrics/auc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
                extra["metrics/mAP"] = average_precision_score(
                    pd.get_dummies(y_true).values, y_prob, average="weighted"
                )
        except ValueError:
            extra["metrics/auc"] = 0.0
            extra["metrics/mAP"] = 0.0

        self.extra_metrics = extra

        # 注入额外指标到 stats → 写进 results.csv + TensorBoard
        for k, v in extra.items():
            stats[k] = v

        # fitness 键用于 best.pt 保存判断（validate 中 pop 该键）
        stats["fitness"] = extra["metrics/auc"]

        # top5-acc 对二分类无意义（恒为 1.0），移除以免破坏 results.png 绘图列数
        stats.pop("metrics/accuracy_top5", None)

        self.y_true.clear()
        self.y_prob.clear()

        return stats

    def print_results(self):
        """打印包含额外指标的评估结果"""
        super().print_results()
        if self.extra_metrics:
            em = self.extra_metrics
            print(
                f"  Precision:{em['metrics/precision']:>8.3g}  "
                f"Recall:{em['metrics/recall']:>8.3g}  "
                f"F1:{em['metrics/f1']:>8.3g}  "
                f"AUC:{em['metrics/auc']:>8.3g}  "
                f"mAP:{em['metrics/mAP']:>8.3g}"
            )


class CustomClassificationTrainer(ClassificationTrainer):
    """
    使用 PRFClassificationValidator 替换默认验证器
    早停基于 AUC 而非 top1_acc
    """

    @property
    def fitness(self):
        return self.metrics.get("metrics/auc", 0.0)

    @fitness.setter
    def fitness(self, value):
        pass  # 从 metrics 实时计算，忽略直接赋值

    def get_validator(self):
        return PRFClassificationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self.args,
        )


def run_training(
    data_dir: str,
    model_path: str,
    epochs: int = 50,
    patience: int = 0,
    batch: int = 32,
    imgsz: int = 256,
    device: int = 0,
    optimizer: str = "AdamW",
    weight_decay: float = 0.01,
    lr0: float = 0.001,
    lrf: float = 0.0001,
    warmup_epochs: int = 5,
    label_smoothing: float = 0.1,
    save_period: int = -1,
    cache: bool = False,
    resume: bool = False,
    amp: bool = False,
    pretrained: bool = False,
    plots: bool = True,
    name: str = "exp",
    workers: int = 0,
    seed: int = 0,
):
    from ultralytics import YOLO, settings

    settings["tensorboard"] = True
    import importlib
    from ultralytics.utils.callbacks import tensorboard as tb_cb
    importlib.reload(tb_cb)

    model = YOLO(model_path)

    model.train(
        data=data_dir,
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        device=device,
        optimizer=optimizer,
        weight_decay=weight_decay,
        lr0=lr0,
        lrf=lrf,
        warmup_epochs=warmup_epochs,
        label_smoothing=label_smoothing,
        save=True,
        save_period=save_period,
        cache=cache,
        amp=amp,
        pretrained=pretrained,
        resume=resume,
        name=name,
        workers=workers,
        plots=plots,
        trainer=CustomClassificationTrainer,
        seed=seed,
    )
