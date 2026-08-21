"""
主控模块
配置训练/评估参数，调度 data_loader → train → eval
"""

import os
import gc
import shutil
import random
import argparse

os.environ["OMP_NUM_THREADS"] = "1"

YAML_DIR = "./model/yaml"
DEFAULT_NAME = "exp"
DEFAULT_YAML = "yolov8x-cls.yaml"


def resolve_yaml(name: str, model_yaml: str | None) -> str:
    if name and name != DEFAULT_NAME:
        return os.path.join(YAML_DIR, f"{name}.yaml")
    if model_yaml:
        return model_yaml
    return os.path.join(YAML_DIR, DEFAULT_YAML)


def parse_args():
    parser = argparse.ArgumentParser(
        description="X-ray Pneumonia Classification (YOLOv8x-cls)"
    )

    parser.add_argument("--data", default="./data/dataset/",
                        help="dataset 根目录（含 train/ val/ test/）")
    parser.add_argument("--model_yaml", default=None,
                        help="模型 yaml 配置文件路径（默认从 --name 推导）")

    parser.add_argument("--epochs", type=int, default=30,
                        help="总训练轮数")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停耐心值（基于 AUC，0=不早停）")
    parser.add_argument("--batch", type=int, default=32,
                        help="batch size")
    parser.add_argument("--imgsz", type=int, default=256,
                        help="输入图像尺寸")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU 设备号")

    # 优化器与学习率
    parser.add_argument("--optimizer", default="AdamW", ## SGD/AdamW
                        help="优化器")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--lr0", type=float, default=0.001,
                        help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.0001,
                        help="最终学习率")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="学习率预热轮数")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="标签平滑")

    # 其他配置
    parser.add_argument("--save_period", type=int, default=-1,
                        help="保存间隔（-1 不保存中间模型）")
    parser.add_argument("--workers", type=int, default=0,
                        help="数据加载线程数")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="加载预训练权重")
    parser.add_argument("--cache", action="store_true", default=False,
                        help="是否缓存数据集到内存")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="启用自动混合精度")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（0=自动从 1~999 中选一个）")
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help="实验名称（同时也是 yaml 文件名，不含扩展名）")

    # 训练指令配置
    parser.add_argument("--train", action="store_true", default=False,
                        help="执行训练")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="执行评估")

    return parser.parse_args()


def confirm(purpose: str, args) -> bool:
    print("=" * 55)
    print(f"  计划: {purpose}")
    if purpose == "训练":
        print(f"  实验名称 : {args.name}")
        print(f"  YAML     : {args.model_yaml}")
        print(f"  数据     : {args.data}")
        print(f"  epochs   : {args.epochs}")
        print(f"  imgsz    : {args.imgsz}")
        print(f"  batch    : {args.batch}")
        print(f"  优化器   : {args.optimizer}")
        print(f"  warmup   : {args.warmup_epochs}")
        print(f"  pretrained: {args.pretrained}")
    else:
        print(f"  实验名称 : {args.name}")
        print(f"  权重     : {os.path.join('./runs/classify', args.name, 'weights', 'best.pt')}")
        print(f"  数据     : {os.path.join(args.data, 'test')}")
    print("=" * 55)
    resp = input(f"是否执行{purpose}？(y/N): ").strip().lower()
    return resp in ("y", "yes", "执行训练", "执行评估")


def main():
    args = parse_args()

    args.model_yaml = resolve_yaml(args.name, args.model_yaml)

    runs_dir = os.path.join("./runs/classify", args.name)
    best_pt = os.path.join(runs_dir, "weights", "best.pt")

    if args.train:
        if args.seed == 0:
            args.seed = random.randint(1, 999)
        print(f"  随机种子: {args.seed}")

        # 崩溃恢复：检测到上次训练未完成（无 best.pt）则清理
        if os.path.exists(runs_dir) and not os.path.exists(best_pt):
            print(f"[清理] 检测到上次训练未完成，删除残留目录: {runs_dir}")
            shutil.rmtree(runs_dir)

        if not confirm("训练", args):
            print("  已取消")
            return

        print("=" * 55)
        print(f"  开始训练: {args.name}")
        print(f"  模型: {args.model_yaml}")
        print(f"  数据: {args.data}")
        print(f"  epochs: {args.epochs}, batch: {args.batch}, imgsz: {args.imgsz}")
        print("=" * 55)

        from train import run_training
        run_training(
            data_dir=args.data,
            model_path=args.model_yaml,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,
            label_smoothing=args.label_smoothing,
            save_period=args.save_period,
            cache=args.cache,
            amp=args.amp,
            pretrained=args.pretrained,
            workers=args.workers,
            name=args.name,
            seed=args.seed,
        )

        gc.collect()
        print(f"  训练完成，模型已保存至: {best_pt}")

    if args.eval:
        if not os.path.exists(best_pt):
            print(f"[警告] 未找到 best.pt，跳过评估: {best_pt}")
            return

        if not confirm("评估", args):
            print("  已取消")
            return

        print("=" * 55)
        print(f"  开始评估: {args.name}")
        print("=" * 55)

        from eval import evaluate_model
        from data_loader import get_class_names

        class_names = get_class_names(args.data)
        test_dir = os.path.join(args.data, "test")

        evaluate_model(
            test_folder=test_dir,
            model_path=best_pt,
            class_names=class_names,
        )

        gc.collect()


if __name__ == "__main__":
    main()
