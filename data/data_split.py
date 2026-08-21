import os
import shutil
import random
import argparse
from collections import defaultdict

import cv2
import numpy as np


# 脚本所在目录的绝对路径
_BASE = os.path.dirname(os.path.abspath(__file__))

# 全局随机种子
SEED = 123


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="将数据集划分为 train / val / test")
    parser.add_argument("--source", default=os.path.join(_BASE, "basedata"))  # 原始数据目录
    parser.add_argument("--dest", default=os.path.join(_BASE, "dataset"))    # 输出目录
    parser.add_argument("--val_ratio", type=float, default=0.1)    # 验证集比例
    parser.add_argument("--seed", type=int, default=SEED)          # 随机种子
    parser.add_argument("--clahe", action="store_true", default=False,
                        help="应用 CLAHE 对比度限制自适应直方图均衡化增强")
    parser.add_argument("--clip_limit", type=float, default=2.0,
                        help="CLAHE 对比度限制阈值（默认 2.0）")
    return parser.parse_args()


def collect_files(data_dir: str) -> dict:
    """
    扫描 data_dir 下每个子目录（类别），收集所有文件路径
    返回 {类名: [文件路径列表]}
    """
    classes = sorted(os.listdir(data_dir))
    cls_files = {}
    for c in classes:
        cdir = os.path.join(data_dir, c)
        if not os.path.isdir(cdir):
            continue  # 跳过非目录项
        files = [
            os.path.join(cdir, f)
            for f in os.listdir(cdir)
            if os.path.isfile(os.path.join(cdir, f))
        ]
        if files:
            cls_files[c] = files
    return cls_files


def split_class_files(files: list, val_ratio: float, seed: int):
    """
    对某个类别的文件列表按比例 shuffle 切分
    返回 (train_list, val_list)
    """
    random.seed(seed)
    shuffled = sorted(files)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))  # 至少留 1 张给 val
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def apply_clahe(src_path: str, dst_path: str, clip_limit: float = 2.0,
                tile_size: tuple = (8, 8)):
    """
    读取原图 → CLAHE 增强 → 保存为 3 通道 BGR 图像
    """
    img = cv2.imread(src_path)
    if img is None:
        print(f"[警告] 无法读取: {src_path}，跳过 CLAHE")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(dst_path, enhanced_bgr)


def main():
    args = parse_args()
    random.seed(args.seed)

    source = os.path.normpath(args.source)  # 源路径标准化
    dest = os.path.normpath(args.dest)      # 目标路径标准化

    train_dir = os.path.join(source, "train")
    test_dir = os.path.join(source, "test")

    # 扫描训练集
    print(f"[1/3] 扫描 {train_dir} ...")
    train_files = collect_files(train_dir)
    print(f"\t\t训练集类别: {list(train_files.keys())}")
    for c, f in train_files.items():
        print(f"\t\t{c}: {len(f)} 张")

    # 扫描测试集
    print(f"[2/3] 扫描 {test_dir} ...")
    test_files = collect_files(test_dir)
    for c, f in test_files.items():
        print(f"\t\t{c}: {len(f)} 张")

    # 按比例分割 train → train + val
    print(f"\n[3/3] 按 val_ratio={args.val_ratio} 分层抽样 ...")
    copy_tasks = []  # (src, dst) 待复制列表
    stats = defaultdict(lambda: defaultdict(int))  # 统计各 split 各 class 数量

    for cls, files in train_files.items():
        # 用 种子+类名哈希 保证每类的 shuffle 不同但可复现
        train_sub, val_sub = split_class_files(files, args.val_ratio, args.seed + hash(cls) % 10000)
        for f in train_sub:
            rel_path = os.path.join("train", cls, os.path.basename(f))
            copy_tasks.append((f, os.path.join(dest, rel_path)))
            stats["train"][cls] += 1
        for f in val_sub:
            rel_path = os.path.join("val", cls, os.path.basename(f))
            copy_tasks.append((f, os.path.join(dest, rel_path)))
            stats["val"][cls] += 1

    # 测试集原样复制
    for cls, files in test_files.items():
        for f in files:
            rel_path = os.path.join("test", cls, os.path.basename(f))
            copy_tasks.append((f, os.path.join(dest, rel_path)))
            stats["test"][cls] += 1

    # 清理目标目录（如果存在），避免多次执行叠加文件
    if os.path.exists(dest):
        shutil.rmtree(dest)

    # 顺序复制（或 CLAHE 增强后写入），避免并发冲突
    clahe_count = 0
    for src, dst in copy_tasks:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if args.clahe:
            apply_clahe(src, dst, args.clip_limit)
            clahe_count += 1
        else:
            shutil.copy2(src, dst)

    if args.clahe:
        print(f"\nCLAHE 增强已应用，共处理 {clahe_count} 张图像 (clip_limit={args.clip_limit})")

    # 打印汇总
    print(f"\n{'='*50}")
    print(f"输出目录: {dest}")
    print(f"{'='*50}")
    for split_name in ["train", "val", "test"]:
        print(f"\n--- {split_name} ---")
        for cls, cnt in stats[split_name].items():
            print(f"  {cls}: {cnt}")
        total = sum(stats[split_name].values())
        print(f"  合计: {total}")
    print(f"\n随机种子: {args.seed}")


if __name__ == "__main__":
    main()
