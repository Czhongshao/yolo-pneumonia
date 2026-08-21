"""
将原始 chest_xray 数据汇总到 basedata
- 合并 chest_xray/train/ 和 chest_xray/test/
- 按 NORMAL / PNEUMONIA 分类（不再保留原始 split）
- MD5 全局去重
"""

import os
import shutil
import hashlib
import argparse
from collections import defaultdict

_BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(_BASE, "chest_xray")
DEST = os.path.join(_BASE, "basedata")

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="汇总 chest_xray → basedata（MD5 去重）")
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--dest", default=DEST)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", default=False,
                        help="强制重建（默认基于已有数据跳过）")
    return parser.parse_args()


def md5_of_file(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_all_files(data_dir: str) -> list:
    items = []
    VALID_EXTS = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}
    for split in sorted(os.listdir(data_dir)):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VALID_EXTS:
                    continue
                items.append((cls, os.path.join(cls_dir, fname)))
    return items


def main():
    args = parse_args()

    dest = os.path.normpath(args.dest)
    if os.path.exists(dest) and not args.force:
        print(f"[跳过] {dest} 已存在，使用 --force 强制重建")
        return

    print(f"[1/3] 扫描 {args.source} ...")
    all_files = scan_all_files(args.source)
    print(f"    共扫描到 {len(all_files)} 张图片\n")

    # MD5 全局去重（合并 train+test，只按类别区分）
    md5_index = {}
    kept = []
    removed = []

    for cls, fpath in all_files:
        md5 = md5_of_file(fpath)
        if md5 in md5_index:
            removed.append((cls, fpath))
        else:
            md5_index[md5] = (cls, fpath)
            kept.append((cls, fpath))

    # 清理并重建 basedata（只有 class 目录，无 split 层级）
    if os.path.exists(dest):
        print(f"[2/3] 清理旧目录 {dest} ...")
        shutil.rmtree(dest)

    print(f"[3/3] 复制去重后的文件到 {dest} ...")
    cls_stats = defaultdict(int)
    for cls, src_path in kept:
        rel = os.path.join(cls, os.path.basename(src_path))
        dst_path = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        cls_stats[cls] += 1

    print(f"\n{'='*50}")
    print(f"  汇总完成: {dest}")
    print(f"{'='*50}")
    for cls, cnt in sorted(cls_stats.items()):
        print(f"  {cls}: {cnt} 张")
    print(f"  总计: {sum(cls_stats.values())} 张")
    print(f"  去重移除: {len(removed)} 张")
    print(f"  随机种子: {args.seed}")


if __name__ == "__main__":
    main()
