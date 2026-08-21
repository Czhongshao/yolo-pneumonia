import os
import shutil
import hashlib
from collections import defaultdict


# 脚本所在目录的绝对路径
_BASE = os.path.dirname(os.path.abspath(__file__))

# 源数据（只读，不做任何修改）
SOURCE = os.path.join(_BASE, "chest_xray")
# 目标数据（去重后的唯一图片）
DEST = os.path.join(_BASE, "basedata")


def md5_of_file(filepath: str) -> str:
    """计算文件的 MD5 哈希值"""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_all_files(data_dir: str) -> list:
    """
    递归扫描 data_dir 下所有图片文件
    返回 [(split, class, filepath), ...]
    """
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
                fpath = os.path.join(cls_dir, fname)
                items.append((split, cls, fpath))
    return items


def main():
    print(f"[1/2] 扫描 {SOURCE} ...")
    all_files = scan_all_files(SOURCE)
    print(f"    共扫描到 {len(all_files)} 张图片\n")

    # 计算 MD5，全局去重
    md5_index = {}          # md5 → (split, cls, filepath)  首次出现的文件
    kept = []               # 保留的 (split, cls, src_path)
    removed = []            # 跳过的 (split, cls, src_path)

    for split, cls, fpath in all_files:
        md5 = md5_of_file(fpath)
        if md5 in md5_index:
            removed.append((split, cls, fpath))
        else:
            md5_index[md5] = (split, cls, fpath)
            kept.append((split, cls, fpath))

    # 复制保留的文件到 DEST
    print(f"[2/2] 复制去重后的文件到 {DEST} ...")
    copy_count = 0
    for split, cls, src_path in kept:
        rel = os.path.join(split, cls, os.path.basename(src_path))
        dst_path = os.path.join(DEST, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copy_count += 1

    # 打印汇总
    print(f"\n{'='*55}")
    print(f"  去重完成")
    print(f"{'='*55}")
    print(f"  保留的唯一图片: {copy_count} 张")
    print(f"  跳过的重复图片: {len(removed)} 张")
    print(f"  MD5 唯一编码数: {len(md5_index)} 个")

    if removed:
        print(f"\n{'='*55}")
        print(f"  以下重复图片已被跳过（未复制到 {DEST}）：")
        print(f"{'='*55}")
        for split, cls, fpath in removed:
            print(f"  [跳过] {split}/{cls}/ {os.path.basename(fpath)}")


if __name__ == "__main__":
    main()
