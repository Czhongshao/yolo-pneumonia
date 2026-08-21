"""
数据加载模块
从 dataset/ 目录中读取 train / val / test 的路径与标签
"""

import os
import glob
from typing import List, Tuple, Optional

import pandas as pd


def find_image_classes(images_path: str) -> List[Tuple[str, int]]:
    """
    扫描 images_path 下的所有子目录（类别），返回 (类名, 文件数)

    参数
    ----
    images_path : str
        包含类别子目录的父路径

    返回值
    ------
    List[Tuple[str, int]]
        [(类名, 图片数量), ...]
    """
    result = []
    for item in os.listdir(images_path):
        item_path = os.path.join(images_path, item)
        if os.path.isdir(item_path):
            file_count = len([
                f for f in os.listdir(item_path)
                if os.path.isfile(os.path.join(item_path, f))
            ])
            result.append((item, file_count))
    return result


def df_from_image_folders(images_path: str, extension: Optional[str] = "jpeg") -> pd.DataFrame:
    """
    递归扫描 images_path 下所有指定扩展名的图片，构建 DataFrame

    参数
    ----
    images_path : str
        图片根目录（子目录名为类别标签）
    extension : str
        图片扩展名（默认 jpeg）

    返回值
    ------
    pd.DataFrame
        两列: 'path'（文件路径）, 'label'（类别标签）
    """
    label = []
    path = []
    image_files = glob.glob(
        os.path.join(images_path, "**", f"*.{extension.lower()}"),
        recursive=True
    )
    for file in image_files:
        dirpath = os.path.dirname(file)
        folder_name = os.path.basename(dirpath)
        label.append(folder_name)
        path.append(file)

    return pd.DataFrame({"path": path, "label": label})


def get_split_paths(dataset_root: str) -> dict:
    """
    获取 dataset 目录下 train / val / test 的路径

    参数
    ----
    dataset_root : str
        dataset 根目录（含 train/ val/ test/ 子目录）

    返回值
    ------
    dict
        {"train": str, "val": str, "test": str}
    """
    return {
        "train": os.path.join(dataset_root, "train"),
        "val": os.path.join(dataset_root, "val"),
        "test": os.path.join(dataset_root, "test"),
    }


def get_class_names(dataset_root: str) -> List[str]:
    """
    从 train 目录中获取类别名称列表

    参数
    ----
    dataset_root : str
        dataset 根目录

    返回值
    ------
    List[str]
        类别名称，排序后返回
    """
    train_dir = os.path.join(dataset_root, "train")
    if not os.path.isdir(train_dir):
        return []
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    return classes
