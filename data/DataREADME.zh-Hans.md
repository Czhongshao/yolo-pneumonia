# 数据说明文档

<p align="center"><a href="DataREADME.md">English</a> | 简体中文</p>

## 一、数据来源

<!-- ~~数据来源于 Kaggle 平台公开数据集：[chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).~~ -->

数据集源自文章：[Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5). 取自 [ChestXRay2017](https://data.mendeley.com/datasets/rscbjbr9sj/2).

原始数据位于 `./data/chest_xray/`，已预分割为 train / test：

```txt
chest_xray
  ├─train
  │   ├─NORMAL
  │   └─PNEUMONIA
  └─test
      ├─NORMAL
      └─PNEUMONIA
```

---

## 二、数据汇总 & 去重 — organize_basedata.py

`organize_basedata.py` 将 `chest_xray/train/` 和 `chest_xray/test/` **合并**，并按类别（NORMAL / PNEUMONIA）汇总到 `basedata/`，同时基于 MD5 去除完全重复的影像。

在项目根目录执行：

```bash
python data/organize_basedata.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | `./data/chest_xray/` | 原始数据目录 |
| `--dest` | `./data/basedata/` | 输出目录（仅含 NORMAL / PNEUMONIA 两类文件夹） |
| `--seed` | `42` | 随机种子 |

### 执行结果示例

```text
保留的唯一图片: 5824 张
跳过的重复图片: 32 张
```

### 输出结构

```txt
basedata
  ├─NORMAL
  └─PNEUMONIA
```

### 注意事项

- `chest_xray/` 为原始数据，**不会被修改**
- 汇总后的文件输出到 `basedata/`
- 如果 `basedata/` 已存在，会被覆盖重建

---

## 三、数据集划分 — data_split.py

`data_split.py` 将 `basedata/` 中的图片按 **8:1:1** 分层抽样为 train / val / test。

在项目根目录执行：

```bash
python data/data_split.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | `./data/basedata/` | 已去重的数据目录（需含 NORMAL / PNEUMONIA） |
| `--dest` | `./data/dataset/` | 输出目录（生成 train / val / test） |
| `--seed` | `42` | 随机种子，保证可复现 |

### 自定义示例

```bash
python data/data_split.py --seed 123
```

### 划分逻辑

1. 扫描 `--source/` 下每个类别的所有图片
2. 对每个类别独立进行 **分层抽样**，按 8:1:1 随机分为 train / val / test
3. 输出目录结构如下：

```txt
dataset
  ├─train
  │   ├─NORMAL
  │   └─PNEUMONIA
  ├─val
  │   ├─NORMAL
  │   └─PNEUMONIA
  └─test
      ├─NORMAL
      └─PNEUMONIA
```

### 执行结果示例

```text
  --- train ---
    NORMAL: 1265
    PNEUMONIA: 3397
    合计: 4662

  --- val ---
    NORMAL: 157
    PNEUMONIA: 424
    合计: 581

  --- test ---
    NORMAL: 157
    PNEUMONIA: 424
    合计: 581
```

---

## 四、全流程速览

```bash
# 1. 汇总去重
python data/organize_basedata.py

# 2. 分割 8:1:1
python data/data_split.py
```

---

## 五、在其他目录中访问数据（符号链接）

如果需要在其他目录（如 `notebook/`）中通过 `./data/...` 的路径访问根目录的 `data/`，
可以使用 **符号链接（symbolic link）**，无需移动或复制文件。

### 示例：在 notebook 目录中访问 data

```bash
# 在 notebook 目录下创建符号链接，指向根目录的 data/
ln -s /home/shao/repo/xray-repo/data /home/shao/repo/xray-repo/notebook/data
```

之后，notebook 代码中的 `./data/basedata/` 即可正确访问根目录的数据：

```python
base_dataset = './data/basedata/'      # → /home/shao/repo/xray-repo/data/basedata/
target_dataset = './data/dataset/'     # → /home/shao/repo/xray-repo/data/dataset/
corrupted_dataset = './data/corrupted_images/'
```

### 通用指令

```bash
# ln -s <实际数据路径> <需要访问的目录下的链接名>
ln -s /home/shao/repo/xray-repo/data ./data
```

### 特点

- 不占用额外磁盘空间
- 删除链接不影响原始数据
- Python 的 `open()`、`os.listdir()`、`shutil` 等均可透明读写
- 在 Jupyter Notebook 中同样有效
