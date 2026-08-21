# Data Documentation

<p align="center">English | <a href="DataREADME.zh-Hans.md">简体中文</a></p>

## I. Data Source

<!-- ~~The dataset is sourced from the Kaggle public dataset: [chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).~~ -->

The dataset is derived from the article: [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5). Retrieved from [ChestXRay2017](https://data.mendeley.com/datasets/rscbjbr9sj/2).

The raw data is located at `./data/chest_xray/`, which has been pre-split into train / test:

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

## II. Data Aggregation & Deduplication — organize_basedata.py

`organize_basedata.py` **merges** the `chest_xray/train/` and `chest_xray/test/` directories, aggregates the images by category (NORMAL / PNEUMONIA) into `basedata/`, and removes exact duplicate images based on MD5 hashing.

Execute the following command in the project root directory:

```bash
python data/organize_basedata.py
```

### Parameter Descriptions

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--source` | `./data/chest_xray/` | Raw data directory |
| `--dest` | `./data/basedata/` | Output directory (containing only NORMAL / PNEUMONIA subfolders) |
| `--seed` | `42` | Random seed |

### Example Execution Output

```text
Unique images retained: 5856
Duplicate images skipped: 7
```

### Output Structure

```txt
basedata
  ├─NORMAL
  └─PNEUMONIA
```

### Important Notes

- The original chest_xray/ data directory will not be modified

- Aggregated files are output to the basedata/ directory

- If basedata/ already exists, it will be overwritten and rebuilt

---

## III. Dataset Splitting — data_split.py

`data_split.py` performs stratified sampling on the images in `basedata/` at a ratio of 8:1:1 to generate train / val / test splits.

Execute the following command in the project root directory:

Execute the following command in the project root directory:

```bash
python data/data_split.py
```


### Parameter Descriptions

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--source` | `./data/basedata/` | Deduplicated data directory (must contain NORMAL / PNEUMONIA subfolders) |
| `--dest` | `./data/dataset/` | Output directory (generates train / val / test subfolders) |
| `--seed` | `42` | Random seed for reproducible results |

### Customization Example

```bash
python data/data_split.py --seed 123
```

### Splitting Logic

1. Scans all images under each category in `--source/`
2. Performs stratified sampling independently for each category, randomly splitting into train / val / test at an 8:1:1 ratio
3. Generates the following output directory structure:

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

### Example Execution Output

```txt
  --- train ---
    NORMAL: 1215
    PNEUMONIA: 3495
    Total: 4710

  --- val ---
    NORMAL: 134
    PNEUMONIA: 388
    Total: 522

  --- test ---
    NORMAL: 234
    PNEUMONIA: 390
    Total: 624
```

## IV. Complete Workflow Overview

```bash
# 1. Aggregate and deduplicate
python data/organize_basedata.py

# 2. Split at 8:1:1
python data/data_split.py
```

## V. Accessing Data from Other Directories (Symbolic Links)

If you need to access the root directory's `data/` from other directories (e.g., `notebook/`) using the path `./data/...`, you may use symbolic links without moving or copying files.

### Example: Accessing data from the notebook directory

```bash
# Create a symbolic link in the notebook directory pointing to the root data/
ln -s /home/shao/repo/xray-repo/data /home/shao/repo/xray-repo/notebook/data
```

Subsequently, the notebook code can correctly access the root data using `./data/basedata/`:

```python
base_dataset = './data/basedata/'      # → /home/shao/repo/xray-repo/data/basedata/
target_dataset = './data/dataset/'     # → /home/shao/repo/xray-repo/data/dataset/
corrupted_dataset = './data/corrupted_images/'
```

### General Command Syntax

```bash
# ln -s <actual_data_path> <link_name_in_target_directory>
ln -s /home/shao/repo/xray-repo/data ./data
```
