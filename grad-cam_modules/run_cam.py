import os
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from gradcam import GradCAM, load_image, overlay_heatmap
from text_overlay import add_text_overlay

##  路径配置

base_dir = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(base_dir, "results")
os.makedirs(results, exist_ok=True)

model_name = str(input("Model Name: ")).strip()
model_result = os.path.join(results, model_name)
os.makedirs(model_result, exist_ok=True)

model_dir = os.path.join(base_dir, "..", "runs", "classify", model_name, "weights", "best.pt")  # 权重路径
model_path = Path(model_dir)
test_dir = os.path.join(base_dir, "testdata")  # 扁平结构，P-xxx / N-xxx 命名

IMG_SIZE = 256   # 模型输入尺寸

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_name_mapping(test_dir: str) -> dict:
    """读取新名→原名映射，无对应文件则回退显示新名。"""
    mappath = os.path.join(test_dir, "_mapping.json")
    if os.path.isfile(mappath):
        with open(mappath) as f:
            return json.load(f)
    return {}


def fix_msf_model(cam_extractor):
    """为缺少 proj 属性的 MSF 模块补充 Identity，兼容 GradCAM 钩子。"""
    model = cam_extractor.model.model.model
    for name, module in model.named_modules():
        if type(module).__name__ == "MSF" and not hasattr(module, "proj"):
            module.proj = nn.Identity()


def get_test_samples(test_dir: str):
    """从扁平目录收集所有 P-xxx（阳性）与 N-xxx（阴性）图像。"""
    images = []
    all_files = sorted(
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    )
    for f in all_files:
        if f.upper().startswith("N-"):
            images.append((os.path.join(test_dir, f), "NORMAL"))
        elif f.upper().startswith("P-"):
            images.append((os.path.join(test_dir, f), "PNEUMONIA"))
    return images


def process_model(model_path: str, model_name: str, fix_msf: bool = False):
    """加载模型 → 前向推理 → GradCAM → 保存 → 打印日志。"""
    mapping = load_name_mapping(test_dir)

    print(f"\n{'='*60}")
    print(f"  Model Path:  {model_path}")
    print(f"  Model Name:  {model_name}")
    print(f"  Device:      {device}")
    print(f"{'='*60}")

    cam_extractor = GradCAM(model_path)
    if fix_msf:
        print("  [Fix] Adding identity proj to MSF...")
        fix_msf_model(cam_extractor)

    test_images = get_test_samples(test_dir)
    print(f"  Image Source: {test_dir}")
    print(f"  Total Images: {len(test_images)}")
    print(f"  Save Dir:     {Path(model_result).resolve()}")
    print()

    out_dir = Path(model_result)

    ##  逐图处理

    records = []
    pbar = tqdm(test_images, desc=f"Processing {len(test_images)} images", unit="img")
    for img_path, true_class in pbar:
        fname = Path(img_path).name
        orig_name = mapping.get(fname, fname)  # 优先用原名，否则回退新名

        tensor, orig_img, _ = load_image(img_path, IMG_SIZE)

        try:
            logits = cam_extractor.forward_only(tensor)
            prob = float(logits[0, 1])
            pred = 1 if prob >= 0.75 else 0  # 分类阈值 0.75，与 eval.py 一致
        except Exception as e:
            print(f"  [FAIL] {fname}: {e}")
            continue
        true = 0 if true_class == "NORMAL" else 1
        ok = true == pred

        gc_name = f"GC_{fname}"                   # CAM 图（互斥：无文本）
        txt_name = f"TXT_{Path(fname).stem}.png"  # 文本标注图（PNG 格式对标参考代码）

        cv2.imwrite(str(out_dir / fname), cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))  # 原图

        if pred == 1:
            cam_map, _ = cam_extractor(tensor, target_class=1)
        else:
            cam_map = None

        # CAM（保持原有逻辑）
        if cam_map is not None:
            overlay_cam, _ = overlay_heatmap(orig_img, cam_map)
        else:
            h, w = orig_img.shape[:2]
            overlay_cam, _ = overlay_heatmap(orig_img, np.zeros((h // 4, w // 4)))
        cv2.imwrite(str(out_dir / gc_name), cv2.cvtColor(overlay_cam, cv2.COLOR_RGB2BGR))

        # 文本标注（互斥：无CAM，PNG 格式对标参考代码）
        overlay_txt = add_text_overlay(orig_img, true_label=true_class, prob=prob)
        cv2.imwrite(str(out_dir / txt_name), cv2.cvtColor(overlay_txt, cv2.COLOR_RGB2BGR))

        records.append((orig_name, fname, gc_name, true_class, pred, prob, ok))
        pbar.set_postfix_str(f"{fname} {'✅' if ok else '❌'}")

    pbar.close()

    ##  终端输出：逐行结果

    LABELS = {0: "NORMAL", 1: "PNEUMONIA"}
    EMOJI = {True: "✅", False: "❌"}

    print(f"  {'Original Name':<30} {'New Name':<12} {'CAM Name':<14} {'Prediction':<12} {'Prob':<8} {'Result'}")
    print(f"  {'-'*30} {'-'*12} {'-'*14} {'-'*12} {'-'*8} {'-'*6}")
    for orig, new, gc, true_cls, pred_int, prob, ok in records:
        pred_label = LABELS[pred_int]
        print(f"  {orig:<30} {new:<12} {gc:<14} {pred_label:<12} {prob:.4f}  {EMOJI[ok]}")

    ##  汇总统计

    total = len(records)
    correct = sum(1 for _, _, _, _, _, _, ok in records if ok)
    norm = [(t, p) for _, _, _, t, p, _, _ in records if t == "NORMAL"]
    pneu = [(t, p) for _, _, _, t, p, _, _ in records if t == "PNEUMONIA"]
    norm_ok = sum(1 for _, p in norm if p == 0)
    pneu_ok = sum(1 for _, p in pneu if p == 1)

    norm_acc = f"{norm_ok}/{len(norm)} = {norm_ok/len(norm)*100:.1f}%" if norm else "0/0 = N/A"
    pneu_acc = f"{pneu_ok}/{len(pneu)} = {pneu_ok/len(pneu)*100:.1f}%" if pneu else "0/0 = N/A"

    save_parent = str(Path(model_result).resolve())
    print(f"\n  {'='*55}")
    print(f"  Accuracy:        {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  NORMAL acc:      {norm_acc}")
    print(f"  PNEUMONIA acc:   {pneu_acc}")
    print(f"  Save Path:       {save_parent}/")

    ##  写入运行日志到结果文件夹

    log_path = out_dir / "run_log.txt"
    with open(log_path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write(f"  Model:       {model_name}\n")
        f.write(f"  Weight:      {model_path}\n")
        f.write(f"  Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Threshold:   0.75\n")
        f.write("=" * 55 + "\n\n")

        f.write(f"  Total images:     {total}\n")
        f.write(f"  Correct:          {correct}\n")
        f.write(f"  Accuracy:         {correct}/{total} = {correct/total*100:.1f}%\n\n")

        f.write(f"  NORMAL:\n")
        f.write(f"    Total:   {len(norm)}\n")
        f.write(f"    Correct: {norm_ok}\n")
        f.write(f"    Acc:     {norm_acc}\n\n")

        f.write(f"  PNEUMONIA:\n")
        f.write(f"    Total:   {len(pneu)}\n")
        f.write(f"    Correct: {pneu_ok}\n")
        f.write(f"    Acc:     {pneu_acc}\n\n")

        f.write("-" * 55 + "\n")
        f.write(f"{'Original Name':<30} {'New Name':<12} {'Prediction':<12} {'Prob':<8} {'Result'}\n")
        f.write("-" * 55 + "\n")
        for orig, new, gc, true_cls, pred_int, prob, ok in records:
            pred_label = LABELS[pred_int]
            mark = "OK" if ok else "FAIL"
            f.write(f"{orig:<30} {new:<12} {pred_label:<12} {prob:.4f}  {mark}\n")
        f.write("-" * 55 + "\n")

    print(f"  Log saved:       {log_path}")

    return correct / total * 100 if total else 0


def main():
    r = process_model(str(model_path), model_name, fix_msf=False)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Accuracy: {r:.1f}%")
    print(f"  Save Path: {Path(model_result).resolve()}/")


if __name__ == "__main__":
    main()
