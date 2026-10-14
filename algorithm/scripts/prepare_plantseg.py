"""把 PlantSeg 转换成项目可直接训练的数据集（含按 URL 重划，消除划分泄漏）。

背景（实测）：PlantSeg 官方划分存在**跨划分重复**——
    316 个源 URL 跨划分，涉及 1556 张图（20.0%）；
    抽 50 对做像素验证，MAE 中位 **0.90**（同一张照片），68% 的图对是重复的。
直接拿官方划分训练并报 val/test 指标，结果会虚高。

本脚本做四件事：
  1. **按源 URL 重新划分**：同一 URL 的所有图强制进同一个划分，杜绝重复泄漏；
  2. **筛选类别**：从 115 类里选样本充足的若干类；
  3. **补全 COCO**：原数据的 `categories` 是空的、`category_id` 全为 0，
     类别名只在 `Metadata.csv` 里，需要 join 后重建；
  4. 输出到 `data/processed_plantseg/{train,val,test}/{images,annotations.json}`。

用法：
    python scripts/prepare_plantseg.py --zip "D:\\下载\\plantseg.zip" --overwrite
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 精选类别：样本充足（>=100）且**作物多样**。
# 不单纯取样本最多的 10 个（那会是 3 个小麦 + 2 个白粉病，模型可能学"作物颜色"
# 而不是"病斑形态"）。这里覆盖 6 种作物。
DEFAULT_CLASSES: list[tuple[str, str, str]] = [
    ("Wheat",   "wheat powdery mildew", "wheat_powdery_mildew"),
    ("Wheat",   "wheat stripe rust",    "wheat_stripe_rust"),
    ("Wheat",   "wheat head scab",      "wheat_fusarium_head_blight"),
    ("Corn",    "corn rust",            "corn_rust"),
    ("Corn",    "corn smut",            "corn_smut"),
    ("Tomato",  "tomato early blight",  "tomato_early_blight"),
    ("Tomato",  "tomato late blight",   "tomato_late_blight"),
    ("Grape",   "grape downy mildew",   "grape_downy_mildew"),
    ("Apple",   "apple scab",           "apple_scab"),
    ("Citrus",  "citrus canker",        "citrus_canker"),
]
SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIO = (0.70, 0.15, 0.15)


def _norm_url(u: str, uniq: str) -> str:
    """归一化 URL。空 URL 用文件名兜底（保证每张图自成一组）。"""
    u = (u or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"[?#].*$", "", u).rstrip("/")
    return u if u else f"__u__{uniq}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PlantSeg -> 项目数据集（按 URL 重划）")
    ap.add_argument("--zip", default=r"D:\下载\plantseg.zip")
    ap.add_argument("--out", default="data/processed_plantseg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classes", type=int, default=0, help=">0 时按样本数自动取前 N 类")
    ap.add_argument("--no-healthy", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    z = zipfile.ZipFile(args.zip)
    meta = list(csv.DictReader(io.StringIO(z.read("plantseg/Metadata.csv").decode("utf-8-sig"))))

    # 图片在 zip 里的路径：文件名 -> 完整路径
    idx: dict[str, str] = {}
    for n in z.namelist():
        if n.startswith("plantseg/images/") and n.lower().endswith((".jpg", ".jpeg", ".png")):
            idx[n.split("/")[-1]] = n
    print(f"Metadata {len(meta)} 行；zip 内图片 {len(idx)} 张")

    # ---- 1. 选类别 ----------------------------------------------------
    cnt = Counter((r["Plant"].strip(), r["Disease"].strip()) for r in meta)
    if args.classes and args.classes > 0:
        picked = [k for k, _ in cnt.most_common(args.classes)]
        mapping = {k: re.sub(r"[^a-z0-9]+", "_", f"{k[0]}_{k[1]}".lower()).strip("_")
                   for k in picked}
    else:
        mapping = {(p, d): nm for p, d, nm in DEFAULT_CLASSES}

    if not args.no_healthy:
        for k in cnt:
            if "healthy" in k[1].lower():
                mapping.setdefault(k, "healthy")

    missing = [k for k in mapping if k not in cnt]
    if missing:
        print(f"[警告] 数据里不存在，跳过：{missing}")
    mapping = {k: v for k, v in mapping.items() if k in cnt}

    print(f"\n选中 {len(mapping)} 类：")
    for k, v in mapping.items():
        print(f"  {v:<34} {k[0]} / {k[1]}  ({cnt[k]} 张)")

    names = sorted({v for v in mapping.values()})           # 类别名（含 healthy）
    cid_of = {n: i for i, n in enumerate(names)}
    print(f"\n标签体系（{len(names)} 类）：{names}")

    # ---- 2. 建样本表 + 按 URL 分组 ------------------------------------
    samples: list[dict] = []
    for r in meta:
        key = (r["Plant"].strip(), r["Disease"].strip())
        if key not in mapping or r["Name"] not in idx:
            continue
        samples.append({
            "name": r["Name"],
            "path": idx[r["Name"]],
            "cls": mapping[key],
            "cid": cid_of[mapping[key]],
            "group": _norm_url(r["URL"], r["Name"]),
        })
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    gkeys = list(groups)
    random.shuffle(gkeys)
    total = len(samples)
    targets = [r * total for r in SPLIT_RATIO]
    counts = [0, 0, 0]
    split_of: dict[str, int] = {}
    for g in gkeys:
        k = min(range(3), key=lambda i: counts[i] / max(1e-9, targets[i]))
        split_of[g] = k
        counts[k] += len(groups[g])
    print(f"\n按 URL 重划（{len(gkeys)} 个来源组 / {total} 张图）：")
    for i, sp in enumerate(SPLIT_NAMES):
        print(f"  {sp:<6} {counts[i]:5d} 张 ({counts[i]/total*100:.1f}%)")

    # ---- 3. 读取官方标注，建立 文件名 -> [ann] 索引 --------------------
    ann_by_name: dict[str, list[dict]] = defaultdict(list)
    for off in SPLIT_NAMES:
        j = json.loads(z.read(f"plantseg/annotation_{off}.json").decode("utf-8"))
        iid2nm = {im["id"]: im["file_name"] for im in j["images"]}
        for a in j["annotations"]:
            nm = iid2nm.get(a["image_id"])
            if nm:
                ann_by_name[nm].append(a)
    print(f"官方标注文件索引完成：{len(ann_by_name)} 张图有标注")

    # ---- 4. 写出 -------------------------------------------------------
    out = ROOT / args.out
    if out.exists():
        if not args.overwrite:
            print(f"\n[中止] {out} 已存在，加 --overwrite")
            return 1
        shutil.rmtree(out)

    for i, sp in enumerate(SPLIT_NAMES):
        d = out / sp
        (d / "images").mkdir(parents=True, exist_ok=True)
        images, anns = [], []
        iid = aid = 0
        n_nobox = 0
        for g, k in split_of.items():
            if k != i:
                continue
            for s in groups[g]:
                try:
                    with z.open(s["path"]) as f:
                        (d / "images" / s["name"]).write_bytes(f.read())
                except Exception as exc:  # noqa: BLE001
                    print(f"[警告] 解出失败 {s['name']}: {exc}")
                    continue
                iid += 1
                images.append({"id": iid, "file_name": s["name"],
                               "class_name": s["cls"]})
                got = [a for a in ann_by_name.get(s["name"], [])
                       if a.get("bbox") and a["bbox"][2] > 1 and a["bbox"][3] > 1]
                if not got:
                    n_nobox += 1
                for a in got:
                    aid += 1
                    anns.append({"id": aid, "image_id": iid, "category_id": s["cid"],
                                 "area": float(a.get("area", a["bbox"][2] * a["bbox"][3])),
                                 "bbox": [float(v) for v in a["bbox"][:4]],
                                 "iscrowd": int(a.get("iscrowd", 0))})
        coco = {
            "info": {"description": "PlantSeg re-split by source URL (no cross-split leakage)",
                     "version": "1.0", "year": 2024},
            "licenses": [], "images": images, "annotations": anns,
            "categories": [{"id": j, "name": n, "supercategory": "plant"}
                           for j, n in enumerate(names)],
        }
        (d / "annotations.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        per = Counter(a["category_id"] for a in anns)
        print(f"  {sp:<6} {len(images):5d} 图 / {len(anns):6d} 框 / 无框图 {n_nobox}")
        for j, n in enumerate(names):
            if per.get(j):
                print(f"          {n:<34} {per[j]:6d} 框")

    print(f"\n[完成] {out}")
    print("下一步：更新 configs/labels.yaml 的类别，然后重训。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
