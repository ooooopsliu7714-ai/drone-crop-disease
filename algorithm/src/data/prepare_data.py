"""数据准备：把 ``data/raw/`` 统一转换成 ``data/processed/{train,val,test}`` 的 COCO 格式。

干什么：
    1. 自动识别 ``data/raw/`` 里的标注形态（COCO json / YOLO txt / 分类文件夹 / 合成样例）；
    2. 统一规整成 COCO dict（class_id 严格按 ``configs/labels.yaml``），按比例划分 train/val/test；
    3. 做类别统计与类别均衡检查（长尾告警），输出 ``data/dataset_stats.json``；
    4. 生成 ``data/processed/<split>/annotations.json``（新增类别自动补进 categories）。

怎么用::

    # 默认：读 data/raw，按 7:2:1 划分
    python -m src.data.prepare_data

    # 指定目录与比例，并自定义随机种子
    python -m src.data.prepare_data --raw-dir data/raw --out-dir data/processed \
        --split-ratio 0.8,0.1,0.1 --seed 42

    # 直接用极小合成样例验证流程（等价于 python -m scripts.make_samples 之后再跑）
    python -m src.data.prepare_data --raw-dir data/samples --out-dir data/processed

依赖：标准库 + PyYAML；numpy / Pillow **仅在** ``--copy-images`` 需要读图像尺寸时才用（有兜底）。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# 允许 `python src/data/prepare_data.py` 直接执行
if __package__ in (None, ""):  # pragma: no cover - 兼容脚本方式运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.labels import load_labels  # noqa: E402
from src.core.paths import DATA_DIR, ensure_dir  # noqa: E402
from src.data.io import (  # noqa: E402
    IMAGE_EXTS,
    empty_coco,
    save_coco,
    yolo_lines_to_coco,
)

#: 类别样本数低于该阈值时，训练侧应改用小样本策略（DINOv2/v3 线性探测）
LONGTAIL_WARN_THRESHOLD = 5


# ---------------------------------------------------------------------------
# 1. 原始数据发现与加载
# ---------------------------------------------------------------------------
def list_images(root: Path) -> List[Path]:
    """递归列出目录下所有图片（按路径排序，保证可复现）。"""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _read_image_size(path: Path) -> Tuple[int, int]:
    """读图片宽高：优先 Pillow，其次 cv2，最后兜底 640x640 并打印提示。"""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is not None:
            return int(img.shape[1]), int(img.shape[0])
    except Exception:
        pass
    print(f"[警告] 无法读取图像尺寸，按 640x640 处理：{path.name}")
    return 640, 640


def _image_record(img_path: Path, image_id: int, root: Path) -> Dict[str, Any]:
    """构造 COCO 的 images[] 记录，file_name 用相对 root 的 posix 路径。"""
    try:
        rel = img_path.relative_to(root).as_posix()
    except ValueError:
        rel = img_path.name
    w, h = _read_image_size(img_path)
    return {"id": int(image_id), "file_name": rel, "width": w, "height": h}


def load_coco_source(json_path: Path, label_names: Sequence[str]) -> Dict[str, Any]:
    """读入一份 COCO json，并把它的类别映射到 labels.yaml 的 class_id。"""
    with open(json_path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    name2id = {n: i for i, n in enumerate(label_names)}
    cat_map: Dict[int, int] = {}
    for cat in raw.get("categories", []):
        cname = str(cat.get("name") or "").strip()
        key = cname.lower().replace(" ", "_").replace("-", "_")
        if key in name2id:
            cat_map[int(cat["id"])] = name2id[key]
        elif cname in name2id:
            cat_map[int(cat["id"])] = name2id[cname]
        else:
            print(f"[警告] COCO 类别 {cname!r} 不在 labels.yaml 中，跳过该类标注")
    images = list(raw.get("images", []))
    anns: List[Dict[str, Any]] = []
    for a in raw.get("annotations", []):
        if int(a.get("category_id", -1)) not in cat_map:
            continue
        bb = a.get("bbox") or [0, 0, 0, 0]
        if float(bb[2]) <= 1 or float(bb[3]) <= 1:
            continue
        anns.append(
            {
                "id": 0,
                "image_id": int(a["image_id"]),
                "category_id": cat_map[int(a["category_id"])],
                "bbox": [float(v) for v in bb[:4]],
                "area": float(a.get("area") or (float(bb[2]) * float(bb[3]))),
                "iscrowd": int(a.get("iscrowd") or 0),
                "segmentation": a.get("segmentation") or [],
            }
        )
    return {"images": images, "annotations": anns, "categories": [], "_source_json": str(json_path)}


def load_yolo_source(img_dir: Path, names: Sequence[str]) -> Dict[str, Any]:
    """读入 YOLO txt 目录（同级可有 classes.txt / names.txt），返回 COCO dict。"""
    imgs = list_images(img_dir)
    images: List[Dict[str, Any]] = []
    anns: List[Dict[str, Any]] = []
    ann_id = 1
    for i, p in enumerate(imgs, start=1):
        rec = _image_record(p, i, img_dir)
        # 标注文件解析顺序（覆盖真实数据集的各种布局）：
        #   1. 同名 .txt（图片与标签放一起，如 <split>/*.jpg + <split>/*.txt）
        #   2. 同级 labels/（如 <dir>/images/x.jpg + <dir>/images/labels/x.txt）
        #   3. **上一级 labels/**（标准 YOLO 布局：<split>/images/x.jpg + <split>/labels/x.txt）
        #      ← 这一条曾经漏掉，导致"加载了 1539 个标签文件却解析出 0 个框"：
        #        外部数据集（train/valid/test × images/labels）全部读不到标注。
        #   4. 上一级的上一级（如 <root>/<split>/images/x.jpg + <root>/labels/x.txt）
        cands = [
            p.with_suffix(".txt"),
            p.parent / "labels" / f"{p.stem}.txt",
            p.parent.parent / "labels" / f"{p.stem}.txt",
            p.parent.parent.parent / "labels" / f"{p.stem}.txt",
        ]
        txt = next((c for c in cands if c.is_file()), None)
        images.append(rec)
        if txt is None:
            continue
        lines = txt.read_text(encoding="utf-8").splitlines()
        new = yolo_lines_to_coco(lines, rec, ann_id, names)
        anns.extend(new)
        ann_id += len(new)
    return {"images": images, "annotations": anns, "categories": [], "_source_dir": str(img_dir)}


def load_classify_source(root: Path) -> Dict[str, Any]:
    """读入分类文件夹结构 ``root/<class_name>/*.jpg``，转成「整图一个框」的 COCO dict。"""
    name2id: Dict[str, int] = {}
    images: List[Dict[str, Any]] = []
    anns: List[Dict[str, Any]] = []
    idx = 1
    aid = 1
    for sub in sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []:
        imgs = [p for p in sub.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not imgs:
            continue
        name2id[sub.name] = idx
        for p in sorted(imgs):
            rec = _image_record(p, aid, root)
            images.append(rec)
            anns.append(
                {
                    "id": aid,
                    "image_id": rec["id"],
                    "category_id": idx,
                    "bbox": [0.0, 0.0, float(rec["width"]), float(rec["height"])],
                    "area": float(rec["width"]) * float(rec["height"]),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            aid += 1
        idx += 1
    return {
        "images": images,
        "annotations": anns,
        "categories": [{"id": v, "name": k} for k, v in name2id.items()],
        "_source_dir": str(root),
        "_raw_class_names": list(name2id.keys()),
    }


def load_raw(raw_dir: Path, label_names: Sequence[str]) -> Tuple[Dict[str, Any], str]:
    """自动识别 ``raw_dir`` 的标注形态并加载。

    返回:
        (coco_dict, 形态说明字符串)
    识别顺序: COCO json -> YOLO txt -> 分类文件夹 -> 报错
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"原始数据目录不存在：{raw_dir}\n"
            "请把标注数据放进去，或先运行 `python -m scripts.make_samples` 生成合成样例。"
        )

    json_files = sorted(
        p
        for p in raw_dir.rglob("*.json")
        if p.name not in ("dataset_stats.json",) and "processed" not in p.parts
    )
    coco_jsons = []
    for p in json_files:
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                head = json.load(f)
        except Exception:
            continue
        if isinstance(head, dict) and {"images", "annotations"} <= set(head.keys()):
            coco_jsons.append(p)
    if coco_jsons:
        merged = {"images": [], "annotations": [], "categories": []}
        for p in coco_jsons:
            part = load_coco_source(p, label_names)
            merged["images"].extend(part["images"])
            merged["annotations"].extend(part["annotations"])
        # image_id 去重重排
        id_map: Dict[int, int] = {}
        for new_id, rec in enumerate(merged["images"], start=1):
            id_map[int(rec["id"])] = new_id
            rec["id"] = new_id
        for i, a in enumerate(merged["annotations"], start=1):
            a["id"] = i
            a["image_id"] = id_map.get(int(a["image_id"]), int(a["image_id"]))
        return merged, f"COCO json ×{len(coco_jsons)}（{', '.join(p.name for p in coco_jsons[:3])}...）"

    txt_files = sorted(p for p in raw_dir.rglob("*.txt") if p.name not in ("classes.txt", "names.txt"))
    if txt_files:
        img_dirs = {p.parent for p in txt_files}
        merged = {"images": [], "annotations": [], "categories": []}
        for d in sorted(img_dirs):
            part = load_yolo_source(d, label_names)
            id_map: Dict[int, int] = {}
            for rec in part["images"]:
                new_id = len(merged["images"]) + 1
                id_map[int(rec["id"])] = new_id
                rec["id"] = new_id
                merged["images"].append(rec)
            for a in part["annotations"]:
                na = dict(a)
                na["id"] = len(merged["annotations"]) + 1
                na["image_id"] = id_map.get(int(a["image_id"]), int(a["image_id"]))
                merged["annotations"].append(na)
        return merged, f"YOLO txt ×{len(txt_files)}（{len(merged['images'])} 张图）"

    subdirs = [d for d in raw_dir.iterdir() if d.is_dir()] if raw_dir.exists() else []
    if subdirs and any(list_images(d) for d in subdirs):
        part = load_classify_source(raw_dir)
        if part.get("_raw_class_names"):
            part["_cn_note"] = "分类文件夹结构，已转成整图框"
        return part, f"分类文件夹 ×{len(subdirs)}"

    raise RuntimeError(
        f"在 {raw_dir} 下没有识别到任何标注（COCO json / YOLO txt / 分类文件夹）。\n"
        "请检查数据放置方式，或运行 `python -m scripts.make_samples` 生成合成样例。"
    )


# ---------------------------------------------------------------------------
# 2. 划分与统计
# ---------------------------------------------------------------------------
def split_image_ids(
    image_ids: Sequence[int],
    ratio: Sequence[float],
    seed: int,
    stratify: Dict[int, int] | None = None,
    min_per_split: int = 1,
) -> Dict[str, List[int]]:
    """按比例划分 image_id（分层轮转 + 打乱，简单可复现）。

    参数:
        image_ids    : 全部 image_id
        ratio        : (train, val, test) 三段比例，和不必等于 1（会归一化）
        seed         : 随机种子
        stratify     : ``{image_id: 主类别 id}``；给出时按类别分层轮转分配，
                       避免「val 里恰好只有多框图」这类偏斜（小数据集尤其明显）
        min_per_split: 样本够时保证 val/test 至少各有这么多张（默认 1），避免空划分
    返回:
        ``{"train": [...], "val": [...], "test": [...]}``
    """
    if len(ratio) != 3 or min(ratio) < 0 or sum(ratio) <= 0:
        raise ValueError(f"--split-ratio 必须是三个非负且和>0 的数，收到 {ratio}")
    rng = random.Random(int(seed))
    ids = list(image_ids)

    if stratify:
        buckets: Dict[int, List[int]] = defaultdict(list)
        for i in ids:
            buckets[int(stratify.get(i, -1))].append(i)
        # 每个类别内部打乱后轮转（round-robin）取图，保证各 split 的类别分布接近
        for k in buckets:
            rng.shuffle(buckets[k])
        ordered: List[int] = []
        cursor = {k: 0 for k in buckets}
        remaining = set(buckets)
        while remaining:
            for k in list(remaining):
                b = buckets[k]
                if cursor[k] < len(b):
                    ordered.append(b[cursor[k]])
                    cursor[k] += 1
                if cursor[k] >= len(b):
                    remaining.discard(k)
    else:
        rng.shuffle(ids)
        ordered = ids

    total = sum(ratio)
    n = len(ordered)
    n_train, n_val, n_test = _allocate_counts(n, ratio, int(min_per_split))
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val : n_train + n_val + n_test],
    }


def _allocate_counts(n: int, ratio: Sequence[float], min_per_split: int = 1) -> Tuple[int, int, int]:
    """把 n 张图按比例分给 train/val/test，并在样本够时保证每个 split 至少 min_per_split 张。

    策略：先按比例取整；若某个「比例 >0 的 split」数量不足下限，就从当前最大的 split
    转移 1 张过来，最多迭代 3 轮（小数据集够用且结果确定）。
    """
    total = float(sum(ratio))
    counts = [int(round(n * r / total)) for r in ratio]
    counts = [max(0, min(c, n)) for c in counts]
    # 修正取整误差，使三者之和恰好等于 n
    while sum(counts) > n:
        i = counts.index(max(counts))
        if counts[i] == 0:
            break
        counts[i] -= 1
    while sum(counts) < n:
        i = counts.index(max(counts))
        counts[i] += 1
    limit = max(0, int(min_per_split))
    if limit > 0 and n >= 2 * limit:
        for _ in range(3):
            for i in range(3):
                if ratio[i] > 0 and counts[i] < limit:
                    j = counts.index(max(counts))
                    if counts[j] > limit and j != i:
                        counts[j] -= 1
                        counts[i] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def compute_stats(coco: Dict[str, Any], lb, splits: Dict[str, List[int]] | None = None) -> Dict[str, Any]:
    """计算类别统计 / 类别均衡检查，返回写进 dataset_stats.json 的 dict。"""
    per_class: Dict[int, Dict[str, Any]] = {
        c.class_id: {
            "class_id": c.class_id,
            "disease_name": c.disease_name,
            "disease_name_cn": c.disease_name_cn,
            "crop": c.crop,
            "is_disease": c.is_disease,
            "boxes": 0,
            "images": 0,
        }
        for c in lb.classes
    }
    img_of_class: Dict[int, set] = defaultdict(set)
    for a in coco["annotations"]:
        cid = int(a["category_id"])
        if cid not in per_class:
            continue
        per_class[cid]["boxes"] += 1
        img_of_class[cid].add(int(a["image_id"]))
    for cid, s in img_of_class.items():
        per_class[cid]["images"] = len(s)

    boxes_list = [per_class[c]["boxes"] for c in sorted(per_class)]
    total_boxes = sum(boxes_list)
    max_boxes = max(boxes_list) if boxes_list else 0
    rare = [per_class[c]["disease_name"] for c in sorted(per_class)
            if 0 < per_class[c]["boxes"] < LONGTAIL_WARN_THRESHOLD]
    missing = [per_class[c]["disease_name"] for c in sorted(per_class) if per_class[c]["boxes"] == 0]

    per_split: Dict[str, Any] = {}
    if splits:
        img2split = {}
        for name, ids in splits.items():
            for i in ids:
                img2split[int(i)] = name
        for name in splits:
            counts = {c: 0 for c in sorted(per_class)}
            for a in coco["annotations"]:
                if img2split.get(int(a["image_id"])) == name and int(a["category_id"]) in counts:
                    counts[int(a["category_id"])] += 1
            per_split[name] = {
                "num_images": len(splits[name]),
                "num_boxes": int(sum(counts.values())),
                "boxes_per_class": {str(k): int(v) for k, v in counts.items()},
            }

    imbalance = 0.0
    if total_boxes > 0 and max_boxes > 0:
        # 简单均衡度：1 - (最大类占比 - 平均类占比)，1 表示完全均衡
        imbalance = round(max(0.0, 1.0 - (max_boxes / total_boxes - 1.0 / max(1, len(per_class)))), 4)

    stats = {
        "labels_version": getattr(lb, "version", 1),
        "num_classes_total": lb.num_classes,
        "num_classes_present": int(sum(1 for v in per_class.values() if v["boxes"] > 0)),
        "num_images": len(coco["images"]),
        "num_boxes": total_boxes,
        "avg_boxes_per_image": round(total_boxes / max(1, len(coco["images"])), 3),
        "classes": [per_class[c] for c in sorted(per_class)],
        "splits": per_split,
        "balance_check": {
            "longtail_threshold": LONGTAIL_WARN_THRESHOLD,
            "rare_classes": rare,
            "missing_classes": missing,
            "balance_score": imbalance,
            "advice": (
                "存在样本数 <5 的稀有类，建议用 scripts/train_cls.py 的 DINOv2/v3 冻结 backbone 小样本路径，"
                "或用 scripts/train_openvocab.py 的 Grounding DINO 文本提示零样本兜底。"
                if rare or missing
                else "类别分布可接受，可直接用 RT-DETR 全量微调。"
            ),
        },
        "source": coco.get("_source_json") or coco.get("_source_dir") or "unknown",
    }
    return stats


def build_processed(
    coco: Dict[str, Any],
    splits: Dict[str, List[int]],
    out_dir: Path,
    lb,
    copy_images: bool = False,
    raw_root: Path | None = None,
) -> Dict[str, Path]:
    """把划分结果写成 COCO json（每个 split 一份），返回 {split: json路径}。

    `raw_root`：原始数据根目录，供 --copy-images 定位源图。必须由调用方传入——
    `coco["_source_dir"]` 在多源合并时会被覆盖（本机实测：load_raw 会把它改成
    不存在的 data/raw，导致 1539 张源图只索引到 9 个，图片一张也复制不出来）。
    """
    id2rec = {int(r["id"]): r for r in coco["images"]}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[int(a["image_id"])].append(a)

    written: Dict[str, Path] = {}
    for split, ids in splits.items():
        sub = empty_coco(lb.sidecar_classes())
        for i in ids:
            rec = id2rec.get(int(i))
            if rec is None:
                continue
            new_rec = dict(rec)
            sub["images"].append(new_rec)
            for a in anns_by_img.get(int(i), []):
                na = dict(a)
                na["id"] = len(sub["annotations"]) + 1
                sub["annotations"].append(na)
        split_dir = ensure_dir(out_dir / split)
        if copy_images:
            # 复制图片后必须把 file_name 重写成相对 annotations.json 的路径。
            # 否则（本机实测踩过）原始 file_name 形如 "train/images/x.jpg"，
            # 而文件实际落在 processed/train/images/x.jpg，路径多了一层 ->
            # 训练脚本报"1077 张图片未找到，已从数据集中剔除"，整个训练集被清空。
            img_out = ensure_dir(split_dir / "images")
            kept: List[Dict[str, Any]] = []
            kept_anns: List[Dict[str, Any]] = []
            for rec in sub["images"]:
                src = _find_src_image(coco, rec, out_dir, raw_root=raw_root)
                if src is None:
                    continue
                shutil.copy2(src, img_out / src.name)
                new_rec = dict(rec)
                new_rec["file_name"] = f"images/{src.name}"
                kept.append(new_rec)
                for a in anns_by_img.get(int(rec["id"]), []):
                    na = dict(a)
                    na["id"] = len(kept_anns) + 1
                    kept_anns.append(na)
            if len(kept) != len(sub["images"]):
                print(
                    f"[警告] {split}: 有 {len(sub['images']) - len(kept)} 张图找不到源文件，"
                    f"已从该划分中剔除（保留 {len(kept)} 张）"
                )
            sub["images"] = kept
            sub["annotations"] = kept_anns
        # 无论是否复制图片，都要写出 annotations.json 并登记路径
        json_path = save_coco(sub, split_dir / "annotations.json")
        written[split] = json_path
    return written


def _find_src_image(
    coco: Dict[str, Any], rec: Dict[str, Any], out_dir: Path, raw_root: Path | None = None
) -> Path | None:
    """在原始数据根目录下定位图片（供 --copy-images 使用）。

    做法：把原始根目录下所有图片按**文件名**建索引，再按 file_name 查。
    为什么不做逐条路径推断：真实数据集布局千奇百怪
    （`<split>/images/`、`<split>/labels/`、图与 txt 同级……），逐条推断在本机
    实测踩过两次坑——val/test 全找不到源图、train 的 annotations.json 被写成空清单。

    `raw_root` 优先，由调用方（main）传入。**不能只依赖 coco["_source_dir"]**：
    多源合并时它会被覆盖成本机不存在的 `data/raw`，本机实测导致 1539 张源图
    只索引到 9 个。索引按 raw_root 分别缓存。
    """
    roots: List[Path] = []
    if raw_root is not None:
        roots.append(Path(raw_root))
    src = coco.get("_source_json") or coco.get("_source_dir")
    if src:
        sp = Path(src)
        roots.append(sp.parent if sp.is_file() else sp)
    roots.extend([out_dir.parent / "raw", out_dir.parent / "samples"])

    cache_key = "_image_index::" + str(roots[0] if roots else "")
    idx = coco.get(cache_key)
    if idx is None:
        idx = {}
        for r in roots:
            if not r.is_dir():
                continue
            for p in r.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    idx.setdefault(p.name, p)
        coco[cache_key] = idx

    name = Path(str(rec.get("file_name", ""))).name
    if name and name in idx:
        return idx[name]
    # 兜底：名字被改写过（例如 .rf.<hash> 后缀）时按主干名再找一次
    stem = Path(name).stem
    for k, v in idx.items():
        if Path(k).stem == stem:
            return v
    return None


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    ap = argparse.ArgumentParser(
        prog="python -m src.data.prepare_data",
        description="把 data/raw 统一转换成 data/processed/{train,val,test} 的 COCO 格式，并生成 dataset_stats.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m src.data.prepare_data\n"
            "  python -m src.data.prepare_data --raw-dir data/samples --split-ratio 0.75,0.125,0.125\n"
        ),
    )
    ap.add_argument("--raw-dir", default=str(DATA_DIR / "raw"), help="原始数据目录，默认 data/raw")
    ap.add_argument("--out-dir", default=str(DATA_DIR / "processed"), help="输出目录，默认 data/processed")
    ap.add_argument("--labels", default=None, help="标签体系 yaml，默认 configs/labels.yaml")
    ap.add_argument(
        "--split-ratio",
        default="0.7,0.2,0.1",
        help="train,val,test 划分比例（逗号分隔三个数），默认 0.7,0.2,0.1",
    )
    ap.add_argument("--seed", type=int, default=42, help="划分随机种子，默认 42")
    ap.add_argument("--copy-images", action="store_true", help="把图片复制到 processed/<split>/images/（占磁盘）")
    ap.add_argument("--stats-name", default="dataset_stats.json", help="统计文件名，默认 dataset_stats.json")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """入口：返回进程退出码（0 成功，非 0 失败）。"""
    args = build_parser().parse_args(argv)
    try:
        ratio = tuple(float(x) for x in str(args.split_ratio).split(","))
    except ValueError:
        print(f"[错误] --split-ratio 解析失败：{args.split_ratio!r}，应为 0.7,0.2,0.1 形式", file=sys.stderr)
        return 2
    if len(ratio) != 3:
        print(f"[错误] --split-ratio 需要三个数，收到 {len(ratio)} 个", file=sys.stderr)
        return 2

    try:
        lb = load_labels(args.labels)
    except Exception as exc:
        print(f"[错误] 标签体系加载失败：{exc}", file=sys.stderr)
        return 3

    raw_dir = Path(args.raw_dir)
    out_dir = ensure_dir(args.out_dir)
    print("=" * 78)
    print("数据准备：data/raw -> data/processed（COCO 格式）")
    print("=" * 78)
    for line in lb.summary_lines():
        print(line)

    try:
        coco, form = load_raw(raw_dir, lb.names)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 4

    if not coco["images"]:
        print(f"[错误] 在 {raw_dir} 下没有找到任何图片", file=sys.stderr)
        return 4
    if not coco["annotations"]:
        print(f"[警告] 在 {raw_dir} 下没有找到任何标注框，只做图片清单转换")

    print(f"\n识别到的数据形态：{form}")
    print(f"图片数：{len(coco['images'])}，标注框数：{len(coco['annotations'])}")

    # 分层划分：用每张图包含的第一个类别作为主类别
    strat: Dict[int, int] = {}
    for rec in coco["images"]:
        anns = [a for a in coco["annotations"] if int(a["image_id"]) == int(rec["id"])]
        strat[int(rec["id"])] = int(anns[0]["category_id"]) if anns else -1

    splits = split_image_ids([int(r["id"]) for r in coco["images"]], ratio, args.seed, strat)
    stats = compute_stats(coco, lb, splits)

    written = build_processed(
        coco, splits, out_dir, lb, copy_images=bool(args.copy_images), raw_root=raw_dir
    )
    stats_path = ensure_dir(out_dir) / args.stats_name
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n划分结果：")
    for name in ("train", "val", "test"):
        s = stats["splits"].get(name, {})
        print(f"  {name:<5s} 图片 {s.get('num_images', 0):>4d} 张，标注框 {s.get('num_boxes', 0):>5d} 个"
              f"  -> {written.get(name)}")

    print("\n类别统计（boxes=标注框数, images=出现该类的图片数）：")
    for c in stats["classes"]:
        flag = "" if c["boxes"] > 0 else "  <-- 本数据集缺失"
        print(f"  [{c['class_id']:>2d}] {c['disease_name']:<30s} {c['disease_name_cn']:<12s} "
              f"boxes={c['boxes']:>5d} images={c['images']:>4d}{flag}")

    bc = stats["balance_check"]
    print(f"\n类别均衡检查：balance_score={bc['balance_score']}")
    if bc["rare_classes"]:
        print(f"  稀有类（<{bc['longtail_threshold']} 个框）：{', '.join(bc['rare_classes'])}")
    if bc["missing_classes"]:
        print(f"  缺失类：{', '.join(bc['missing_classes'])}")
    print(f"  建议：{bc['advice']}")
    print(f"\n[完成] 统计已写入：{stats_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
