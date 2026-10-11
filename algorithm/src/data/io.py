"""COCO / YOLO 标注读写工具（纯标准库 + numpy 可选）。

干什么：
    - 统一的 COCO dict 读写（``images`` / ``annotations`` / ``categories`` 三大字段）；
    - 图片路径解析（COCO 里的 ``file_name`` 常是相对路径，需要在若干候选根目录下探测）；
    - COCO <-> YOLO txt 互转（自动标注 ``scripts/autolabel.py`` 与数据准备都用它）。

怎么用::

    from src.data.io import load_coco, save_coco, resolve_image_path, coco_to_yolo_lines
    coco = load_coco("data/processed/train/annotations.json")
    p = resolve_image_path(coco, coco["images"][0], "data/processed/train")
    lines = coco_to_yolo_lines(coco, coco["images"][0], names=lb.names)

依赖：标准库；仅在使用 ``--copy-images`` 之类功能时用到 shutil。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

__all__ = [
    "load_coco",
    "save_coco",
    "empty_coco",
    "resolve_image_path",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "clip_xyxy",
    "coco_to_yolo_lines",
    "yolo_lines_to_coco",
    "count_annotations_per_category",
]

IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ---------------------------------------------------------------------------
# COCO 基础
# ---------------------------------------------------------------------------
def empty_coco(classes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """构造一个空的 COCO dict；``classes`` 为 sidecar 风格的类别列表。"""
    categories = [
        {
            "id": int(c["class_id"]),
            "name": str(c["disease_name"]),
            "name_cn": str(c.get("disease_name_cn", c["disease_name"])),
        }
        for c in classes
    ]
    return {
        "info": {"description": "drone-crop-disease-algo", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }


def load_coco(path: str | Path) -> Dict[str, Any]:
    """读取 COCO json，做最小结构校验。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"COCO 标注文件不存在：{p}")
    # utf-8-sig：容忍标注 json 带 BOM（Windows 手工编辑常见）
    with open(p, "r", encoding="utf-8-sig") as f:
        coco = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"{p} 不是合法 COCO json：缺少字段 {key!r}")
    return coco


def save_coco(coco: Dict[str, Any], path: str | Path) -> Path:
    """写出 COCO json（UTF-8，保留中文类别名不转义）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    return p


# ---------------------------------------------------------------------------
# 坐标转换
# ---------------------------------------------------------------------------
def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    """COCO 的 [x, y, w, h] -> [x1, y1, x2, y2]。"""
    x, y, w, h = (float(v) for v in box[:4])
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    """[x1, y1, x2, y2] -> COCO 的 [x, y, w, h]。"""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2 - x1, y2 - y1]


def clip_xyxy(box: Sequence[float], width: int, height: int) -> List[float]:
    """把 **COCO 的 [x, y, w, h]** 裁剪到图像范围内，返回像素 [x1, y1, x2, y2]。

    命名保留 ``clip_xyxy`` 是因为返回值是 xyxy；输入沿用 COCO 的 xywh 口径，
    与 :func:`xywh_to_xyxy` 组合即可得未裁剪结果。
    """
    x1, y1, w, h = (float(v) for v in box[:4])
    x2, y2 = x1 + w, y1 + h
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(0.0, min(x2, float(width)))
    y2 = max(0.0, min(y2, float(height)))
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# 图片路径解析
# ---------------------------------------------------------------------------
def resolve_image_path(
    coco: Dict[str, Any],
    image_record: Dict[str, Any],
    json_path_or_dir: str | Path,
    extra_roots: Iterable[str | Path] | None = None,
) -> Path | None:
    """在若干候选根目录下探测图片真实路径，找不到返回 None。

    候选顺序：
        1. ``extra_roots`` 逐个拼接；
        2. 参数所在目录（json 文件取父目录）/ file_name；
        3. 从该目录向上最多 4 层找 ``images/`` 目录 / file_name；
        4. 该目录的 ``images`` 子目录 / file_name。
    """
    file_name = str(image_record.get("file_name") or "")
    if not file_name:
        return None
    rel = Path(file_name)
    base = Path(json_path_or_dir)
    base_dir = base.parent if base.is_file() else base

    cands: List[Path] = []
    for root in extra_roots or []:
        cands.append(Path(root) / rel.name)
        cands.append(Path(root) / rel)
    cands.append(base_dir / rel)
    cands.append(base_dir / rel.name)

    walk = base_dir
    for _ in range(4):
        cands.append(walk / "images" / rel.name)
        walk = walk.parent
    cands.append(base_dir / "images" / rel.name)

    for c in cands:
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    # 兜底：递归找同名文件（样例/小数据集规模下可接受）
    for root in [base_dir] + [Path(r) for r in (extra_roots or [])]:
        if not root.exists():
            continue
        hit = next((p for p in root.rglob(rel.name) if p.is_file()), None)
        if hit is not None:
            return hit
    _ = coco  # 保留参数以便未来支持 image_id 索引
    return None


# ---------------------------------------------------------------------------
# COCO <-> YOLO
# ---------------------------------------------------------------------------
def coco_to_yolo_lines(
    coco: Dict[str, Any],
    image_record: Dict[str, Any],
    names: Sequence[str] | None = None,
) -> List[str]:
    """把某张图的所有标注转成 YOLO 行：``cls cx cy w h``（均归一化到 0~1）。

    ``names`` 给出时按类别名匹配（保证 txt 里的类别顺序与 labels.yaml 一致）；
    未给出时直接用 COCO 的 category_id 当列号。
    """
    img_id = image_record.get("id")
    width = float(image_record.get("width") or 0)
    height = float(image_record.get("height") or 0)
    name2i = {n: i for i, n in enumerate(names)} if names else None
    cat_names = {int(c["id"]): str(c.get("name", c["id"])) for c in coco.get("categories", [])}

    lines: List[str] = []
    for ann in coco.get("annotations", []):
        if ann.get("image_id") != img_id:
            continue
        if width <= 0 or height <= 0:
            raise ValueError(f"image id={img_id} 缺少 width/height，无法归一化")
        x, y, w, h = (float(v) for v in ann["bbox"][:4])
        x1, y1, x2, y2 = clip_xyxy([x, y, w, h], int(width), int(height))
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            continue  # 丢弃退化框
        cat_id = int(ann["category_id"])
        cls = name2i.get(cat_names.get(cat_id, ""), cat_id) if name2i else cat_id
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def yolo_lines_to_coco(
    lines: Sequence[str],
    image_record: Dict[str, Any],
    ann_id_start: int,
    names: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """把 YOLO 行（归一化 cx cy w h）转成 COCO annotation 列表。"""
    width = float(image_record.get("width") or 0)
    height = float(image_record.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"image id={image_record.get('id')} 缺少 width/height")
    out: List[Dict[str, Any]] = []
    aid = ann_id_start
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        bx1, by1, bx2, by2 = clip_xyxy([x1, y1, x2, y2], int(width), int(height))
        if bx2 - bx1 <= 1 or by2 - by1 <= 1:
            continue
        w, h = bx2 - bx1, by2 - by1
        out.append(
            {
                "id": aid,
                "image_id": image_record["id"],
                "category_id": cls,
                "bbox": [round(bx1, 2), round(by1, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
                "segmentation": [],
            }
        )
        aid += 1
    _ = names
    return out


def count_annotations_per_category(coco: Dict[str, Any]) -> Dict[int, int]:
    """统计每类标注框数量，返回 {category_id: count}。"""
    counts: Dict[int, int] = {int(c["id"]): 0 for c in coco.get("categories", [])}
    for ann in coco.get("annotations", []):
        cid = int(ann.get("category_id", -1))
        counts[cid] = counts.get(cid, 0) + 1
    return counts
