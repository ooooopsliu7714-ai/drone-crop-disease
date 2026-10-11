"""检测/分类评估：纯 numpy 实现（不依赖 pycocotools），输出中文报告。

干什么：
    1. 读真值（COCO json 或 YOLO txt）与预测（COCO json 或 YOLO txt，或用 ``--ckpt-dir`` 现场推理）；
    2. 计算 VOC 风格 **AP@0.5**、**mAP@0.5:0.95**（10 个 IoU 阈值）、每类 Precision/Recall/F1；
    3. 输出**混淆矩阵**（按 IoU>=0.5 的匹配对，含「背景」行列）；
    4. 农业专用指标：
       - **发病率** = 患病（真值病斑框并集）面积 / 总面积；
       - **病情指数** = Σ(病级 × 该级面积) / (3 × 总面积) × 100（病级由框面积占比映射，健康=0）；
       - **严重度占比** = healthy/light/moderate/severe 四档的图片数与面积占比。
    5. ``--demo``：在合成样例上自动生成确定性「模拟预测」，无需任何权重即可跑通并打印中文报告。

怎么用::

    # 1) 在合成样例上直接跑（无需权重，确定性输出）
    python -m scripts.eval --demo

    # 2) 评估真实预测（COCO json 预测）
    python -m scripts.eval --gt data/processed/val/annotations.json \
        --pred data/preds/val --pred-format coco --conf 0.25 --out-json models/eval_val.json

    # 3) 用训练好的模型现场推理并评估（torch 后端，不依赖 onnxruntime）
    python -m scripts.eval --gt data/processed/val/annotations.json \
        --ckpt-dir models/det_rtdetrv2 --device auto

    # 4) 评估 YOLO txt 标注的小样例数据
    python -m scripts.eval --demo --gt-format yolo --gt data/samples --pred-format yolo

依赖：numpy（必需）、标准库；运行 ``--ckpt-dir`` 时才需要 torch/transformers；
      **不需要** pycocotools、torchvision、ultralytics。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 允许 `python scripts/eval.py` 直接执行
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.labels import SEVERITY_INDEX, severity_from_area_ratio
from src.core.paths import DATA_DIR, MODELS_DIR, PROJECT_ROOT, ensure_dir
from src.data.io import IMAGE_EXTS, load_coco, resolve_image_path, yolo_lines_to_coco

#: 评估用的 IoU 阈值序列（VOC/COCO 口径 0.5:0.05:0.95）
IOU_THRESHOLDS: Tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
#: 判定为「检出」的 IoU 阈值（用于混淆矩阵与农业指标）
IOU_MATCH = 0.5


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
class Box:
    """一个检测框（真值或预测）。"""

    __slots__ = ("image_id", "class_id", "bbox", "score")

    def __init__(self, image_id: int, class_id: int, bbox: Sequence[float], score: float = 1.0):
        self.image_id = int(image_id)
        self.class_id = int(class_id)
        self.bbox = [float(v) for v in bbox[:4]]  # xyxy
        self.score = float(score)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Box(img={self.image_id}, cls={self.class_id}, xyxy={self.bbox}, s={self.score:.3f})"


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 xyxy 框的 IoU（纯 numpy 无关的标量实现，保证在小数据上足够快且无依赖）。"""
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 载入真值 / 预测
# ---------------------------------------------------------------------------
def _images_and_anns_from_coco(coco: Dict[str, Any], json_path: Path) -> Tuple[Dict[int, Dict[str, Any]], List[Box]]:
    """COCO json -> ({image_id: image_record}, [真值 Box])。"""
    images: Dict[int, Dict[str, Any]] = {}
    for rec in coco["images"]:
        r = dict(rec)
        if not r.get("width") or not r.get("height"):
            p = resolve_image_path(coco, r, json_path)
            if p is not None:
                r["width"], r["height"] = _read_size(p)
        images[int(r["id"])] = r
    boxes: List[Box] = []
    for a in coco["annotations"]:
        x, y, w, h = (float(v) for v in a["bbox"][:4])
        boxes.append(Box(int(a["image_id"]), int(a["category_id"]), [x, y, x + w, y + h], 1.0))
    return images, boxes


def _read_size(path: Path) -> Tuple[int, int]:
    """读图片宽高（Pillow -> cv2 -> 兜底 640x640）。"""
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
    return 640, 640


def _load_yolo_dir(root: Path, names: Sequence[str], is_pred: bool) -> Tuple[Dict[int, Dict[str, Any]], List[Box]]:
    """YOLO txt 目录 -> ({image_id: image_record}, [Box])。

    目录约定：图片与同名 .txt 同级；预测时 conf 取 txt 里的第 6 列（缺省 1.0）。
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在：{root}")
    imgs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    images: Dict[int, Dict[str, Any]] = {}
    boxes: List[Box] = []
    name2id = {n: i for i, n in enumerate(names)}
    for i, p in enumerate(imgs, start=1):
        w, h = _read_size(p)
        images[i] = {"id": i, "file_name": p.name, "width": w, "height": h, "_path": str(p)}
        cands = [p.with_suffix(".txt"), p.parent / "labels" / f"{p.stem}.txt"] if not is_pred else [
            p.with_suffix(".txt"), p.parent / f"{p.stem}.txt"
        ]
        txt = next((c for c in cands if c.is_file()), None)
        if txt is None:
            continue
        for line in txt.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            cls_raw, cx, cy, bw, bh = parts[0], *(float(v) for v in parts[1:5])
            if cls_raw.lstrip("-").isdigit():
                cid = int(float(cls_raw))
            else:  # 允许类别名（自动标注脚本会写类别名便于复核）
                cid = name2id.get(cls_raw.lower().replace(" ", "_"), 0)
            score = float(parts[5]) if len(parts) > 5 else 1.0
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
            boxes.append(Box(i, cid, [x1, y1, x2, y2], score))
    return images, boxes


def load_ground_truth(gt: Path, lb, gt_format: str = "auto") -> Tuple[Dict[int, Dict[str, Any]], List[Box], str]:
    """载入真值，返回 (images, boxes, 实际格式)。

    ``gt`` 可以是：COCO json 文件 / COCO json 所在目录 / YOLO 图片目录。
    """
    gt = Path(gt)
    fmt = gt_format
    if fmt == "auto":
        if gt.is_file() and gt.suffix.lower() == ".json":
            fmt = "coco"
        elif gt.is_dir() and (gt / "annotations.json").is_file():
            fmt = "coco"
            gt = gt / "annotations.json"
        else:
            fmt = "yolo"
    if fmt == "coco":
        if gt.is_dir():
            for cand in ("annotations.json", "_annotations.coco.json"):
                if (gt / cand).is_file():
                    gt = gt / cand
                    break
        coco = load_coco(gt)
        images, boxes = _images_and_anns_from_coco(coco, Path(gt))
        return images, boxes, "coco"
    images, boxes = _load_yolo_dir(gt, lb.names, is_pred=False)
    return images, boxes, "yolo"


def load_predictions(pred: Path, lb, images: Dict[int, Dict[str, Any]], pred_format: str = "auto") -> List[Box]:
    """载入预测框。

    COCO 模式：``pred`` 是 json 文件或含 json 的目录（可多份合并）；
    YOLO 模式 ：``pred`` 是含 ``<stem>.txt`` 的目录（按文件名匹配真值图片 id）。
    """
    pred = Path(pred)
    fmt = pred_format
    if fmt == "auto":
        if pred.is_file() and pred.suffix.lower() == ".json":
            fmt = "coco"
        elif pred.is_dir():
            fmt = "coco" if any(pred.glob("*.json")) else "yolo"
        else:
            fmt = "coco"
    if fmt == "coco":
        files = [pred] if pred.is_file() else sorted(pred.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"{pred} 下没有 json 预测文件")
        boxes: List[Box] = []
        for f in files:
            coco = load_coco(f)
            name2id = lb.name_to_id_map()
            id2name = {int(c["id"]): str(c.get("name", c["id"])) for c in coco.get("categories", [])}
            for a in coco.get("annotations", []):
                cid = int(a["category_id"])
                if cid not in range(lb.num_classes):
                    cid = name2id.get(id2name.get(cid, ""), lb.healthy_id())
                x, y, w, h = (float(v) for v in a["bbox"][:4])
                boxes.append(Box(int(a["image_id"]), cid, [x, y, x + w, y + h], float(a.get("score", 1.0))))
            for res in coco.get("predictions", []) or []:
                cid = int(res.get("category_id", 0))
                b = res.get("bbox") or [0, 0, 0, 0]
                x, y, w, h = (float(v) for v in b[:4])
                boxes.append(Box(int(res["image_id"]), cid, [x, y, x + w, y + h], float(res.get("score", 1.0))))
        return boxes
    # YOLO：按文件名匹配 image_id
    stem2id = {Path(rec["file_name"]).stem: iid for iid, rec in images.items()}
    boxes = []
    for txt in sorted(pred.rglob("*.txt")):
        iid = stem2id.get(txt.stem)
        if iid is None:
            continue
        rec = images[iid]
        w, h = float(rec.get("width") or 1), float(rec.get("height") or 1)
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            score = float(parts[5]) if len(parts) > 5 else 1.0
            boxes.append(Box(iid, cid, [x1, y1, x2, y2], score))
    return boxes


# ---------------------------------------------------------------------------
# 核心指标
# ---------------------------------------------------------------------------
def _greedy_match(gt_list: List[Box], pred_list: List[Box], iou_thr: float) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """贪心匹配（按 score 降序，每个 GT 只能匹配一次）。

    返回: (matches=[(gt_idx, pred_idx, iou)], 未匹配 gt 下标, 未匹配 pred 下标)
    """
    matched_gt: Dict[int, int] = {}
    matches: List[Tuple[int, int, float]] = []
    order = sorted(range(len(pred_list)), key=lambda i: -pred_list[i].score)
    for pi in order:
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gt_list):
            if gi in matched_gt:
                continue
            v = iou_xyxy(g.bbox, pred_list[pi].bbox)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0 and best_iou >= iou_thr:
            matched_gt[best_gi] = pi
            matches.append((best_gi, pi, best_iou))
    unmatched_gt = [gi for gi in range(len(gt_list)) if gi not in matched_gt]
    matched_pred = {pi for _, pi, _ in matches}
    unmatched_pred = [pi for pi in range(len(pred_list)) if pi not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def compute_ap(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    class_id: int,
    iou_thr: float,
    n_images: int,
) -> Dict[str, Any]:
    """单个类别、单个 IoU 阈值下的 AP（VOC 11 点插值 + 连续面积两种都给）。"""
    gts = [b for b in gt_boxes if b.class_id == class_id]
    preds = sorted([b for b in pred_boxes if b.class_id == class_id], key=lambda b: -b.score)
    n_gt = len(gts)
    if n_gt == 0 and not preds:
        return {"ap": 0.0, "ap11": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "n_gt": 0, "n_pred": 0, "empty": True}

    gt_by_img: Dict[int, List[Box]] = defaultdict(list)
    for g in gts:
        gt_by_img[g.image_id].append(g)
    used: Dict[int, set] = defaultdict(set)

    tp = [0] * len(preds)
    fp = [0] * len(preds)
    for i, p in enumerate(preds):
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_by_img.get(p.image_id, [])):
            if j in used[p.image_id]:
                continue
            v = iou_xyxy(g.bbox, p.bbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            used[p.image_id].add(best_j)
            tp[i] = 1
        else:
            fp[i] = 1

    if n_gt == 0:
        # 该类没有真值：所有预测都是误报，AP 记为 0（同时给出误报数便于排查）
        return {"ap": 0.0, "ap11": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": int(sum(fp)), "fn": 0, "n_gt": 0, "n_pred": len(preds), "empty": True}

    cum_tp, cum_fp = 0, 0
    recalls, precisions = [], []
    for i in range(len(preds)):
        cum_tp += tp[i]
        cum_fp += fp[i]
        precisions.append(cum_tp / max(1, cum_tp + cum_fp))
        recalls.append(cum_tp / n_gt)
    # 单调化
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    prev_r = 0.0
    for i, r in enumerate(recalls):
        ap += (r - prev_r) * precisions[i]
        prev_r = r
    ap11 = 0.0
    for t in [x / 10 for x in range(11)]:
        p_at = max([precisions[i] for i in range(len(recalls)) if recalls[i] >= t], default=0.0)
        ap11 += p_at / 11
    final_tp, final_fp = int(sum(tp)), int(sum(fp))
    prec = final_tp / max(1, final_tp + final_fp)
    rec = final_tp / n_gt
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "ap": round(ap, 6),
        "ap11": round(ap11, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "tp": final_tp,
        "fp": final_fp,
        "fn": int(n_gt - final_tp),
        "n_gt": n_gt,
        "n_pred": len(preds),
        "empty": False,
    }


def compute_confusion_matrix(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    num_classes: int,
    iou_thr: float = IOU_MATCH,
) -> Dict[str, Any]:
    """按 IoU>=thr 贪心匹配统计混淆矩阵（行=真值类别，列=预测类别）。

    ``matrix[i][j]``：真值为 i 被判为 j 的数量；``matrix[i][num_classes]`` 为漏检；
    ``matrix[num_classes][j]`` 为背景误报。由真值图与预测图配对后按「每图独立匹配」
    近似实现（类别不同也可匹配，用于发现「把稻瘟病当成水稻白叶枯」这类错误）。
    """
    size = num_classes + 1
    matrix = [[0] * size for _ in range(size)]
    gt_by_img: Dict[int, List[Box]] = defaultdict(list)
    pred_by_img: Dict[int, List[Box]] = defaultdict(list)
    for g in gt_boxes:
        gt_by_img[g.image_id].append(g)
    for p in pred_boxes:
        pred_by_img[p.image_id].append(p)
    for iid in set(gt_by_img) | set(pred_by_img):
        gs = gt_by_img.get(iid, [])
        ps = pred_by_img.get(iid, [])
        matches, un_gt, un_pred = _greedy_match(gs, ps, iou_thr)
        for gi, pi, _ in matches:
            gi_c = min(gs[gi].class_id, num_classes)
            pi_c = min(ps[pi].class_id, num_classes)
            matrix[gi_c][pi_c] += 1
        for gi in un_gt:
            matrix[min(gs[gi].class_id, num_classes)][num_classes] += 1
        for pi in un_pred:
            matrix[num_classes][min(ps[pi].class_id, num_classes)] += 1
    return {"matrix": matrix, "labels": [str(i) for i in range(num_classes)] + ["__background__"],
            "iou_threshold": iou_thr}


def compute_agri_metrics(
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    images: Dict[int, Dict[str, Any]],
    lb,
    iou_thr: float = IOU_MATCH,
) -> Dict[str, Any]:
    """农业专用指标：发病率、病情指数、严重度占比。

    定义（写清楚，避免各人理解不一致）：
        - 面积口径：**病灶框面积之和 / 图片面积之和**（并集不做去重，属于保守估计，
          与农技端「病斑面积占比」的目测习惯更接近）；
        - 发病率 = Σ患病(真值)框面积 / Σ图片面积；
        - 病情指数 = Σ(病级 × 该框面积) / (3 × Σ图片面积) × 100，病级由
          ``severity_from_area_ratio(框面积/图片面积)`` 给出（healthy=0, light=1, moderate=2, severe=3）；
        - 严重度占比：图片级 severity（按该图患病面积占比分档）的四档图片数与面积占比；
        - 同时给出「检出框的发病率」，用于对比漏检带来的低估（检出率 = 检出病斑面积 / 真值病斑面积）。
    """
    total_area = 0.0
    gt_dis_area = 0.0
    gt_weighted = 0.0
    pred_dis_area = 0.0
    pred_weighted = 0.0
    per_image: List[Dict[str, Any]] = []
    sev_img_count: Dict[str, int] = {k: 0 for k in SEVERITY_INDEX}
    sev_area: Dict[str, float] = {k: 0.0 for k in SEVERITY_INDEX}
    per_class_area: Dict[int, float] = defaultdict(float)

    pred_by_img: Dict[int, List[Box]] = defaultdict(list)
    for p in pred_boxes:
        pred_by_img[p.image_id].append(p)
    gt_by_img: Dict[int, List[Box]] = defaultdict(list)
    for g in gt_boxes:
        gt_by_img[g.image_id].append(g)

    for iid, rec in images.items():
        w = float(rec.get("width") or 0)
        h = float(rec.get("height") or 0)
        img_area = max(1.0, w * h)
        total_area += img_area
        gs = gt_by_img.get(iid, [])
        ps = pred_by_img.get(iid, [])
        g_area = 0.0
        g_w = 0.0
        for g in gs:
            if not lb.is_disease(g.class_id):
                continue
            a = max(0.0, g.bbox[2] - g.bbox[0]) * max(0.0, g.bbox[3] - g.bbox[1])
            g_area += a
            ratio = min(1.0, a / img_area)
            lvl = SEVERITY_INDEX[severity_from_area_ratio(ratio)]
            g_w += lvl * a
            per_class_area[g.class_id] += a
        p_area = 0.0
        p_w = 0.0
        for p in ps:
            if not lb.is_disease(p.class_id):
                continue
            a = max(0.0, p.bbox[2] - p.bbox[0]) * max(0.0, p.bbox[3] - p.bbox[1])
            p_area += a
            ratio = min(1.0, a / img_area)
            p_w += SEVERITY_INDEX[severity_from_area_ratio(ratio)] * a
        img_ratio = min(1.0, g_area / img_area)
        img_sev = severity_from_area_ratio(img_ratio)
        sev_img_count[img_sev] += 1
        sev_area[img_sev] += img_area
        gt_dis_area += g_area
        gt_weighted += g_w
        pred_dis_area += p_area
        pred_weighted += p_w
        per_image.append(
            {
                "image_id": iid,
                "file_name": rec.get("file_name"),
                "area_px": round(img_area, 1),
                "gt_disease_area": round(g_area, 1),
                "gt_ratio": round(img_ratio, 6),
                "gt_severity": img_sev,
                "pred_disease_area": round(p_area, 1),
                "pred_ratio": round(min(1.0, p_area / img_area), 6),
            }
        )

    incidence = gt_dis_area / total_area if total_area > 0 else 0.0
    incidence_pred = pred_dis_area / total_area if total_area > 0 else 0.0
    # 发病率 > 100% 说明标注是「单株级/互相重叠」的，而不是「病斑级」不重叠标注：
    # 框面积相加会重复计入重叠区域（本机实测：外部胁迫数据集为单株标注、每图 13.6 框，
    # 真值发病率算出 105%）。这不是计算错误，但必须显式提示，
    # 否则报告里出现 >100% 会被当成硬伤，也说明该类指标在此数据集上不可用。
    area_note = ""
    if incidence > 1.0 or incidence_pred > 1.0:
        area_note = (
            "发病率超过 100%：本数据集的标注框互相重叠（单株级标注），"
            "框面积之和会重复计算重叠区域，因此该数据集上「发病率/病情指数」不可用；"
            "要用这些指标，需要病斑级（不重叠）标注，或改用框面积并集口径。"
        )
    di = gt_weighted / (3.0 * total_area) * 100.0 if total_area > 0 else 0.0
    di_pred = pred_weighted / (3.0 * total_area) * 100.0 if total_area > 0 else 0.0
    n_img = max(1, len(images))
    return {
        "total_area_px": round(total_area, 1),
        "num_images": len(images),
        "incidence": round(incidence, 6),
        "incidence_pred": round(incidence_pred, 6),
        "disease_incidence_rate": round(incidence, 6),
        "disease_index": round(di, 4),
        "disease_index_pred": round(di_pred, 4),
        "area_metric_note": area_note,
        "detection_area_ratio": round(pred_dis_area / gt_dis_area, 4) if gt_dis_area > 0 else 0.0,
        "severity_distribution": {
            k: {
                "images": int(sev_img_count[k]),
                "image_ratio": round(sev_img_count[k] / n_img, 4),
                "area_ratio": round(sev_area[k] / total_area, 4) if total_area > 0 else 0.0,
            }
            for k in SEVERITY_INDEX
        },
        "disease_area_by_class": {str(k): round(v, 1) for k, v in sorted(per_class_area.items())},
        "per_image": per_image,
    }


def evaluate(
    images: Dict[int, Dict[str, Any]],
    gt_boxes: Sequence[Box],
    pred_boxes: Sequence[Box],
    lb,
    conf_threshold: float = 0.25,
) -> Dict[str, Any]:
    """汇总所有指标（AP / PRF1 / 混淆矩阵 / 农业指标）。

    ⚠️ 两个口径必须分开（早期版本混在一起，导致 mAP 被人为截断）：
      - **AP / mAP 是阈值无关指标**：必须用**全部预测**（按分数排序）画 PR 曲线。
        早期版本先按 conf_threshold 砍掉预测再算 AP，等于截断曲线——
        实测同一模型 conf=0.05 时 mAP50=0.2521，conf=0.25 时只有 0.0039，
        看起来像"模型没学会"，其实是评估方法错了。
      - **P/R/F1 是某个工作点上的指标**：按 conf_threshold 过滤后再算，
        用于选服务端阈值。
    所以下面分成 `all_preds`（算 AP）与 `preds`（算 P/R/F1）。
    """
    all_preds = sorted(pred_boxes, key=lambda p: -p.score)   # AP 用：全部预测
    preds = [p for p in all_preds if p.score >= float(conf_threshold)]  # P/R/F1 用：阈值过滤后
    n_cls = lb.num_classes

    per_class: Dict[int, Dict[str, Any]] = {}
    ap_matrix: Dict[int, Dict[str, float]] = {}
    for cid in range(n_cls):
        row: Dict[str, float] = {}
        for thr in IOU_THRESHOLDS:
            # AP：用全部预测（阈值无关）
            m = compute_ap(gt_boxes, all_preds, cid, thr, len(images))
            row[thr] = m["ap"]
            if thr == 0.5:
                # P/R/F1：用阈值过滤后的预测（有实际意义的工作点指标）
                mt = compute_ap(gt_boxes, preds, cid, thr, len(images))
                per_class[cid] = {
                    **m,
                    "precision": mt["precision"],
                    "recall": mt["recall"],
                    "f1": mt["f1"],
                    "tp": mt["tp"],
                    "fp": mt["fp"],
                    "fn": mt["fn"],
                    "n_pred": mt["n_pred"],
                    "note": "ap 为阈值无关（用全部预测）；precision/recall/f1 为当前 conf 工作点",
                }
        ap_matrix[cid] = row

    present = [cid for cid in range(n_cls) if (per_class.get(cid, {}).get("n_gt", 0) or 0) > 0]
    map50 = sum(ap_matrix[c][0.5] for c in present) / len(present) if present else 0.0
    map5095 = (
        sum(sum(ap_matrix[c].values()) / len(IOU_THRESHOLDS) for c in present) / len(present) if present else 0.0
    )
    total_tp = sum(per_class.get(c, {}).get("tp", 0) for c in range(n_cls))
    total_fp = sum(per_class.get(c, {}).get("fp", 0) for c in range(n_cls))
    total_fn = sum(per_class.get(c, {}).get("fn", 0) for c in range(n_cls))
    micro_p = total_tp / max(1, total_tp + total_fp)
    micro_r = total_tp / max(1, total_tp + total_fn)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    return {
        "num_images": len(images),
        "num_gt": len(gt_boxes),
        "num_pred_raw": len(pred_boxes),
        "num_pred_kept": len(preds),
        "conf_threshold": float(conf_threshold),
        "map50": round(map50, 6),
        "map50_95": round(map5095, 6),
        "micro": {"precision": round(micro_p, 6), "recall": round(micro_r, 6), "f1": round(micro_f1, 6),
                  "tp": total_tp, "fp": total_fp, "fn": total_fn},
        "per_class": {str(k): v for k, v in per_class.items()},
        "ap_by_iou": {str(k): {str(t): round(v, 6) for t, v in row.items()} for k, row in ap_matrix.items()},
        "confusion_matrix": compute_confusion_matrix(gt_boxes, preds, n_cls, IOU_MATCH),
        "agri": compute_agri_metrics(gt_boxes, preds, images, lb, IOU_MATCH),
    }


# ---------------------------------------------------------------------------
# 模拟预测（--demo 用，确定性，不引入随机性以便复现）
# ---------------------------------------------------------------------------
def synthesize_predictions(gt_boxes: Sequence[Box], lb) -> List[Box]:
    """基于真值生成**确定性**模拟预测：约 15% 漏检、20% 类别混淆、每类 1~2 个误报。

    定位误差按 IoU 阈值放大（IoU 越高的评估点要求定位越准），这样 mAP@0.5:0.95 才会
    像真实模型一样随 IoU 阈值单调下降，而不是一条直线；整个函数不含随机数，跑两次结果一致。
    """
    scale = 0.05  # 基础抖动系数（相对框尺寸），对应 IoU 约 0.75~0.95
    out: List[Box] = []
    n = len(gt_boxes)
    for i, g in enumerate(gt_boxes):
        h = int(abs(g.bbox[0] * 7 + g.bbox[1] * 13 + g.image_id * 31 + g.class_id * 17)) % 100
        if h < 15:  # 15% 漏检
            continue
        w = max(4.0, float(g.bbox[2] - g.bbox[0]))
        hh = max(4.0, float(g.bbox[3] - g.bbox[1]))
        # 抖动幅度按 (h%5) 分 5 档：0 -> 几乎完美；4 -> 较大偏差（高 IoU 阈值下会掉分）
        k = (h % 5) / 4.0
        dx = (k * scale * 1.6 - scale * 0.8) * w
        dy = (k * scale * 1.2 - scale * 0.6) * hh
        box = [g.bbox[0] + dx, g.bbox[1] + dy, g.bbox[2] + dx, g.bbox[3] + dy]
        cls = g.class_id
        if h >= 85 and lb.num_classes > 1:  # 15% 类别混淆
            cls = (g.class_id + 1) % lb.num_classes
            if cls == lb.healthy_id() and lb.num_classes > 2:
                cls = 2
        score = round(min(0.99, 0.62 + (h % 37) / 100.0), 4)
        out.append(Box(g.image_id, cls, box, score))
    # 每类补 1 个误报（放在不同位置，确保 FP 列非零）
    for c in range(min(lb.num_classes, 4)):
        if n == 0:
            break
        ref = gt_boxes[c % n]
        out.append(
            Box(ref.image_id, c, [5.0 + 30 * c, 5.0, 45.0 + 30 * c, 45.0], round(0.35 + 0.05 * c, 4))
        )
    return out


# ---------------------------------------------------------------------------
# 预测来源：用训练产物现场推理
# ---------------------------------------------------------------------------
def predict_with_ckpt(
    images: Dict[int, Dict[str, Any]],
    gt_json: Path,
    ckpt_dir: str | Path,
    device: str = "auto",
    image_roots: Sequence[str] | None = None,
    max_images: int = 0,
    conf_threshold: float | None = None,
) -> List[Box]:
    """用 ``src.train.predictors.build_predictor`` 现场推理（torch 后端）。

    `conf_threshold`：**必须在这里设**，不能只靠评估阶段的过滤。
    原因（本机实测）：预测器内部用自己的 conf_threshold（sidecar 里是 0.25）先过滤一遍，
    评估的 `--conf` 是在那之后才生效的。当模型分数整体偏低时（实测最高分只有 0.13~0.26），
    0.25 的内部阈值会把几乎所有框丢掉——308 张图只剩 51 个框，
    AP 的上限被锁死在 51/4251，看起来像"模型完全没学会"。
    评估是**阈值无关**的指标，应当尽量保留低分框来画完整的 PR 曲线。
    """
    import numpy as np

    from src.train.predictors import build_predictor

    predictor = build_predictor(str(ckpt_dir), device=device)
    if conf_threshold is not None:
        # 取更小的那个，保证评估能看到低分框；记录以便日志可追溯
        old = getattr(predictor, "conf_threshold", None)
        predictor.conf_threshold = float(conf_threshold)
        print(
            f"[信息] 预测器内部置信度阈值 {old} -> {predictor.conf_threshold}"
            "（评估需要看到低分框才能算出真实 AP）"
        )
    boxes: List[Box] = []
    ids = sorted(images.keys())
    if max_images and max_images > 0:
        ids = ids[: int(max_images)]
    for iid in ids:
        rec = images[iid]
        p = resolve_image_path({}, rec, gt_json, extra_roots=image_roots)
        if p is None:
            cand = Path(str(rec.get("file_name", "")))
            p = cand if cand.is_file() else None
        if p is None:
            print(f"[警告] 找不到图片，跳过 image_id={iid}（{rec.get('file_name')}）")
            continue
        try:
            import cv2

            img = cv2.imread(str(p))
        except Exception as exc:
            raise RuntimeError(f"读取图片失败（推理需要 opencv-python）：{p}（{exc}）") from exc
        if img is None:
            print(f"[警告] cv2 读不到图片：{p}")
            continue
        res = predictor.predict_debug([np.asarray(img)])[0]
        for b in res.get("boxes", []):
            boxes.append(Box(iid, int(b["class_id"]), b["bbox"], float(b["confidence"])))
    return boxes


# ---------------------------------------------------------------------------
# 报告打印
# ---------------------------------------------------------------------------
def print_report(result: Dict[str, Any], lb, gt_source: str, pred_source: str) -> None:
    """打印中文评估报告。"""
    line = "=" * 78
    print(line)
    print("农作物病害检测评估报告（纯 numpy 实现，未使用 pycocotools）")
    print(line)
    print(f"真值来源：{gt_source}")
    print(f"预测来源：{pred_source}")
    print(f"图片数：{result['num_images']}  真值框：{result['num_gt']}  "
          f"预测框（过滤前/后）：{result['num_pred_raw']} / {result['num_pred_kept']}  "
          f"置信度阈值：{result['conf_threshold']}")

    print("\n【总体指标】")
    print(f"  AP@0.5        (mAP50)     = {result['map50']:.4f}")
    print(f"  mAP@0.5:0.95              = {result['map50_95']:.4f}")
    m = result["micro"]
    print(f"  微平均 Precision/Recall/F1 = {m['precision']:.4f} / {m['recall']:.4f} / {m['f1']:.4f}"
          f"  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

    print("\n【每类指标 @IoU0.5】")
    print(f"  {'class_id':>8s} {'类别':<30s} {'中文名':<12s} {'GT':>5s} {'Pred':>5s} "
          f"{'TP':>5s} {'FP':>5s} {'FN':>5s} {'P':>7s} {'R':>7s} {'F1':>7s} {'AP50':>7s}")
    for cid in range(lb.num_classes):
        r = result["per_class"].get(str(cid))
        if not r:
            continue
        ap50 = result["ap_by_iou"].get(str(cid), {}).get("0.5", 0.0)
        print(f"  {cid:>8d} {lb.name_of(cid):<30s} {lb.cn_of(cid):<12s} {r['n_gt']:>5d} {r['n_pred']:>5d} "
              f"{r['tp']:>5d} {r['fp']:>5d} {r['fn']:>5d} {r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['f1']:>7.4f} {ap50:>7.4f}")

    print("\n【各 IoU 阈值下的 mAP（仅统计有真值的类别）】")
    present = [c for c in range(lb.num_classes) if result["per_class"].get(str(c), {}).get("n_gt", 0) > 0]
    header = "  " + " ".join(f"{t:>6.2f}" for t in IOU_THRESHOLDS)
    print(header)
    vals = []
    for t in IOU_THRESHOLDS:
        s = sum(result["ap_by_iou"].get(str(c), {}).get(str(t), 0.0) for c in present)
        vals.append(s / max(1, len(present)))
    print("  " + " ".join(f"{v:>6.4f}" for v in vals))

    cm = result["confusion_matrix"]["matrix"]
    n_cls = lb.num_classes
    print(f"\n【混淆矩阵 @IoU>=0.5】行=真值类别，列=预测类别（最后一列=漏检，最后一行=背景误报）")
    short = [f"{cid}" for cid in range(n_cls)] + ["BG"]
    print("  " + " " * 6 + " ".join(f"{s:>5s}" for s in short))
    for i in range(n_cls + 1):
        label = "BG" if i == n_cls else f"{i}:{lb.name_of(i)[:8]}"
        print(f"  {label:>6s} " + " ".join(f"{v:>5d}" for v in cm[i]))
    diag = sum(cm[i][i] for i in range(n_cls))
    tot = sum(sum(r) for r in cm)
    print(f"  对角命中 {diag} / 总配对 {tot}（其中一致率 {diag / max(1, tot):.4f}）")

    ag = result["agri"]
    print("\n【农业专用指标】")
    print(f"  图片数 {ag['num_images']}，总像素面积 {ag['total_area_px']:.0f}")
    print(f"  发病率（真值病斑面积/总面积）      = {ag['incidence'] * 100:.4f}%")
    print(f"  发病率（模型检出病斑面积/总面积）  = {ag['incidence_pred'] * 100:.4f}%")
    print(f"  病情指数 DI（Σ病级×面积/(3×总面积)×100） = {ag['disease_index']:.4f}")
    print(f"  病情指数 DI（按模型检出框计算）          = {ag['disease_index_pred']:.4f}")
    print(f"  病斑面积检出率（检出面积/真值面积）      = {ag['detection_area_ratio'] * 100:.2f}%")
    if ag.get("area_metric_note"):
        print(f"\n  ⚠️ {ag['area_metric_note']}")
    print("  严重度占比（图片级，按该图患病面积占比分档）：")
    for k in ("healthy", "light", "moderate", "severe"):
        d = ag["severity_distribution"][k]
        cn = {"healthy": "健康", "light": "轻度", "moderate": "中度", "severe": "重度"}[k]
        print(f"    {k:<9s} {cn}  图片 {d['images']:>4d} 张（{d['image_ratio'] * 100:>6.2f}%）  "
              f"面积占比 {d['area_ratio'] * 100:>6.2f}%")
    if ag["disease_area_by_class"]:
        print("  各病害病斑面积（像素）：")
        for k, v in ag["disease_area_by_class"].items():
            cid = int(k)
            print(f"    [{cid:>2d}] {lb.name_of(cid):<30s} {lb.cn_of(cid):<12s} {v:>12.1f}")
    print(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    ap = argparse.ArgumentParser(
        prog="python -m scripts.eval",
        description="检测评估（纯 numpy，无 pycocotools）：VOC AP@0.5 / mAP@0.5:0.95 / P-R-F1 / 混淆矩阵 / 发病率 / 病情指数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m scripts.eval --demo\n"
            "  python -m scripts.eval --gt data/processed/val/annotations.json --pred data/preds/val\n"
            "  python -m scripts.eval --gt data/processed/val/annotations.json --ckpt-dir models/det_rtdetrv2\n"
        ),
    )
    ap.add_argument("--gt", default=None, help="真值：COCO json 文件或含标注的目录（--demo 时可省）")
    ap.add_argument("--gt-format", default="auto", choices=["auto", "coco", "yolo"], help="真值格式，默认自动识别")
    ap.add_argument("--pred", default=None, help="预测：COCO json（文件/目录）或 YOLO txt 目录")
    ap.add_argument("--pred-format", default="auto", choices=["auto", "coco", "yolo"], help="预测格式，默认自动识别")
    ap.add_argument("--ckpt-dir", default=None, help="用训练产物现场推理（models/<name>），与 --pred 二选一")
    ap.add_argument("--device", default="auto", help="现场推理的设备：auto/cpu/cuda")
    ap.add_argument("--image-root", action="append", default=[], help="图片搜索根目录，可重复传入")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    ap.add_argument("--demo", action="store_true", help="在 data/samples 上生成确定性模拟预测并评估")
    ap.add_argument("--demo-dir", default=str(DATA_DIR / "samples"), help="--demo 使用的样例目录")
    ap.add_argument("--max-images", type=int, default=0, help="现场推理最多处理多少张图（0=全部）")
    ap.add_argument("--only-images", default=None,
                    help="只评估名单里的图（每行一个文件名或 split/文件名）。"
                         "用于剔除与 train 重复的图、得到无泄漏的诚实指标")
    ap.add_argument("--out-json", default=None, help="把完整指标写入 json（默认不写）")
    ap.add_argument("--quiet", action="store_true", help="只写 json，不打印详细报告")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口：返回退出码（0 成功；1 真实指标全为 0/无数据；2 参数错误）。"""
    args = build_parser().parse_args(argv)
    try:
        from src.core.labels import load_labels

        lb = load_labels()
    except Exception as exc:
        print(f"[错误] 标签体系加载失败：{exc}", file=sys.stderr)
        return 2

    gt_source = args.gt
    pred_source = args.pred
    pred_boxes: List[Box] = []
    images: Dict[int, Dict[str, Any]] = {}
    try:
        if args.demo:
            demo = Path(args.demo_dir)
            if not demo.exists():
                print(
                    f"[错误] 样例目录不存在：{demo}\n"
                    "  请先运行 `python -m scripts.make_samples` 生成合成样例。",
                    file=sys.stderr,
                )
                return 2
            images, gt_boxes, fmt = load_ground_truth(demo, lb, args.gt_format)
            gt_source = f"{demo}（{fmt}，合成样例）"
            pred_boxes = synthesize_predictions(gt_boxes, lb)
            pred_source = "内置确定性模拟预测（--demo）"
        else:
            if not args.gt:
                print("[错误] 需要 --gt 指定真值（或用 --demo）", file=sys.stderr)
                return 2
            images, gt_boxes, fmt = load_ground_truth(Path(args.gt), lb, args.gt_format)
            gt_source = f"{args.gt}（{fmt}）"

            # 按名单过滤：用于剔除与 train 重复的图，得到无泄漏的诚实指标
            # （本机实测该数据集 28.4% 的 val 图在 train 里有像素级副本）
            if args.only_images:
                keep = set()
                for line in Path(args.only_images).read_text(encoding="utf-8-sig").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        keep.add(Path(line).name)   # 只比较文件名，忽略前缀
                before = len(images)
                images = {k: v for k, v in images.items() if Path(str(v.get("file_name", ""))).name in keep}
                gt_boxes = [b for b in gt_boxes if b.image_id in images]
                print(f"[信息] --only-images 过滤：{before} -> {len(images)} 张图，{len(gt_boxes)} 个真值框")
                gt_source = f"{gt_source}（名单过滤：{Path(args.only_images).name}）"

            if args.ckpt_dir:
                pred_boxes = predict_with_ckpt(
                    images, Path(args.gt), args.ckpt_dir, device=args.device,
                    image_roots=args.image_root, max_images=args.max_images,
                    # AP 是阈值无关指标：让预测器也放低阈值，避免它在评估前就把低分框丢掉
                    conf_threshold=min(float(args.conf), 0.05),
                )
                pred_source = f"{args.ckpt_dir} 现场推理（device={args.device}）"
            elif args.pred:
                pred_boxes = load_predictions(Path(args.pred), lb, images, args.pred_format)
                pred_source = f"{args.pred}（{args.pred_format}）"
            else:
                print("[错误] 需要 --pred 或 --ckpt-dir 之一（或用 --demo）", file=sys.stderr)
                return 2
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    if not images:
        print("[错误] 没有读到任何图片，无法评估", file=sys.stderr)
        return 2
    if not gt_boxes:
        print("[警告] 真值里没有任何标注框，指标将全为 0（请检查标注文件）", file=sys.stderr)

    result = evaluate(images, gt_boxes, pred_boxes, lb, conf_threshold=args.conf)
    result["gt_source"] = gt_source
    result["pred_source"] = pred_source
    result["labels_version"] = getattr(lb, "version", 1)

    if not args.quiet:
        print_report(result, lb, gt_source, pred_source)

    if args.out_json:
        out = Path(args.out_json)
        ensure_dir(out.parent)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[完成] 指标已写入 {out}")

    if result["num_gt"] == 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
