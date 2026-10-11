"""开放词汇 / 零样本检测核心：YOLO-World + Grounding DINO。

干什么：
    1. ``YOLO-World``（ultralytics 惰性 import）：设置自定义类别词汇后用 COCO/YOLO 数据微调，
       新病害类别只要改词表，不用重新标注整个数据集；
    2. ``Grounding DINO``（transformers）：用中文/英文病名当文本提示直接零样本出框，
       既用于演示「新增类别免标注」，也用于给无标注航拍图自动预标注（伪标签）；
    3. 伪标签落盘工具：``save_yolo_labels`` / ``write_autolabel_report``，产物交给农技端人工复核。

怎么用::

    from src.train.openvocab import build_grounding_dino, detect_with_grounding_dino, save_pseudo_labels
    model, processor, name = build_grounding_dino("IDEA-Research/grounding-dino-tiny", "cpu")
    dets = detect_with_grounding_dino(model, processor, img_rgb, prompts, "cpu", box_threshold=0.3)
    save_pseudo_labels("data/pseudo", "DJI_0001", dets, (640, 640))

依赖：torch / transformers（Grounding DINO）；ultralytics（YOLO-World）；numpy/Pillow。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.labels import LabelBook
from src.data.torch_datasets import RTDetrCocoDataset
from src.train.common import log, set_seed

__all__ = [
    "DEFAULT_GROUNDING_DINO",
    "DEFAULT_YOLO_WORLD",
    "build_grounding_dino",
    "detect_with_grounding_dino",
    "boxes_to_coco",
    "save_pseudo_labels",
    "write_autolabel_report",
    "prepare_yoloworld_dataset",
    "train_yolo_world",
    "set_yolo_world_vocab",
]

#: 推荐权重（Grounding DINO tiny 在 CPU 上也能跑，适合本项目的初筛场景）
DEFAULT_GROUNDING_DINO = "IDEA-Research/grounding-dino-tiny"
#: YOLO-World 预训练权重（ultralytics 仓库名）
DEFAULT_YOLO_WORLD = "yolov8s-worldv2.pt"


# ---------------------------------------------------------------------------
# Grounding DINO
# ---------------------------------------------------------------------------
def build_grounding_dino(
    model_name: str = DEFAULT_GROUNDING_DINO,
    device: str = "cpu",
    local_dir: str | Path | None = None,
):
    """加载 Grounding DINO（零样本开放词汇检测器）。

    参数:
        model_name: transformers 仓库名（默认 tiny）
        device    : 'cpu' / 'cuda' / 'cuda:0'
        local_dir : 本地权重目录（离线环境用）
    返回:
        (model, processor, 实际使用的名字)
    """
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except Exception as exc:
        raise RuntimeError(
            "Grounding DINO 需要 torch 与 transformers。\n"
            "  pip install torch transformers\n"
            f"（原始错误：{exc}）"
        ) from exc
    src = str(local_dir) if (local_dir and Path(local_dir).is_dir()) else str(model_name)
    try:
        processor = AutoProcessor.from_pretrained(src)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(src)
    except Exception as exc:
        raise RuntimeError(
            f"加载 Grounding DINO 失败：{src}\n"
            "  首次使用需联网下载（grounding-dino-tiny 约 700MB）；离线请用 --local-dir 指定本地目录。\n"
            f"（原始错误：{exc}）"
        ) from exc
    model = model.to(device).eval()
    _ = torch
    log(f"Grounding DINO 已加载（{src}）到 {device}")
    return model, processor, src


def detect_with_grounding_dino(
    model: Any,
    processor: Any,
    image_rgb: Any,
    prompts: Sequence[str],
    device: str = "cpu",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    max_boxes: int = 300,
) -> List[Dict[str, Any]]:
    """单图零样本检测。

    参数:
        image_rgb : RGB 的 PIL.Image 或 numpy HxWx3(uint8, RGB)
        prompts   : 文本提示列表，如 ['rice blast', 'healthy leaf']（也可中文）
        box_threshold / text_threshold: 置信度阈值
    返回:
        [{"label": str, "score": float, "bbox": [x1,y1,x2,y2] 像素}, ...]
    """
    try:
        import numpy as np
        import torch
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Grounding DINO 推理需要 numpy/torch/Pillow：{exc}") from exc

    pil = image_rgb
    if isinstance(image_rgb, np.ndarray):
        pil = Image.fromarray(np.asarray(image_rgb)).convert("RGB")
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    text = ". ".join([p.strip().lower() for p in prompts if str(p).strip()]) + "."
    inputs = processor(images=pil, text=text, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs,
        inputs.get("input_ids"),
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        target_sizes=[pil.size[::-1]],
    )[0]
    out: List[Dict[str, Any]] = []
    for box, score, label in zip(res["boxes"], res["scores"], res.get("labels", [])):
        if len(out) >= int(max_boxes):
            break
        out.append(
            {
                "label": str(label),
                "score": round(float(score), 6),
                "bbox": [round(float(v), 2) for v in box.tolist()],
            }
        )
    return out


# ---------------------------------------------------------------------------
# 伪标签落盘（自动标注闭环）
# ---------------------------------------------------------------------------
def boxes_to_coco(
    detections: Sequence[Dict[str, Any]],
    image_record: Dict[str, Any],
    label_book: LabelBook,
    ann_id_start: int = 1,
) -> List[Dict[str, Any]]:
    """把 Grounding DINO 的检测结果转成 COCO annotation 列表。

    标签名会做「英文空格 -> 下划线」的归一化再查 labels.yaml；查不到时归到 healthy 类
    并保留原始 label 到 ``_raw_label`` 供人工复核。
    """
    name2id = label_book.name_to_id_map()
    out: List[Dict[str, Any]] = []
    aid = int(ann_id_start)
    for det in detections:
        raw = str(det["label"]).strip()
        key = raw.lower().replace(" ", "_")
        cid = name2id.get(key)
        if cid is None:
            cid = next((i for n, i in name2id.items() if n.replace("_", " ") == raw.lower()), None)
        if cid is None:
            cid = label_book.healthy_id()
        x1, y1, x2, y2 = (float(v) for v in det["bbox"][:4])
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if w <= 1 or h <= 1:
            continue
        out.append(
            {
                "id": aid,
                "image_id": int(image_record["id"]),
                "category_id": int(cid),
                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
                "segmentation": [],
                "score": float(det["score"]),
                "_raw_label": raw,
            }
        )
        aid += 1
    return out


def save_pseudo_labels(
    pseudo_dir: str | Path,
    image_stem: str,
    detections: Sequence[Dict[str, Any]],
    image_size: Tuple[int, int],
    label_book: LabelBook,
    score_field: str = "score",
) -> Path:
    """把检测结果写成 YOLO 伪标签 ``data/pseudo/<stem>.txt``。

    行格式：``cls cx cy w h score``（第 6 列是置信度，YOLO 标准只读前 5 列，
    多出的 score 列给人工复核时排序用，训练前可用 ``--strip-score`` 去掉）。
    """
    out = Path(pseudo_dir)
    out.mkdir(parents=True, exist_ok=True)
    width, height = float(image_size[0]), float(image_size[1])
    name2id = label_book.name_to_id_map()
    lines: List[str] = []
    for det in detections:
        raw = str(det["label"]).strip()
        key = raw.lower().replace(" ", "_")
        cid = name2id.get(key)
        if cid is None:
            cid = next((i for n, i in name2id.items() if n.replace("_", " ") == raw.lower()), None)
        if cid is None:
            cid = label_book.healthy_id()
        x1, y1, x2, y2 = (float(v) for v in det["bbox"][:4])
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        score = float(det.get(score_field, 1.0))
        raw_name = raw.replace(" ", "_")
        lines.append(
            f"{int(cid)} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f} "
            f"{score:.4f} {raw_name}"
        )
    p = out / f"{image_stem}.txt"
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p


def write_autolabel_report(
    report_path: str | Path,
    entries: Sequence[Dict[str, Any]],
    label_book: LabelBook,
    model_name: str,
    thresholds: Dict[str, Any],
) -> Path:
    """写自动标注报告（中文 json）：每张图的框数、类别分布、平均置信度，便于人工复核排优先级。"""
    per_cls: Dict[int, int] = {}
    scores: List[float] = []
    total_boxes = 0
    for e in entries:
        total_boxes += int(e.get("num_boxes", 0))
        scores.append(float(e.get("mean_score", 0.0)))
        for cid, c in (e.get("class_counts") or {}).items():
            per_cls[int(cid)] = per_cls.get(int(cid), 0) + int(c)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "thresholds": thresholds,
        "num_images": len(entries),
        "num_boxes": total_boxes,
        "mean_confidence": round(sum(scores) / max(1, len(scores)), 4),
        "boxes_per_category": {str(k): v for k, v in sorted(per_cls.items())},
        "category_names": {str(k): label_book.cn_of(k) for k in sorted(per_cls)},
        "review_hint": (
            "请农技人员按 mean_score 从低到高优先复核低置信度图；"
            "pseudo/*.txt 第 6 列是置信度，可用 --strip-score 生成训练用纯 YOLO 标签。"
        ),
        "images": list(entries),
    }
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return p


# ---------------------------------------------------------------------------
# YOLO-World
# ---------------------------------------------------------------------------
def prepare_yoloworld_dataset(
    train_coco: str | Path,
    val_coco: str | Path,
    out_dir: str | Path,
    label_book: LabelBook,
) -> Path:
    """YOLO-World 微调用的数据集（YOLO 目录结构 + data.yaml，与 RT-DETR 后端共用一套逻辑）。"""
    out = Path(out_dir)
    for split, coco in (("train", train_coco), ("val", val_coco)):
        ds = RTDetrCocoDataset(coco)
        ds.export_yolo_dataset(out / split, names=label_book.names, copy_images=True)
        log(f"YOLO-World {split} 集：{len(ds)} 张 -> {out / split}")
    data_yaml = (
        f"path: {out.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        f"nc: {label_book.num_classes}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(label_book.names))
    )
    p = out / "data.yaml"
    p.write_text(data_yaml, encoding="utf-8")
    return p


def set_yolo_world_vocab(model: Any, label_book: LabelBook, lang: str = "en") -> List[str]:
    """设置 YOLO-World 的自定义类别词表（新增病害类别只改这里，不必重标注）。"""
    vocab = label_book.openvocab_prompts(lang=lang)
    try:
        model.set_classes(vocab)
    except Exception as exc:
        raise RuntimeError(
            "model.set_classes(vocab) 失败，请确认使用的是 YOLO-World 权重（*world*.pt）。\n"
            f"（原始错误：{exc}）"
        ) from exc
    log(f"已设置 YOLO-World 词表（{lang}）：{vocab}")
    return vocab


def train_yolo_world(
    data_yaml: str | Path,
    out_dir: str | Path,
    label_book: LabelBook,
    model_name: str = DEFAULT_YOLO_WORLD,
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 4,
    device: str = "cpu",
    lang: str = "en",
    freeze: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """YOLO-World 微调（惰性 import ultralytics；缺包时给中文提示）。

    参数:
        freeze: 冻结前 N 层（8GB 显存可设 10 加速、省显存）
    """
    try:
        from ultralytics import YOLOWorld
    except Exception as exc:
        raise RuntimeError(
            "未安装 ultralytics，无法使用 YOLO-World。\n"
            "  pip install ultralytics\n"
            "  提示：不装也能用 Grounding DINO 零样本路线（transformers 即可）完成开放词汇演示。\n"
            f"（原始错误：{exc}）"
        ) from exc
    set_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLOWorld(str(model_name))
    vocab = set_yolo_world_vocab(model, label_book, lang=lang)
    model.train(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch_size),
        device=("cpu" if str(device).startswith("cpu") else 0),
        project=str(out),
        name="yoloworld",
        exist_ok=True,
        freeze=int(freeze),
        seed=int(seed),
    )
    import shutil

    src = out / "yoloworld" / "weights" / "best.pt"
    dst = out / "best.pt"
    if src.is_file():
        shutil.copy2(src, dst)
        log(f"已复制 YOLO-World 最优权重 -> {dst}")
    else:
        log(f"未找到 YOLO-World 输出权重：{src}", "警告")
    return {"backend": "yoloworld", "weights": str(dst), "vocab": vocab, "run_dir": str(src.parent.parent)}
