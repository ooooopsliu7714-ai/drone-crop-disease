"""torch Dataset 集合：RT-DETR 检测数据集、分类数据集、纯推理数据集。

干什么：
    - 把 ``data/processed/<split>/annotations.json`` 读成 PyTorch Dataset；
    - 检测样本返回 ``{"image": PIL.Image, "image_id": int, "annotations": [...]}``（COCO 风格），
      由训练脚本按各自后端（transformers / ultralytics / paddle）在 collate 阶段转格式；
    - 分类样本返回 ``{"pixel_values": Tensor(C,H,W), "label": Tensor}``，自带
      ImageNet 归一化的最小预处理（不依赖 torchvision / albumentations）；
    - 另提供 ``SimpleImageFolder`` 给自动标注 / 推理演示用（无标注也要能遍历图片）。

怎么用::

    from src.data.torch_datasets import RTDetrCocoDataset, ImageClassifyDataset, discover_coco_json
    ds = RTDetrCocoDataset("data/processed/train/annotations.json")
    ds[0]["image"].size, len(ds[0]["annotations"])
    cls_ds = ImageClassifyDataset.from_coco("data/processed/train/annotations.json")

依赖：torch（必需，缺失时中文提示）、Pillow / numpy（必需）；不依赖 torchvision、albumentations。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.data.io import (
    IMAGE_EXTS,
    load_coco,
    resolve_image_path,
    xywh_to_xyxy,
)
from src.core.paths import default_image_roots

__all__ = [
    "ensure_torch",
    "discover_coco_json",
    "RTDetrCocoDataset",
    "ImageClassifyDataset",
    "SimpleImageFolder",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

#: DINOv2/DINOv3 官方预处理使用的 ImageNet 统计量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_torch() -> Any:
    """惰性 import torch；缺失时给出中文提示并抛 RuntimeError。"""
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "缺少 PyTorch，训练与 torch 后端推理都无法运行。\n"
            "  安装（CPU 版）：pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  安装（RTX 5070 / Blackwell sm_120 需要 cu128）：\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            f"（原始错误：{exc}）"
        ) from exc
    import torch

    return torch


def discover_coco_json(split_dir: str | Path) -> Optional[Path]:
    """在 ``processed/<split>/`` 下寻找 annotations.json（兼容几个常见命名）。"""
    d = Path(split_dir)
    for name in ("annotations.json", "_annotations.coco.json", "instances.json", "coco.json"):
        p = d / name
        if p.is_file():
            return p
    hits = sorted(d.glob("*.json")) if d.is_dir() else []
    for p in hits:
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                d0 = json.load(f)
            if isinstance(d0, dict) and {"images", "annotations"} <= set(d0.keys()):
                return p
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# 检测数据集
# ---------------------------------------------------------------------------
class RTDetrCocoDataset:
    """RT-DETR / YOLO 通用的 COCO 检测数据集（不做任何图像缩放，交给各自的 image processor）。

    参数:
        coco_json    : COCO 标注 json 路径
        image_roots  : 额外的图片搜索根目录（列表）
        with_image   : False 时只返回标注（用于纯统计 / 评估）
        min_box_size : 过滤过小框（像素），默认 2
    """

    def __init__(
        self,
        coco_json: str | Path,
        image_roots: Sequence[str | Path] | None = None,
        with_image: bool = True,
        min_box_size: float = 2.0,
        auto_image_roots: bool = True,
    ) -> None:
        self.coco_json = Path(coco_json)
        self.coco = load_coco(self.coco_json)
        self.auto_image_roots = bool(auto_image_roots)
        roots = [Path(r) for r in (image_roots or [])]
        if self.auto_image_roots:
            for r in default_image_roots(self.coco_json):
                if r not in roots:
                    roots.append(r)
        self.image_roots = roots
        self.with_image = with_image
        self.min_box_size = float(min_box_size)

        self.images: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []
        self._missing_images: List[str] = []

        anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for a in self.coco["annotations"]:
            x, y, w, h = (float(v) for v in a["bbox"][:4])
            if w < self.min_box_size or h < self.min_box_size:
                continue
            anns_by_img.setdefault(int(a["image_id"]), []).append(a)

        for rec in self.coco["images"]:
            img_id = int(rec["id"])
            anns = anns_by_img.get(img_id, [])
            if not anns:
                continue  # 检测训练不使用无标注图（负样本单独处理）
            path = resolve_image_path(self.coco, rec, self.coco_json, extra_roots=self.image_roots)
            if path is None:
                self._missing_images.append(str(rec.get("file_name")))
                continue
            self.images.append(rec)
            self.records.append({"image": path, "image_id": img_id, "annotations": anns})

        if self._missing_images:
            print(
                f"[警告] {self.coco_json.name}: {len(self._missing_images)} 张图片未找到（示例："
                f"{self._missing_images[:3]}），已从数据集中剔除。\n"
                f"        已自动探测的图片根目录：{[str(r) for r in self.image_roots]}；"
                "如需补充请用 --image-root 指定。"
            )

    # ---- 容器协议 -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        out: Dict[str, Any] = {
            "image_id": int(r["image_id"]),
            "file_name": str(r["image"].name),
            "annotations": r["annotations"],
        }
        if self.with_image:
            try:
                from PIL import Image

                out["image"] = Image.open(r["image"]).convert("RGB")
            except Exception as exc:
                raise RuntimeError(f"读取图片失败：{r['image']}（{exc}）") from exc
        else:
            out["image"] = None
        return out

    # ---- 指标 / 统计 ----------------------------------------------------
    @property
    def num_classes(self) -> int:
        """数据集里出现过的最大 category_id + 1（应与 labels.yaml 的类别数一致）。"""
        cats = [int(c["id"]) for c in self.coco.get("categories", [])]
        return (max(cats) + 1) if cats else 0

    def category_names(self) -> List[str]:
        """类别 id -> 名称（按 id 排序）。"""
        cats = sorted(self.coco.get("categories", []), key=lambda c: int(c["id"]))
        return [str(c["name"]) for c in cats]

    def label_stats(self) -> Dict[str, Any]:
        """每类框数、每图平均框数等统计（训练前打印用）。"""
        per_cls: Dict[int, int] = {}
        for r in self.records:
            for a in r["annotations"]:
                cid = int(a["category_id"])
                per_cls[cid] = per_cls.get(cid, 0) + 1
        n_box = sum(per_cls.values())
        return {
            "num_images": len(self.records),
            "num_boxes": n_box,
            "avg_boxes": round(n_box / max(1, len(self.records)), 3),
            "per_class": dict(sorted(per_cls.items())),
            "missing_images": len(self._missing_images),
        }

    # ---- 各后端需要的目标格式 -------------------------------------------
    def to_hf_targets(self, idx: int, image_width: int, image_height: int) -> Dict[str, Any]:
        """转成 transformers RTDetrImageProcessor 期望的 target 格式。

        说明：``annotations`` 里如果是 ``box`` 字段（cxcywh 归一化）则直接使用，
        否则用 COCO 的 ``bbox``（xywh 像素）换算。这里的坐标统一取 **像素 xyxy**，
        交给 image processor 做缩放与归一化（processor 内部按 COCO 检测口径处理）。
        """
        r = self.records[idx]
        boxes: List[List[float]] = []
        classes: List[int] = []
        areas: List[float] = []
        for a in r["annotations"]:
            if "box" in a:  # 已是归一化 cxcywh
                cx, cy, w, h = (float(v) for v in a["box"][:4])
                x1 = (cx - w / 2.0) * image_width
                y1 = (cy - h / 2.0) * image_height
                x2 = (cx + w / 2.0) * image_width
                y2 = (cy + h / 2.0) * image_height
            else:
                x1, y1, x2, y2 = xywh_to_xyxy(a["bbox"])
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([x1, y1, x2, y2])
            classes.append(int(a["category_id"]))
            areas.append((x2 - x1) * (y2 - y1))
        return {
            "image_id": int(r["image_id"]),
            "class_labels": classes,
            "boxes": boxes,
            "area": areas,
            "orig_size": [int(image_height), int(image_width)],
        }

    def to_yolo_lines(self, idx: int, names: Sequence[str] | None = None) -> List[str]:
        """按 YOLO 归一化格式导出该图标注（ultralytics / YOLO-World 分支用）。"""
        from src.data.io import coco_to_yolo_lines

        r = self.records[idx]
        rec = dict(self.images[idx])
        rec["width"] = rec.get("width") or 1
        rec["height"] = rec.get("height") or 1
        sub = {
            "images": [rec],
            "annotations": r["annotations"],
            "categories": self.coco.get("categories", []),
        }
        return coco_to_yolo_lines(sub, rec, names=names)

    def export_yolo_dataset(self, out_dir: str | Path, names: Sequence[str], copy_images: bool = True) -> Path:
        """把本数据集落成 YOLO 目录结构（``images/`` + ``labels/`` + ``data.yaml``）。

        供 ``--backend ultralytics-rtdetr`` 与 YOLO-World 分支使用。
        """
        import shutil

        out = Path(out_dir)
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(len(self.records)):
            r = self.records[i]
            if copy_images:
                shutil.copy2(r["image"], out / "images" / r["image"].name)
            lines = self.to_yolo_lines(i, names=names)
            (out / "labels" / f"{r['image'].stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        yaml_text = (
            f"path: {out.resolve().as_posix()}\n"
            "train: images\n"
            "val: images\n"
            f"nc: {len(names)}\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
        )
        (out / "data.yaml").write_text(yaml_text, encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# 分类数据集
# ---------------------------------------------------------------------------
class ImageClassifyDataset:
    """病害分类数据集：把 COCO 的检测框裁成分类样本，或直接读分类文件夹。

    样本来源两种：
        1. ``from_coco(coco_json)``   —— 每个检测框裁一个 patch，标签为框的类别
           （``--crop-margin`` 可外扩，缓解病斑贴边）；
        2. ``from_folder(root)``      —— ``root/<class_name>/*.jpg``，类名映射到 labels.yaml。
    """

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        image_size: int = 224,
        crop_margin: float = 0.1,
        train: bool = False,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.crop_margin = float(crop_margin)
        self.train = bool(train)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import numpy as np
        import torch

        s = self.samples[idx]
        pil = self._read_pil(s["image_path"], s.get("bbox"))
        arr = self._to_tensor(pil, np, torch)
        return {"pixel_values": arr, "label": torch.tensor(int(s["label"]), dtype=torch.long),
                "image_path": str(s["image_path"])}

    # ---- 内部工具 -------------------------------------------------------
    def _read_pil(self, path: Path, bbox: Optional[Sequence[float]]):
        """读图 + 可选裁剪 + Resize（resample 用 PIL 常量，避免依赖 torchvision）。"""
        from PIL import Image

        im = Image.open(path).convert("RGB")
        if bbox is not None:
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            mx = (x2 - x1) * self.crop_margin
            my = (y2 - y1) * self.crop_margin
            x1 = max(0.0, x1 - mx)
            y1 = max(0.0, y1 - my)
            x2 = min(float(im.width), x2 + mx)
            y2 = min(float(im.height), y2 + my)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                im = im.crop((int(x1), int(y1), int(math.ceil(x2)), int(math.ceil(y2))))
        resample = Image.BICUBIC
        if self.train:
            # 训练态做最小增广：随机水平翻转，避免引入 albumentations 依赖
            import random

            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return im.resize((self.image_size, self.image_size), resample)

    def _to_tensor(self, pil, np, torch):
        """PIL -> 归一化 CHW float tensor。"""
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(arr))

    # ---- 构造器 ---------------------------------------------------------
    @classmethod
    def from_coco(
        cls,
        coco_json: str | Path,
        image_roots: Sequence[str | Path] | None = None,
        image_size: int = 224,
        crop_margin: float = 0.1,
        train: bool = False,
        exclude_class_ids: Sequence[int] = (),
        full_image_for_healthy: bool = True,
        max_samples: int = 0,
        auto_image_roots: bool = True,
    ) -> "ImageClassifyDataset":
        """从 COCO 标注生成分类样本（每个检测框一个 patch）。

        参数:
            exclude_class_ids     : 不参与分类的 class_id
            full_image_for_healthy: healthy 类（整图无病）用整图裁剪，而不是病斑小框
            max_samples           : >0 时按类别均匀截断到该数量，控制训练时长
            auto_image_roots      : 自动探测 ``data/raw`` / ``data/samples`` 等图片根目录
        """
        ds = RTDetrCocoDataset(coco_json, image_roots=image_roots, with_image=False,
                               auto_image_roots=auto_image_roots)
        exc = set(int(c) for c in exclude_class_ids)
        healthy_id = 0
        samples: List[Dict[str, Any]] = []
        for r in ds.records:
            for a in r["annotations"]:
                cid = int(a["category_id"])
                if cid in exc:
                    continue
                bbox = None
                if not (full_image_for_healthy and cid == healthy_id):
                    bbox = xywh_to_xyxy(a["bbox"])
                samples.append({"image_path": r["image"], "bbox": bbox, "label": cid})

        if max_samples and max_samples > 0:
            from collections import defaultdict

            by: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for s in samples:
                by[int(s["label"])].append(s)
            capped: List[Dict[str, Any]] = []
            for k in sorted(by):
                capped.extend(by[k][: int(max_samples)])
            samples = capped

        if not samples:
            raise RuntimeError(f"{coco_json} 中没有可用的分类样本（都是空标注？）")
        return cls(samples, image_size=image_size, crop_margin=crop_margin, train=train)

    @classmethod
    def from_folder(
        cls,
        root: str | Path,
        name_to_id: Dict[str, int],
        image_size: int = 224,
        train: bool = False,
    ) -> "ImageClassifyDataset":
        """从 ``root/<class_name>/*.jpg`` 生成分类样本；类名需能映射到 labels.yaml。"""
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"分类数据目录不存在：{root}")
        samples: List[Dict[str, Any]] = []
        unknown: List[str] = []
        for sub in sorted(d for d in root.iterdir() if d.is_dir()):
            key = sub.name.lower().replace(" ", "_").replace("-", "_")
            if key not in name_to_id and sub.name not in name_to_id:
                unknown.append(sub.name)
                continue
            cid = int(name_to_id.get(key, name_to_id.get(sub.name)))
            for p in sorted(x for x in sub.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTS):
                samples.append({"image_path": p, "bbox": None, "label": cid})
        if unknown:
            print(f"[警告] 以下分类目录名不在 labels.yaml 中，已跳过：{unknown}")
        if not samples:
            raise RuntimeError(f"{root} 下没有找到可用的分类样本")
        return cls(samples, image_size=image_size, train=train)

    def class_counts(self) -> Dict[int, int]:
        """每类样本数（打印/回调用）。"""
        counts: Dict[int, int] = {}
        for s in self.samples:
            counts[int(s["label"])] = counts.get(int(s["label"]), 0) + 1
        return dict(sorted(counts.items()))

    def class_weights(self) -> List[float]:
        """按类别频率的倒数给交叉熵权重（缓解长尾）。"""
        counts = self.class_counts()
        if not counts:
            return []
        n_cls = max(counts) + 1
        total = sum(counts.values())
        weights = [0.0] * n_cls
        for cid in range(n_cls):
            c = counts.get(cid, 0)
            weights[cid] = 0.0 if c == 0 else round(total / (len(counts) * c), 6)
        return weights


# ---------------------------------------------------------------------------
# 推理 / 自动标注用数据集
# ---------------------------------------------------------------------------
class SimpleImageFolder:
    """只遍历图片的 Dataset（自动标注、伪标签、可视化都用它）。"""

    def __init__(self, root: str | Path, recursive: bool = True) -> None:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"图片目录不存在：{root}")
        it = root.rglob("*") if recursive else root.glob("*")
        self.paths = sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> str:
        return str(self.paths[idx])


def _self_check() -> int:  # pragma: no cover - 手工调试用
    """命令行自检：``python -m src.data.torch_datasets <coco.json>``。"""
    if len(sys.argv) < 2:
        print("用法: python -m src.data.torch_datasets <coco.json> [image_root]")
        return 2
    ds = RTDetrCocoDataset(sys.argv[1], image_roots=sys.argv[2:])
    print(json.dumps(ds.label_stats(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_self_check())
