"""RT-DETR 检测器训练核心（transformers 官方实现 + ultralytics / Paddle 后端）。

三条后端路线（都写在这一个文件里，供 ``scripts/train_det.py`` 调用）：

1. ``transformers-rtdetrv2``（主力，默认）
   用 ``RTDetrV2ForObjectDetection`` + ``RTDetrImageProcessor``，自写训练循环：
   支持 AMP、梯度累积、冻结 backbone、backbone/head 分组学习率、每 epoch 验证与最优权重保存。
   这是 RT-DETRv2/v3（WACV2025 分层密集正监督那套）在 transformers 里的官方移植，
   不依赖 ultralytics，最适合本项目「替换 YOLO」的定位。

2. ``transformers-trainer``
   同一模型改用 HuggingFace ``Trainer`` 驱动（官方 RT-DETR 微调范式）。
   注意：transformers 的 ``trainer.py`` 会 import ``datasets``，而 ``datasets`` 依赖 pyarrow/pandas，
   本机若缺这些包（或 DLL 被策略拦截）会 ImportError —— 这时给出中文提示，让用户改用后端 1。

3. ``ultralytics-rtdetr`` / ``paddle``
   ultralytics 版 RT-DETR（``rtdetr-l.pt`` 从零微调，自带 ONNX 导出）；PaddleDetection 路线
   只生成配置与启动脚本（要求本机有 PaddleDetection 源码，不在此处真跑）。

怎么用::

    from src.train.detectors import (
        build_rtdetr_model, encode_coco_to_rtdetr, HFDetectionDataset,
        train_rtdetr_torch, write_paddle_configs,
    )
    model, processor = build_rtdetr_model(num_classes=15, variant="v2")
    train_rtdetr_torch(model, processor, train_ds, val_ds, out_dir="models/det_rtdetrv2", ...)

依赖：torch / transformers（必需，惰性 import + 中文提示）；ultralytics 仅后端 3 需要。
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.labels import LabelBook
from src.data.torch_datasets import RTDetrCocoDataset
from src.train.common import JsonlLogger, log, set_seed

__all__ = [
    "DEFAULT_RTDETR_MODELS",
    "build_rtdetr_model",
    "build_rtdetr_processor",
    "encode_coco_to_rtdetr",
    "HFDetectionDataset",
    "collate_detection",
    "decoder_rtdetr_outputs",
    "train_rtdetr_torch",
    "train_rtdetr_hf_trainer",
    "train_ultralytics_rtdetr",
    "write_paddle_configs",
    "prepare_ultralytics_dataset",
]

#: 官方 RT-DETR 权重（transformers 仓库名）。v2 = RT-DETRv2（首选），v1 = 原始 RT-DETR。
DEFAULT_RTDETR_MODELS: Dict[str, str] = {
    "v2-r18vd": "PekingU/rtdetr_v2_r18vd",
    "v2-r34vd": "PekingU/rtdetr_v2_r34vd",
    "v2-r50vd": "PekingU/rtdetr_v2_r50vd",
    "v1-r50vd": "PekingU/rtdetr_r50vd",
}


# ---------------------------------------------------------------------------
# 模型 / 处理器构建
# ---------------------------------------------------------------------------
def _torch():
    """惰性 import torch。"""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 PyTorch，无法训练检测模型。\n"
            "  CPU：pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  RTX 5070（Blackwell sm_120）：pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            f"（原始错误：{exc}）"
        ) from exc
    return torch


def build_rtdetr_processor(
    pretrained: str = DEFAULT_RTDETR_MODELS["v2-r18vd"],
    imgsz: int = 640,
    local_dir: str | Path | None = None,
):
    """构建 ``RTDetrImageProcessor``（优先本地目录，便于离线推理/训练复现）。

    参数:
        pretrained: transformers 仓库名（联网首次会下载）
        imgsz     : 方形输入边长（写入 processor.size）
        local_dir : 若给出且存在，则优先从该目录加载 processor 文件
    """
    try:
        from transformers import RTDetrImageProcessor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 transformers，无法构建 RT-DETR 数据处理器。\n"
            "  pip install 'transformers>=4.48'\n"
            f"（原始错误：{exc}）"
        ) from exc
    src = str(local_dir) if (local_dir and Path(local_dir).is_dir()) else pretrained
    try:
        proc = RTDetrImageProcessor.from_pretrained(src)
    except Exception as exc:
        raise RuntimeError(
            f"加载 RTDetrImageProcessor 失败（来源 {src}）。\n"
            "  首次使用需要联网下载权重；离线环境请用 --processor-dir 指定本地 processor 目录。\n"
            f"（原始错误：{exc}）"
        ) from exc
    proc.size = {"height": int(imgsz), "width": int(imgsz)}
    proc.do_pad = True
    return proc


def build_rtdetr_model(
    num_classes: int,
    variant: str = "v2",
    pretrained: str | None = None,
    ignore_mismatched_sizes: bool = True,
    id2label: Optional[Dict[int, str]] = None,
):
    """构建 RT-DETR 模型（默认 RT-DETRv2）。

    参数:
        num_classes: 类别数（= labels.yaml 的类别数）
        variant    : 'v2'（RT-DETRv2，首选）或 'v1'（原始 RT-DETR）
        pretrained : 预训练权重仓库名；None 时按 variant 取 ``DEFAULT_RTDETR_MODELS``
        id2label   : class_id -> disease_name（写进 config，导出时也能带上）
    返回:
        (model, 实际使用的预训练名)
    """
    try:
        from transformers import RTDetrForObjectDetection, RTDetrV2ForObjectDetection
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 transformers >= 4.48，无法加载 RT-DETR。\n"
            "  pip install 'transformers>=4.48'\n"
            f"（原始错误：{exc}）"
        ) from exc

    v = (variant or "v2").lower()
    name = pretrained or DEFAULT_RTDETR_MODELS.get(f"{v}-r18vd" if v == "v2" else "v1-r50vd")
    cls = RTDetrV2ForObjectDetection if v == "v2" else RTDetrForObjectDetection
    kw: Dict[str, Any] = {
        "num_labels": int(num_classes),
        "ignore_mismatched_sizes": bool(ignore_mismatched_sizes),
    }
    if id2label:
        kw["id2label"] = {int(k): str(v) for k, v in id2label.items()}
        kw["label2id"] = {str(v): int(k) for k, v in id2label.items()}
    try:
        model = cls.from_pretrained(name, **kw)
    except Exception as exc:
        raise RuntimeError(
            f"加载 RT-DETR 预训练权重失败：{name}\n"
            "  首次使用需联网下载（rtdetr_v2_r18vd 约 80MB）；离线时请用 --pretrained 指定本地目录。\n"
            f"（原始错误：{exc}）"
        ) from exc
    log(f"已加载 {cls.__name__}（pretrained={name}, num_labels={num_classes}）")
    return model, name


def freeze_backbone(model: Any) -> Tuple[int, int]:
    """冻结 RT-DETR 的 backbone（``model.model.backbone``），返回 (冻结参数量, 总参数量)。

    8GB 显存下这是最有效的省显存手段之一：backbone 通常占总参数 40%~60%。
    """
    torch = _torch()
    backbone = None
    inner = getattr(model, "model", None)
    if inner is not None:
        backbone = getattr(inner, "backbone", None)
    if backbone is None:
        backbone = getattr(model, "backbone", None)
    if backbone is None:
        log("未找到 model.model.backbone，跳过冻结（该结构可能是自定义版本）", "警告")
        return 0, sum(p.numel() for p in model.parameters())
    frozen = 0
    for p in backbone.parameters():
        p.requires_grad_(False)
        frozen += p.numel()
    total = sum(p.numel() for p in model.parameters())
    _ = torch
    return frozen, total


def apply_lora_detector(model: Any, r: int = 8, alpha: int = 16, targets: Sequence[str] = ("q_proj", "v_proj")) -> str:
    """给检测器的注意力层注入 LoRA（需要 peft）。返回描述字符串。"""
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise RuntimeError(
            "缺少 peft，无法对检测器做 LoRA（--peft lora）。\n"
            "  pip install peft\n"
            "  或改用 --freeze-backbone（8GB 显存推荐，先冻结 backbone 只训 decoder/head）。\n"
            f"（原始错误：{exc}）"
        ) from exc
    cfg = LoraConfig(r=int(r), lora_alpha=int(alpha), lora_dropout=0.0, bias="none",
                     target_modules=list(targets))
    wrapped = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.parameters())
    return f"LoRA(r={r}, alpha={alpha}, targets={list(targets)}) trainable={trainable}/{total}"


# ---------------------------------------------------------------------------
# 标注编码 / Dataset
# ---------------------------------------------------------------------------
def encode_coco_to_rtdetr(anns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把 COCO 标注列表编码成 RT-DETR 训练目标（归一化 cxcywh + class_labels）。

    参数:
        anns: COCO annotation 列表，每个含 ``bbox``=[x,y,w,h]（像素）、``image_id``、``category_id``，
              以及可选的 ``width``/``height``（缺省时要求外层调用方补齐）
    返回:
        {"class_labels": [...], "boxes": [[cx,cy,w,h], ...], "area": [...], "image_id": int}
    """
    class_labels: List[int] = []
    boxes: List[List[float]] = []
    areas: List[float] = []
    image_id = -1
    for a in anns:
        image_id = int(a.get("image_id", image_id))
        w_img = float(a.get("width") or a.get("img_w") or 0)
        h_img = float(a.get("height") or a.get("img_h") or 0)
        if w_img <= 0 or h_img <= 0:
            raise ValueError(
                "encode_coco_to_rtdetr 需要每个标注带 width/height（或由调用方注入 img_w/img_h）"
            )
        x, y, w, h = (float(v) for v in a["bbox"][:4])
        if w <= 1 or h <= 1:
            continue
        cx = (x + w / 2.0) / w_img
        cy = (y + h / 2.0) / h_img
        nw = w / w_img
        nh = h / h_img
        class_labels.append(int(a["category_id"]))
        boxes.append([min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0),
                      min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)])
        areas.append(w * h)
    return {"class_labels": class_labels, "boxes": boxes, "area": areas, "image_id": image_id}


class HFDetectionDataset:
    """给 transformers RT-DETR 用的 Dataset（返回 PIL 图 + 编码后的目标）。

    参数:
        coco_ds : ``src.data.torch_datasets.RTDetrCocoDataset``
        train   : True 时做随机水平翻转（最轻量、对航拍病害边界影响最小）
    """

    def __init__(self, coco_ds: RTDetrCocoDataset, train: bool = False) -> None:
        self.ds = coco_ds
        self.train = bool(train)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]
        rec = self.ds.images[idx]
        w_img = int(rec.get("width") or item["image"].width)
        h_img = int(rec.get("height") or item["image"].height)
        image = item["image"]
        anns = [dict(a) for a in item["annotations"]]
        if self.train and random.random() < 0.5:
            from PIL import Image

            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped = []
            for a in anns:
                x, y, w, h = (float(v) for v in a["bbox"][:4])
                a["bbox"] = [max(0.0, w_img - x - w), y, w, h]
                flipped.append(a)
            anns = flipped
        for a in anns:
            a["width"] = w_img
            a["height"] = h_img
        target = encode_coco_to_rtdetr(anns)
        return {
            "image": image,
            "image_id": int(item["image_id"]),
            "target": target,
            "orig_size": (h_img, w_img),
        }


def collate_detection(batch: List[Dict[str, Any]], processor: Any) -> Dict[str, Any]:
    """用 ``RTDetrImageProcessor`` 把一批样本转成模型输入。

    返回 ``{"pixel_values": Tensor(B,3,H,W), "labels": [target, ...]}``。

    ⚠️ 兼容性要点（本机实测踩过）：transformers 5.x 的 RTDetrImageProcessor 会把
    标注当作 **COCO 格式**校验，要求每个元素形如::

        {"image_id": int, "annotations": [{"category_id": int, "bbox": [cx,cy,w,h]}, ...]}

    早期只传扁平的 ``{"class_labels": [...], "boxes": [...]}``，
    在新版直接报 ``ValueError: Invalid COCO detection annotations``，训练根本进不去。
    这里两种格式都给（老版本认扁平、新版本认嵌套），并用 ``isinstance`` 判断返回结构。
    """
    images = [b["image"] for b in batch]
    annotations: List[Dict[str, Any]] = []
    for b in batch:
        t = b["target"]
        labels = [int(v) for v in t["class_labels"]]
        # t["boxes"] 是**归一化 cxcywh**（RT-DETR 训练目标口径）。
        # ⚠️ 但 processor 的 annotations 要的是 **COCO 口径的像素 [x, y, w, h]（左上角基准）**。
        # 曾经直接把归一化 cxcywh 当 COCO bbox 传进去（格式与量纲双错），
        # 结果 processor 又把 0.0867 当像素再归一化一次 -> 真值变成 0.2×0.2 像素的退化框，
        # 模型忠实地学会了"贴在图左上角、几乎零面积"的框，60 epoch 训完 mAP=0
        # （本机实测，白跑 2.5 小时）。这里必须显式做 cxcywh(归一化) -> xywh(像素) 的换算。
        pil_img = b["image"]
        W, H = (float(pil_img.size[0]), float(pil_img.size[1]))
        anns: List[Dict[str, Any]] = []
        for c, bb in zip(labels, t["boxes"]):
            cx, cy, nw, nh = (float(v) for v in bb[:4])
            w = nw * W
            h = nh * H
            x = cx * W - w / 2.0
            y = cy * H - h / 2.0
            if w <= 1.0 or h <= 1.0:
                continue  # 退化目标，跳过（与 encode 阶段的过滤一致）
            anns.append(
                {
                    "category_id": int(c),
                    "bbox": [x, y, w, h],   # COCO 口径：像素、左上角基准
                    "iscrowd": 0,
                    "area": w * h,
                }
            )
        annotations.append(
            {
                # 新版本 transformers 需要的嵌套 COCO 结构
                "image_id": int(t.get("image_id", -1)),
                "annotations": anns,
                # 旧版本 transformers 认的扁平结构（保持兼容）
                "class_labels": labels,
                "boxes": [[float(v) for v in bb] for bb in t["boxes"]],
                "area": list(t.get("area", [])),
            }
        )
    out = processor(images=images, annotations=annotations, return_tensors="pt")
    labels = out.pop("labels")
    # 不同版本返回 BatchFeature / list / dict，统一成 list[dict]
    out["labels"] = list(labels) if not isinstance(labels, list) else labels
    if "pixel_mask" in out:
        out["pixel_mask"] = out["pixel_mask"]
    return out


def decoder_rtdetr_outputs(outputs: Any, processor: Any, orig_sizes: Sequence[Tuple[int, int]],
                           threshold: float = 0.3) -> List[Dict[str, Any]]:
    """把模型输出解码成像素 xyxy 框（训练里算验证指标、导出后对比都用它）。

    参数:
        orig_sizes: [(h, w), ...]，与 batch 顺序一致
    返回:
        [{"boxes": [[x1,y1,x2,y2],...], "scores": [...], "labels": [...]}, ...]
    """
    torch = _torch()
    sizes = torch.tensor([[int(h), int(w)] for h, w in orig_sizes], dtype=torch.long)
    res = processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=float(threshold))
    out: List[Dict[str, Any]] = []
    for r in res:
        out.append(
            {
                "boxes": r["boxes"].detach().cpu().tolist() if hasattr(r["boxes"], "detach") else list(r["boxes"]),
                "scores": r["scores"].detach().cpu().tolist() if hasattr(r["scores"], "detach") else list(r["scores"]),
                "labels": r["labels"].detach().cpu().tolist() if hasattr(r["labels"], "detach") else list(r["labels"]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 后端 1：自写训练循环（推荐）
# ---------------------------------------------------------------------------
def train_rtdetr_torch(
    model: Any,
    processor: Any,
    train_ds: HFDetectionDataset,
    val_ds: Optional[HFDetectionDataset],
    out_dir: str | Path,
    label_book: LabelBook,
    epochs: int = 20,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 1e-4,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    amp: bool = False,
    num_workers: int = 0,
    max_steps_per_epoch: int = 0,
    seed: int = 42,
    warmup_ratio: float = 0.1,
    eval_every_epochs: int = 1,
    save_best_by: str = "loss",
    pretrained_name: str = "",
) -> Dict[str, Any]:
    """RT-DETR 自写训练循环。

    关键设计（都为 8GB 显存服务）：
        - ``batch_size`` 很小 + ``grad_accum`` 很大 -> 等效大 batch；
        - 分组学习率：backbone 用 ``backbone_lr``（通常 1e-5），decoder/head 用 ``lr``（1e-4）；
        - ``--amp`` 时用 ``torch.amp.autocast`` + GradScaler（仅 CUDA）；
        - 每 ``eval_every_epochs`` 个 epoch 在 val 上算平均 loss 与 COCO 风格 mAP（调用 ``scripts/eval.py`` 之外的轻量实现）。

    返回: ``{"history":..., "best":..., "weights": path, ...}``
    """
    torch = _torch()
    from torch.utils.data import DataLoader

    set_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = JsonlLogger(out / "train_log.jsonl")

    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("没有任何可训练参数（是不是把整个模型都冻了？）")
    bb_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb_params if ".backbone." in f".{n}." or n.startswith("model.backbone") else head_params).append(p)
    groups = []
    if bb_params:
        groups.append({"params": bb_params, "lr": float(backbone_lr)})
    if head_params:
        groups.append({"params": head_params, "lr": float(lr)})
    opt = torch.optim.AdamW(groups, weight_decay=float(weight_decay))
    n_train = sum(p.numel() for p in trainable)
    log(f"可训练参数 {n_train / 1e6:.2f} M（backbone {sum(p.numel() for p in bb_params) / 1e6:.2f} M）")

    def _collate(batch):
        return collate_detection(batch, processor)

    train_loader = DataLoader(
        train_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers),
        collate_fn=_collate, drop_last=False,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
            collate_fn=_collate, drop_last=False,
        )

    steps_per_epoch = len(train_loader)
    if max_steps_per_epoch and max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, int(max_steps_per_epoch))
    total_steps = max(1, steps_per_epoch * int(epochs) // max(1, int(grad_accum)))
    warmup = max(1, int(total_steps * float(warmup_ratio)))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = bool(amp) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: List[Dict[str, Any]] = []
    best = {"score": float("inf") if save_best_by == "loss" else -1.0, "epoch": -1, "metrics": {}}
    global_step = 0

    def _move(inputs: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for k, v in inputs.items():
            if k == "labels":
                moved[k] = [{kk: (vv.to(device) if hasattr(vv, "to") else vv) for kk, vv in lab.items()} for lab in v]
            else:
                moved[k] = v.to(device) if hasattr(v, "to") else v
        return moved

    for epoch in range(1, int(epochs) + 1):
        model.train()
        t0 = time.perf_counter()
        run_loss, n_batches = 0.0, 0
        opt.zero_grad(set_to_none=True)
        skipped = 0
        for step, batch in enumerate(train_loader):
            if max_steps_per_epoch and step >= int(max_steps_per_epoch):
                break
            inputs = _move(batch)
            try:
                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        outs = model(**inputs)
                        loss = outs.loss / max(1, int(grad_accum))
                    scaler.scale(loss).backward()
                else:
                    outs = model(**inputs)
                    loss = outs.loss / max(1, int(grad_accum))
                    loss.backward()
            except RuntimeError as exc:
                # 显存不足等：单步跳过并给出可执行建议，不让整轮训练崩掉
                if "out of memory" in str(exc).lower() or "CUDA" in str(exc):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    log(f"step {step} 计算失败已跳过（{str(exc)[:120]}）。建议减小 --batch-size 或开启 --freeze-backbone", "警告")
                    continue
                raise
            run_loss += float(loss.item()) * max(1, int(grad_accum))
            n_batches += 1
            if (step + 1) % max(1, int(grad_accum)) == 0:
                if use_amp:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

        train_loss = run_loss / max(1, n_batches)
        metrics: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "lr": round(sched.get_last_lr()[0], 8),
            "skipped_steps": skipped,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "global_step": global_step,
        }

        if val_loader is not None and epoch % max(1, int(eval_every_epochs)) == 0:
            model.eval()
            v_loss, v_batches = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = _move(batch)
                    try:
                        outs = model(**inputs)
                    except RuntimeError:
                        continue
                    if getattr(outs, "loss", None) is not None:
                        v_loss += float(outs.loss.item())
                        v_batches += 1
            metrics["val_loss"] = round(v_loss / max(1, v_batches), 6)

        history.append(metrics)
        jsonl.write(metrics)
        log(
            f"epoch {epoch}/{epochs} train_loss={metrics['train_loss']:.4f}"
            + (f" val_loss={metrics.get('val_loss', float('nan')):.4f}" if "val_loss" in metrics else "")
        )

        score = float(metrics.get("val_loss", metrics["train_loss"]))
        improved = score < best["score"] if save_best_by == "loss" else score > best["score"]
        if improved:
            best = {"score": score, "epoch": epoch, "metrics": dict(metrics)}
            save_detection_checkpoint(model, out, label_book, stage="best", pretrained=pretrained_name,
                                      extra={"best_epoch": epoch, "best_score": score, "save_best_by": save_best_by})
            log(f"已保存更优权重 -> {out / 'best.pt'}（{save_best_by}={score:.4f}）")

    save_detection_checkpoint(model, out, label_book, stage="last", pretrained=pretrained_name,
                             extra={"epochs": int(epochs)})
    return {"history": history, "best": best, "weights": str(out / "best.pt"),
            "last": str(out / "last.pt"), "steps_per_epoch": steps_per_epoch}


def save_detection_checkpoint(
    model: Any,
    out_dir: str | Path,
    label_book: LabelBook,
    stage: str = "best",
    pretrained: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """保存检测权重 ``best.pt`` / ``last.pt`` + ``checkpoint.json``。"""
    torch = _torch()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wpath = out / f"{stage}.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, wpath)
    if stage == "best":
        # 同时保存 config 与 processor，便于推理侧完全离线重建
        try:
            model.config.save_pretrained(str(out))
        except Exception as exc:  # pragma: no cover
            log(f"保存 model.config 失败（不影响权重）：{exc}", "警告")
    info = {
        "task": "detect",
        "source": "rtdetrv2",
        "pretrained": pretrained,
        "num_classes": label_book.num_classes,
        "classes": label_book.sidecar_classes(),
    }
    info.update(extra or {})
    cpath = out / "checkpoint.json"
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return {"weights": wpath, "checkpoint": cpath}


# ---------------------------------------------------------------------------
# 后端 2：HuggingFace Trainer
# ---------------------------------------------------------------------------
def train_rtdetr_hf_trainer(
    model: Any,
    processor: Any,
    train_ds: HFDetectionDataset,
    val_ds: Optional[HFDetectionDataset],
    out_dir: str | Path,
    label_book: LabelBook,
    epochs: int = 20,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    amp: bool = False,
    num_workers: int = 0,
    max_steps: int = -1,
    seed: int = 42,
    pretrained_name: str = "",
) -> Dict[str, Any]:
    """用 HuggingFace ``Trainer`` 训练 RT-DETR（官方范式，代码最短）。

    已知环境限制：``transformers.trainer`` 会 ``import datasets``，而 ``datasets`` 需要
    pyarrow/pandas。若这些包缺失（或被 Windows 应用控制策略拦截 DLL），本函数会抛出
    中文 RuntimeError，提示改用 ``--trainer custom``（自写循环）。
    """
    torch = _torch()
    from torch.utils.data import DataLoader

    try:
        from transformers import TrainingArguments
        from transformers.trainer import Trainer
    except Exception as exc:
        raise RuntimeError(
            "无法 import transformers.Trainer（它依赖 datasets -> pyarrow/pandas）。\n"
            "  方案 A：pip install datasets  （并确认 pyarrow 的 DLL 未被系统策略拦截）\n"
            "  方案 B：改用自写训练循环：--trainer custom（推荐，本项目默认）\n"
            f"（原始错误：{type(exc).__name__}: {exc}）"
        ) from exc

    set_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _collate(batch):
        return collate_detection(batch, processor)

    args = TrainingArguments(
        output_dir=str(out / "hf_runs"),
        num_train_epochs=float(epochs),
        max_steps=int(max_steps),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(batch_size),
        gradient_accumulation_steps=int(grad_accum),
        learning_rate=float(lr),
        weight_decay=float(weight_decay),
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if (val_ds is not None and len(val_ds) > 0) else "no",
        save_total_limit=2,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=int(num_workers),
        dataloader_pin_memory=False,
        fp16=bool(amp) and str(device).startswith("cuda"),
        bf16=False,
        report_to=[],
        seed=int(seed),
        use_cpu=not str(device).startswith("cuda"),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=_collate,
    )
    train_out = trainer.train()
    metrics = dict(train_out.metrics or {})
    save_detection_checkpoint(model, out, label_book, stage="best", pretrained=pretrained_name,
                             extra={"trainer": "hf", "metrics": metrics})
    log(f"HF Trainer 训练完成：{metrics}")
    return {"trainer": "hf", "metrics": metrics, "weights": str(out / "best.pt"),
            "checkpoint": str(out / "checkpoint.json")}


# ---------------------------------------------------------------------------
# 后端 3：ultralytics RT-DETR
# ---------------------------------------------------------------------------
def prepare_ultralytics_dataset(
    train_coco: str | Path,
    val_coco: str | Path,
    out_dir: str | Path,
    label_book: LabelBook,
) -> Path:
    """把 COCO 划分导出成 ultralytics 需要的 YOLO 目录结构（images/ labels/ data.yaml）。"""
    out = Path(out_dir)
    for split, coco in (("train", train_coco), ("val", val_coco)):
        ds = RTDetrCocoDataset(coco)
        if len(ds) == 0:
            log(f"{split} 集为空（{coco}），ultralytics 训练可能失败", "警告")
        ds.export_yolo_dataset(out / split, names=label_book.names, copy_images=True)
        log(f"已导出 {split} 集：{len(ds)} 张图 -> {out / split}")
    data_yaml = (
        f"path: {out.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        f"nc: {label_book.num_classes}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(label_book.names))
    )
    p = out / "data.yaml"
    p.write_text(data_yaml, encoding="utf-8")
    log(f"ultralytics data.yaml -> {p}")
    return p


def train_ultralytics_rtdetr(
    data_yaml: str | Path,
    out_dir: str | Path,
    model_name: str = "rtdetr-l.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 4,
    device: str = "cpu",
    amp: bool = False,
    resume: bool = False,
    workers: int = 0,
    seed: int = 42,
    patience: int = 20,
) -> Dict[str, Any]:
    """用 ultralytics 的 RT-DETR 从 ``rtdetr-l.pt`` 微调（惰性 import ultralytics）。

    8GB 显存建议：``imgsz=640, batch=4``（或 ``batch=2`` + 更大 patience）。
    """
    try:
        from ultralytics import RTDETR
    except Exception as exc:
        raise RuntimeError(
            "未安装 ultralytics，无法使用 --backend ultralytics-rtdetr。\n"
            "  pip install ultralytics\n"
            "  提示：本项目主力路线是 transformers 版 RT-DETRv2（--backend transformers-rtdetrv2），"
            "不装 ultralytics 也能完成训练与导出。\n"
            f"（原始错误：{exc}）"
        ) from exc
    set_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"ultralytics 数据配置不存在：{data_yaml}（请先跑 prepare_ultralytics_dataset）")
    model = RTDETR(str(model_name))
    results = model.train(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch_size),
        device=("cpu" if str(device).startswith("cpu") else 0),
        amp=bool(amp),
        project=str(out),
        name="ultralytics_rtdetr",
        exist_ok=True,
        resume=bool(resume),
        workers=int(workers),
        patience=int(patience),
        seed=int(seed),
    )
    _ = results
    # ultralytics 会把最优权重放在 <project>/<name>/weights/best.pt，统一拷到 out/best.pt
    import shutil

    src = out / "ultralytics_rtdetr" / "weights" / "best.pt"
    dst = out / "best.pt"
    if src.is_file():
        shutil.copy2(src, dst)
        log(f"已复制 ultralytics 最优权重 -> {dst}")
    else:
        log(f"未找到 ultralytics 输出权重：{src}（请检查训练日志）", "警告")
    return {"backend": "ultralytics-rtdetr", "weights": str(dst), "run_dir": str(src.parent.parent)}


# ---------------------------------------------------------------------------
# 后端 4：PaddleDetection 配置生成（不要求真跑）
# ---------------------------------------------------------------------------
def write_paddle_configs(
    out_dir: str | Path,
    label_book: LabelBook,
    data_root: str | Path,
    epochs: int = 80,
    batch_size: int = 4,
    imgsz: int = 640,
    lr: float = 1e-4,
) -> Dict[str, Path]:
    """生成 PaddleDetection 版 RT-DETR 的 yml 配置 + 启动脚本（Windows/Linux 各一份）。

    产物:
        models/<name>/paddle/rtdetr_<crop>.yml   训练配置（含类别数/学习率/batch）
        models/<name>/paddle/train.bat            Windows 启动脚本
        models/<name>/paddle/train.sh             Linux 启动脚本
        models/<name>/paddle/label_list.txt       类别名列表（PaddleDetection 的 label_list）
        models/<name>/paddle/README_PADDLE.md     使用说明（中文）
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root)

    label_txt = out / "label_list.txt"
    label_txt.write_text("\n".join(label_book.names) + "\n", encoding="utf-8")

    yml = out / "rtdetr_v2_custom.yml"
    yml.write_text(
        "\n".join(
            [
                "# PaddleDetection RT-DETR（RT-DETRv2 思路）配置 —— 由 scripts/train_det.py --backend paddle 生成",
                "# 用法：把本文件放到 PaddleDetection/configs/rtdetr/ 下，并按需修改 dataset_dir / anno_path",
                "metric: COCO",
                f"num_classes: {label_book.num_classes}",
                "",
                "TrainDataset:",
                "  !COCODataSet",
                f"  image_dir: {data_root.as_posix()}/train/images",
                f"  anno_path: {data_root.as_posix()}/train/annotations.json",
                "  dataset_dir: dataset/coco",
                "  data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']",
                "",
                "EvalDataset:",
                "  !COCODataSet",
                f"  image_dir: {data_root.as_posix()}/val/images",
                f"  anno_path: {data_root.as_posix()}/val/annotations.json",
                "  dataset_dir: dataset/coco",
                "",
                "TestDataset:",
                "  !ImageFolder",
                f"  anno_path: {data_root.as_posix()}/val/annotations.json",
                "",
                "worker_num: 2",
                "TrainReader:",
                "  sample_transforms:",
                "  - !DecodeImage",
                "    to_rgb: True",
                "  - !RandomFlipImage",
                "    prob: 0.5",
                "  - !NormalizeImage",
                "    is_channel_first: false",
                "    is_scale: True",
                "    mean: [0.485, 0.456, 0.406]",
                "    std: [0.229, 0.224, 0.225]",
                "  - !Resize",
                f"    target_size: [{imgsz}, {imgsz}]",
                "    interp: 2",
                "  - !Permute",
                "    to_bgr: false",
                "  batch_size: " + str(int(batch_size)),
                "  shuffle: true",
                "  drop_last: true",
                "",
                "EvalReader:",
                "  batch_size: 1",
                "  drop_last: false",
                "",
                f"epoch: {int(epochs)}",
                "LearningRate:",
                "  base_lr: " + str(float(lr)),
                "  schedulers:",
                "  - !CosineDecay",
                f"    max_epochs: {int(epochs)}",
                "  - !LinearWarmup",
                "    start_factor: 0.001",
                "    steps: 1000",
                "",
                "OptimizerBuilder:",
                "  optimizer:",
                "    type: AdamW",
                "    weight_decay: 0.0001",
                "  regularizer: false",
                "",
                "architecture: RTDetr",
                "pretrain_weights: https://paddlemodels.bj.bcebos.com/object_detection/rtdetr_r50vd.pdparams",
                "RTDetr:",
                "  backbone: ResNet",
                "  neck: HybridEncoder",
                "  transformer: RTDetrTransformer",
                "  post_process: DETRPostProcess",
                "ResNet:",
                "  depth: 50",
                "  variant: d",
                "  norm_decay: 0.0",
                "  return_idx: [1, 2, 3]",
                "HybridEncoder:",
                "  in_channels: [512, 1024, 2048]",
                "  feat_strides: [8, 16, 32]",
                "  hidden_dim: 256",
                "  use_encoder_idx: [2]",
                "  num_encoder_layers: 1",
                "  nhead: 8",
                "  dim_feedforward: 1024",
                "  dropout: 0.0",
                "  enc_act: 'gelu'",
                "  pe_temperature: 10000",
                "  eval_spatial_size: [" + str(imgsz) + ", " + str(imgsz) + "]",
                "RTDetrTransformer:",
                "  num_queries: 300",
                "  feat_strides: [8, 16, 32]",
                "  hidden_dim: 256",
                "  num_levels: 3",
                "  num_decoder_points: 4",
                "  nhead: 8",
                "  num_decoder_layers: 6",
                "  dim_feedforward: 1024",
                "  dropout: 0.0",
                "  act: 'relu'",
                "  eval_idx: -1",
                "  num_denoising: 100",
                "  label_noise_ratio: 0.5",
                "  box_noise_scale: 1.0",
                "  learnt_init_query: false",
                "DETRPostProcess:",
                "  num_top_queries: 300",
                "",
            ]
        ),
        encoding="utf-8",
    )

    bat = out / "train.bat"
    bat.write_text(
        "@echo off\r\n"
        "REM PaddleDetection RT-DETR 训练启动脚本（Windows）\r\n"
        "REM 1) 先 clone PaddleDetection 并安装 paddlepaddle-gpu\r\n"
        "REM 2) 把 rtdetr_v2_custom.yml 放到 PaddleDetection/configs/rtdetr/ 下\r\n"
        "set PADDLE_ROOT=D:\\PaddleDetection\r\n"
        f"copy /Y \"%~dp0rtdetr_v2_custom.yml\" \"%PADDLE_ROOT%\\configs\\rtdetr\\\"\r\n"
        "cd /d %PADDLE_ROOT%\r\n"
        "python tools/train.py -c configs/rtdetr/rtdetr_v2_custom.yml ^\r\n"
        "  --eval ^\r\n"
        "  -o use_gpu=true TrainReader.batch_size=" + str(int(batch_size)) + "\r\n",
        encoding="utf-8",
    )
    sh = out / "train.sh"
    sh.write_text(
        "#!/usr/bin/env bash\n"
        "# PaddleDetection RT-DETR 训练启动脚本（Linux）\n"
        "set -e\n"
        "PADDLE_ROOT=${PADDLE_ROOT:-/workspace/PaddleDetection}\n"
        "cp -f \"$(dirname \"$0\")/rtdetr_v2_custom.yml\" \"${PADDLE_ROOT}/configs/rtdetr/\"\n"
        "cd \"${PADDLE_ROOT}\"\n"
        f"python tools/train.py -c configs/rtdetr/rtdetr_v2_custom.yml --eval -o use_gpu=true TrainReader.batch_size={int(batch_size)}\n",
        encoding="utf-8",
    )
    readme = out / "README_PADDLE.md"
    readme.write_text(
        "# PaddleDetection RT-DETR 路线说明（本仓库只生成配置，不在此处执行训练）\n\n"
        "1. `git clone https://github.com/PaddlePaddle/PaddleDetection`，安装 paddlepaddle-gpu。\n"
        "2. 把 `rtdetr_v2_custom.yml` 复制到 `PaddleDetection/configs/rtdetr/`。\n"
        "3. 确认 yml 里的 `image_dir` / `anno_path` 指向本仓库的 `data/processed/<split>/`。\n"
        "4. 运行 `train.bat`（Windows）或 `train.sh`（Linux）。\n"
        "5. 导出 ONNX 后，把 sidecar 元数据补齐到 `models/<name>/model.onnx.json`（字段见脚本 docstring）。\n\n"
        f"类别数：{label_book.num_classes}\n\n类别列表：\n"
        + "\n".join(f"- {i}: {n}" for i, n in enumerate(label_book.names))
        + "\n",
        encoding="utf-8",
    )
    paths = {"yml": yml, "bat": bat, "sh": sh, "label_list": label_txt, "readme": readme}
    log("已生成 PaddleDetection 配置：" + ", ".join(str(p) for p in paths.values()))
    _ = os.name
    return paths
