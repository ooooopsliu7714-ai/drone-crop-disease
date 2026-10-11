"""病害分类与小样本冷启动（DINOv2 / DINOv3 自监督特征 + 线性探测 / LoRA）。

干什么：
    用 2025 主流的自监督视觉 backbone（DINOv2 / DINOv3）当特征提取器：
        - 数据少（每类 <30 张）→ 冻结 backbone，只训 ``nn.Linear`` 头（线性探测）；
        - 数据多           → ``--peft lora``（需要 peft 包）或全量微调（``--unfreeze``）；
    训练完把 ``{"backbone.*": ..., "classifier.*": ..., "norm.*": ...}`` 存成 ``best.pt``，
    并写 ``models/<name>/checkpoint.json`` 供推理侧重建完全相同的网络结构。

怎么用（完整命令行见 ``scripts/train_cls.py`` 的 docstring）::

    from src.train.classifiers import DinoClassifier, train_classifier
    model = DinoClassifier("facebook/dinov2-base", num_classes=15)
    hist = train_classifier(model, train_ds, val_ds, out_dir="models/cls_dinov2", epochs=10)

依赖：torch（必需）、transformers（必需，惰性 import 并中文提示）；
      peft 仅在 ``--peft lora`` 时需要；不依赖 timm / albumentations / torchvision。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.labels import LabelBook
from src.data.torch_datasets import IMAGENET_MEAN, IMAGENET_STD, ImageClassifyDataset
from src.train.common import JsonlLogger, log, set_seed

__all__ = [
    "DinoClassifier",
    "save_classifier_weights",
    "train_classifier",
    "evaluate_classifier",
    "DEFAULT_BACKBONES",
]

#: 推荐 backbone（transformers 里的官方仓库名；DINOv3 若本地不可用会自动回退 DINOv2）
DEFAULT_BACKBONES: Dict[str, str] = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
    "dinov3-small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}


def _torch():
    """惰性 import torch，缺包时中文提示。"""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 PyTorch，无法训练分类模型。\n"
            "  CPU：pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  RTX 5070（Blackwell sm_120）：pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            f"（原始错误：{exc}）"
        ) from exc
    return torch


def build_backbone(name: str, pretrained: bool = True):
    """加载 DINOv2/DINOv3 backbone（只取 encoder，不要分类头）。"""
    try:
        from transformers import AutoConfig, AutoModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 transformers，无法加载 DINOv2/DINOv3 backbone。\n"
            "  pip install 'transformers>=4.48'\n"
            f"（原始错误：{exc}）"
        ) from exc
    if not pretrained:
        cfg = AutoConfig.from_pretrained(name)
        return AutoModel.from_config(cfg)
    try:
        return AutoModel.from_pretrained(name)
    except Exception as exc:
        # 离线 / 未下载时的友好提示（不静默吞掉）
        raise RuntimeError(
            f"加载 backbone 失败：{name}\n"
            "  1) 首次使用需要联网下载权重（facebook/dinov2-base 约 350MB）；\n"
            "  2) 离线环境请先 `huggingface-cli download <repo>` 或用 --backbone 指定本地目录。\n"
            f"（原始错误：{exc}）"
        ) from exc


class DinoClassifier:
    """DINOv2/DINOv3 + 线性分类头。

    结构（state_dict 的键名固定，推理侧 ``src/train/predictors.py`` 按同名加载）::

        backbone.*        # AutoModel（可能被冻结或加 LoRA）
        norm.*            # LayerNorm(hidden)
        dropout           # 无参数
        classifier.*      # Linear(hidden, num_classes)

    参数:
        backbone_name : 见 ``DEFAULT_BACKBONES``，也可以是本地目录
        num_classes   : 类别数（= labels.yaml 的类别数）
        pretrained    : 是否加载预训练权重
        pooling       : 'cls'（取 CLS token）或 'mean'（patch 特征平均）
        dropout       : 分类头前的 dropout 概率
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        pooling: str = "cls",
        dropout: float = 0.1,
    ) -> None:
        torch = _torch()
        self.backbone_name = backbone_name
        self.num_classes = int(num_classes)
        self.pooling = pooling
        self.backbone = build_backbone(backbone_name, pretrained=pretrained)
        hidden = int(getattr(self.backbone.config, "hidden_size", 0) or 0)
        if hidden <= 0:
            raise RuntimeError(f"无法从 {backbone_name} 的 config 读出 hidden_size，无法建分类头")
        self.hidden_size = hidden

        class _Net(torch.nn.Module):
            """真正的 nn.Module 包装（训练与推理共用同一结构）。"""

            def __init__(self, backbone, hidden_size: int, num_classes: int, pooling: str, dropout: float):
                super().__init__()
                self.backbone = backbone
                self.pooling = pooling
                self.dropout = torch.nn.Dropout(float(dropout))
                self.norm = torch.nn.LayerNorm(hidden_size)
                self.classifier = torch.nn.Linear(hidden_size, int(num_classes))

            def forward(self, pixel_values):
                outs = self.backbone(pixel_values=pixel_values)
                pooled = getattr(outs, "pooler_output", None)
                if pooled is None:
                    last = getattr(outs, "last_hidden_state", None)
                    if last is None:
                        raise RuntimeError("backbone 输出里没有 pooler_output / last_hidden_state")
                    if self.pooling == "mean" and last.ndim == 3:
                        pooled = last.mean(dim=1)
                    else:
                        pooled = last[:, 0, :] if last.ndim == 3 else last
                return self.classifier(self.dropout(self.norm(pooled)))

        self.net = _Net(self.backbone, hidden, self.num_classes, pooling, dropout)

    # ---- 便捷访问 -------------------------------------------------------
    def to(self, device: str):
        """把网络搬到设备上（链式调用）。"""
        self.net.to(device)
        return self

    def parameters(self):
        """可训练参数迭代器（默认只给 requires_grad=True 的）。"""
        return self.net.parameters()

    def state_dict(self) -> Dict[str, Any]:
        """完整 state_dict（``backbone.* / norm.* / classifier.*``）。"""
        return self.net.state_dict()

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False):
        """加载 state_dict（默认宽松匹配，便于只加载线性头）。"""
        return self.net.load_state_dict(state, strict=strict)

    # ---- 冻结 / LoRA ----------------------------------------------------
    def freeze_backbone(self) -> int:
        """冻结 backbone 全部参数，返回被冻结的参数量。"""
        n = 0
        for p in self.net.backbone.parameters():
            p.requires_grad_(False)
            n += p.numel()
        return n

    def unfreeze_backbone(self, unfreeze_last_n_layers: int = 0) -> int:
        """解冻 backbone（``unfreeze_last_n_layers>0`` 时只解冻最后 N 层）。"""
        layers = list(self.net.backbone.children())
        keep = max(0, int(unfreeze_last_n_layers))
        target = layers if keep == 0 else layers[-keep:]
        n = 0
        for mod in target:
            for p in mod.parameters():
                p.requires_grad_(True)
                n += p.numel()
        return n

    def apply_lora(self, r: int = 8, alpha: int = 16, dropout: float = 0.0, target_modules: Optional[Sequence[str]] = None) -> str:
        """给 backbone 注入 LoRA（需要 peft 包）。

        返回: 实际注入的模块名描述字符串。
        缺 peft 时抛 RuntimeError（中文提示），调用方应回退到「冻结 backbone 只训头」。
        """
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as exc:
            raise RuntimeError(
                "缺少 peft，无法使用 LoRA 微调。\n"
                "  pip install peft\n"
                "  或者去掉 --peft lora，走「冻结 backbone + 只训线性头」路径（数据少时反而更稳）。\n"
                f"（原始错误：{exc}）"
            ) from exc
        torch = _torch()
        mods = list(target_modules or ["query", "key", "value", "dense"])
        self.freeze_backbone()
        cfg = LoraConfig(
            r=int(r),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            bias="none",
            target_modules=mods,
        )
        wrapped = get_peft_model(self.net.backbone, cfg)
        self.net.backbone = wrapped
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.net.parameters())
        _ = torch
        return f"LoRA(r={r}, alpha={alpha}, targets={mods}) trainable={trainable}/{total}"


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def _device_tensor(x, device):
    """把 tensor 搬到设备。"""
    return x.to(device) if hasattr(x, "to") else x


def evaluate_classifier(model: DinoClassifier, dataset: ImageClassifyDataset, device: str, max_batches: int = 0) -> Dict[str, Any]:
    """在给定数据集上算 accuracy / macro-F1 / 混淆矩阵（纯 torch + 手写指标）。"""
    torch = _torch()
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    model.net.eval()
    n_cls = model.num_classes
    conf = [[0] * n_cls for _ in range(n_cls)]
    losses, n_seen = 0.0, 0
    crit = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = _device_tensor(batch["pixel_values"].float(), device)
            y = _device_tensor(batch["label"].long(), device)
            logits = model.net(pixel_values=x)
            losses += float(crit(logits, y).item()) * y.numel()
            pred = logits.argmax(dim=-1)
            for t, p in zip(y.cpu().tolist(), pred.cpu().tolist()):
                conf[int(t)][int(p)] += 1
            n_seen += int(y.numel())
            if max_batches and bi + 1 >= max_batches:
                break
    acc = sum(conf[i][i] for i in range(n_cls)) / max(1, n_seen)
    f1s = []
    per_cls = []
    for c in range(n_cls):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(n_cls)) - tp
        fn = sum(conf[c]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = sum(conf[c])
        if support > 0:
            f1s.append(f1)
        per_cls.append({"class_id": c, "support": support, "precision": round(prec, 4),
                        "recall": round(rec, 4), "f1": round(f1, 4)})
    return {
        "loss": round(losses / max(1, n_seen), 6),
        "accuracy": round(acc, 6),
        "macro_f1": round(sum(f1s) / max(1, len(f1s)), 6),
        "num_samples": n_seen,
        "per_class": per_cls,
        "confusion_matrix": conf,
    }


def save_classifier_weights(
    model: DinoClassifier,
    out_dir: str | Path,
    label_book: LabelBook,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """保存 ``best.pt``（state_dict）+ ``checkpoint.json``（推理侧重建结构用）。"""
    torch = _torch()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wpath = out / "best.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, wpath)
    info: Dict[str, Any] = {
        "task": "classify",
        "backbone": model.backbone_name,
        "num_classes": model.num_classes,
        "hidden_size": model.hidden_size,
        "pooling": model.pooling,
        "head": "linear",
        "classes": label_book.sidecar_classes(),
        "image_size": None,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }
    info.update(extra or {})
    cpath = out / "checkpoint.json"
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return {"weights": wpath, "checkpoint": cpath}


def train_classifier(
    model: DinoClassifier,
    train_ds: ImageClassifyDataset,
    val_ds: Optional[ImageClassifyDataset],
    out_dir: str | Path,
    label_book: LabelBook,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    weight_decay: float = 0.05,
    device: str = "cpu",
    amp: bool = False,
    grad_accum: int = 1,
    num_workers: int = 0,
    class_weights: Optional[Sequence[float]] = None,
    max_steps_per_epoch: int = 0,
    seed: int = 42,
    warmup_ratio: float = 0.1,
    freeze_backbone: bool = True,
) -> Dict[str, Any]:
    """线性探测 / LoRA / 全量微调的训练循环。

    返回: ``{"history": [...], "best": {...}, "weights": path, ...}``

    说明:
        - 只有 ``requires_grad=True`` 的参数进优化器，并用两组学习率
          （backbone 用 ``backbone_lr``，分类头用 ``lr``），这是小样本微调的标准做法；
        - ``amp`` 只在 CUDA 上生效（CPU 上自动忽略）；
        - ``--max-steps-per-epoch`` 便于用 8 张样例在 CPU 上快速跑通全流程。
    """
    torch = _torch()
    from torch.utils.data import DataLoader

    set_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = JsonlLogger(out / "train_log.jsonl")

    if freeze_backbone:
        frozen = model.freeze_backbone()
        log(f"已冻结 backbone 参数 {frozen / 1e6:.2f} M（只训分类头，小样本推荐）")
    model.to(device)

    head_params = [p for n, p in model.net.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    bb_params = [p for n, p in model.net.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    groups = []
    if bb_params:
        groups.append({"params": bb_params, "lr": float(backbone_lr)})
    if head_params:
        groups.append({"params": head_params, "lr": float(lr)})
    if not groups:
        raise RuntimeError("没有任何可训练参数：请检查 freeze_backbone / apply_lora 的调用顺序")
    opt = torch.optim.AdamW(groups, weight_decay=float(weight_decay))
    log(f"可训练参数：backbone {sum(p.numel() for p in bb_params)}，头 {sum(p.numel() for p in head_params)}")

    train_loader = DataLoader(
        train_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers), drop_last=False
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

    cw = None
    if class_weights:
        cw = torch.tensor([float(v) for v in class_weights], dtype=torch.float32)
        if float(cw.sum()) <= 0:
            cw = None
        else:
            cw = cw.to(device)
            # 权重为 0 的类别（数据里没有）不参与 loss
            cw = torch.where(cw <= 0, torch.zeros_like(cw), cw)
    crit = torch.nn.CrossEntropyLoss(weight=cw)

    use_amp = bool(amp) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: List[Dict[str, Any]] = []
    best = {"macro_f1": -1.0, "accuracy": -1.0, "epoch": -1}
    global_step = 0
    for epoch in range(1, int(epochs) + 1):
        model.net.train()
        t0 = time.perf_counter()
        run_loss, n_seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            if max_steps_per_epoch and step >= int(max_steps_per_epoch):
                break
            x = _device_tensor(batch["pixel_values"].float(), device)
            y = _device_tensor(batch["label"].long(), device)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model.net(pixel_values=x)
                    loss = crit(logits, y) / max(1, int(grad_accum))
                scaler.scale(loss).backward()
            else:
                logits = model.net(pixel_values=x)
                loss = crit(logits, y) / max(1, int(grad_accum))
                loss.backward()
            run_loss += float(loss.item()) * int(y.numel()) * max(1, int(grad_accum))
            n_seen += int(y.numel())

            if (step + 1) % max(1, int(grad_accum)) == 0:
                if use_amp:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([p for p in model.net.parameters() if p.requires_grad], 5.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_([p for p in model.net.parameters() if p.requires_grad], 5.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

        train_loss = run_loss / max(1, n_seen)
        metrics: Dict[str, Any] = {"train_loss": round(train_loss, 6), "lr": round(sched.get_last_lr()[0], 8)}
        if val_ds is not None and len(val_ds) > 0:
            vm = evaluate_classifier(model, val_ds, device)
            metrics.update({f"val_{k}": v for k, v in vm.items() if k not in ("per_class", "confusion_matrix")})
            metrics["val"] = vm
        metrics.update({"epoch": epoch, "elapsed_sec": round(time.perf_counter() - t0, 2),
                        "global_step": global_step})
        history.append(metrics)
        jsonl.write(metrics)
        log(
            f"epoch {epoch}/{epochs} train_loss={metrics['train_loss']:.4f} "
            + (f"val_acc={metrics.get('val_accuracy', float('nan')):.4f} "
               f"val_macro_f1={metrics.get('val_macro_f1', float('nan')):.4f}" if val_ds is not None and len(val_ds) else "")
        )

        score = float(metrics.get("val_macro_f1", -train_loss))
        if score > best["macro_f1"]:
            best = {
                "macro_f1": score,
                "accuracy": float(metrics.get("val_accuracy", -1.0)),
                "epoch": epoch,
                "train_loss": train_loss,
            }
            save_classifier_weights(
                model, out, label_book,
                extra={"best_epoch": epoch, "best_macro_f1": score,
                       "image_size": getattr(train_ds, "image_size", 224),
                       "mean": list(getattr(train_ds, "mean", IMAGENET_MEAN)),
                       "std": list(getattr(train_ds, "std", IMAGENET_STD))},
            )
            log(f"已保存更优权重 -> {out / 'best.pt'}（macro_f1={score:.4f}）")

    if best["epoch"] < 0:  # 极端情况（0 epoch）兜底保存
        save_classifier_weights(model, out, label_book)
    return {"history": history, "best": best, "weights": str(out / "best.pt"),
            "checkpoint": str(out / "checkpoint.json"), "steps_per_epoch": steps_per_epoch}
