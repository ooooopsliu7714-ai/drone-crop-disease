"""病害分类与小样本冷启动训练：DINOv2 / DINOv3 自监督特征 + 线性探测 / LoRA。

这条链路负责回答「**是什么病**」，也是本项目的创新点之一：
    用 2025 年主流的自监督视觉 backbone（DINOv2 / DINOv3）提取通用作物纹理/病斑特征，
    数据少时冻结 backbone 只训线性头（几十张图就能出可用结果），
    数据多时用 LoRA / 全量微调继续提升；训练完自动导出 sidecar 元数据。

怎么用::

    # 1) 查看参数（不需要 transformers，缺依赖只给中文提示）
    python -m scripts.train_cls --help

    # 2) 标准流程：从 data/processed 读取，训练并导出
    python -m scripts.train_cls --config configs/train_cls.yaml \
        --data-root data/processed --backbone facebook/dinov2-base \
        --epochs 10 --batch-size 16 --freeze-backbone \
        --name cls_dinov2 --export-onnx

    # 3) 8 张合成样例上的冒烟测试（CPU 上几十秒跑完，验证全链路）
    python -m scripts.train_cls --data-root data/processed --backbone facebook/dinov2-base \
        --epochs 1 --max-steps-per-epoch 1 --batch-size 4 --name cls_smoke

    # 4) 数据较多时用 LoRA（需要 peft）
    python -m scripts.train_cls --data-root data/processed --peft lora --epochs 20 \
        --lr 1e-4 --backbone-lr 1e-5 --amp --batch-size 8 --name cls_dinov2_lora

    # 5) 只做特征导出/体检，不训练（确认数据与 backbone 可用）
    python -m scripts.train_cls --data-root data/processed --dry-run

依赖：
    必需 —— torch、transformers、Pillow、numpy、PyYAML
    可选 —— peft（--peft lora）、onnx（--export-onnx 生成 ONNX 文件；不装则只写 sidecar json）
    不需要 —— timm / albumentations / torchvision / ultralytics / pycocotools

产物：
    models/<name>/best.pt            state_dict（backbone.* / norm.* / classifier.*）
    models/<name>/checkpoint.json    重建网络所需的结构信息（backbone/num_classes/pooling）
    models/<name>/model.onnx.json    sidecar 元数据（推理侧 src/infer/ 按它加载）
    models/<name>/model.onnx         仅在 --export-onnx 且环境支持时生成
    models/<name>/train_log.jsonl    每 epoch 指标
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import dump_config, load_config
from src.core.labels import load_labels
from src.core.paths import DATA_DIR, MODELS_DIR, PROJECT_ROOT, ensure_dir, model_dir


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    ap = argparse.ArgumentParser(
        prog="python -m scripts.train_cls",
        description="DINOv2/DINOv3 病害分类训练（线性探测 / LoRA / 全量微调）+ ONNX/sidecar 导出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m scripts.train_cls --help\n"
            "  python -m scripts.train_cls --data-root data/processed --epochs 10 --freeze-backbone\n"
            "  python -m scripts.train_cls --data-root data/processed --peft lora --amp --name cls_dinov2_lora\n"
        ),
    )
    ap.add_argument("--config", default=None, help="YAML 配置（默认 configs/train_cls.yaml，存在时自动读取）")
    ap.add_argument("--set", dest="extra_sets", action="append", default=[],
                    help="覆盖配置项，形如 train.epochs=20 model.backbone=facebook/dinov2-base，可重复")
    ap.add_argument("--data-root", default=str(DATA_DIR / "processed"),
                    help="数据根目录（含 train/val/annotations.json），默认 data/processed")
    ap.add_argument("--coco", default=None, help="直接指定单个 COCO json（不划分 train/val 时用）")
    ap.add_argument("--image-root", action="append", default=[], help="图片搜索根目录，可重复")
    ap.add_argument("--backbone", default=None,
                    help="DINOv2/DINOv3 仓库名或本地目录，默认 facebook/dinov2-base")
    ap.add_argument("--backbone-dir", default=None, help="本地 backbone 目录（离线环境用，等价于 --backbone 传本地路径）")
    ap.add_argument("--name", default="cls_dinov2", help="模型名，产物写到 models/<name>/")
    ap.add_argument("--image-size", type=int, default=224, help="分类输入边长（DINOv2 默认 224）")
    ap.add_argument("--crop-margin", type=float, default=0.1, help="从检测框裁 patch 时的外扩比例")
    ap.add_argument("--max-samples-per-class", type=int, default=0, help="每类最多取多少样本（0=不限）")
    ap.add_argument("--exclude-classes", nargs="*", type=int, default=[],
                    help="不参与分类的 class_id（例如把 healthy 排除做纯病害分类）")
    ap.add_argument("--epochs", type=int, default=None, help="训练轮数")
    ap.add_argument("--batch-size", type=int, default=None, help="批大小")
    ap.add_argument("--lr", type=float, default=None, help="分类头学习率")
    ap.add_argument("--backbone-lr", type=float, default=None, help="backbone 学习率（微调时用）")
    ap.add_argument("--weight-decay", type=float, default=None, help="权重衰减")
    ap.add_argument("--grad-accum", type=int, default=None, help="梯度累积步数（等效大 batch）")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None, help="每轮最多多少 step（冒烟测试用）")
    ap.add_argument("--device", default=None, help="auto / cpu / cuda，默认 auto")
    ap.add_argument("--amp", action="store_true", default=None, help="混合精度（仅 CUDA 生效）")
    ap.add_argument("--freeze-backbone", action="store_true", default=None, help="冻结 backbone，只训线性头（默认）")
    ap.add_argument("--unfreeze", action="store_true", default=None, help="解冻 backbone 做全量微调")
    ap.add_argument("--unfreeze-layers", type=int, default=0, help="只解冻 backbone 最后 N 层（0=全部）")
    ap.add_argument("--peft", default=None, choices=["none", "lora"], help="参数高效微调方式，默认 none")
    ap.add_argument("--lora-r", type=int, default=8, help="LoRA rank，默认 8")
    ap.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha，默认 16")
    ap.add_argument("--pooling", default=None, choices=["cls", "mean"], help="特征池化方式，默认 cls")
    ap.add_argument("--dropout", type=float, default=None, help="分类头 dropout")
    ap.add_argument("--class-weight", action="store_true", default=None, help="按类别频率加权交叉熵（缓解长尾）")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader 线程数")
    ap.add_argument("--seed", type=int, default=None, help="随机种子")
    ap.add_argument("--export-onnx", action="store_true", help="训练后导出 ONNX（缺 onnx 包时只写 sidecar）")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset，默认 17")
    ap.add_argument("--dry-run", action="store_true", help="只校验数据与 backbone 配置，不训练")
    return ap


# ---------------------------------------------------------------------------
def _check_deps() -> Optional[str]:
    """检查训练必需依赖，返回错误说明（None 表示 OK）。"""
    missing: List[str] = []
    for pkg, hint in (
        ("torch", "pip install torch --index-url https://download.pytorch.org/whl/cpu"),
        ("transformers", "pip install 'transformers>=4.48'"),
        ("numpy", "pip install numpy"),
        ("PIL", "pip install pillow"),
    ):
        try:
            __import__(pkg)
        except Exception:
            missing.append(f"  - {pkg}: {hint}")
    if missing:
        return "缺少训练分类模型所需的依赖：\n" + "\n".join(missing)
    return None


def resolve_backbone(args: argparse.Namespace, cfg: Dict[str, Any]) -> str:
    """解析 backbone 名称：命令行 > 本地目录 > 配置 > 默认。"""
    from src.train.classifiers import DEFAULT_BACKBONES

    name = args.backbone or args.backbone_dir or cfg.get("model", {}).get("backbone")
    if not name:
        name = DEFAULT_BACKBONES["dinov2-base"]
    if args.backbone_dir:
        name = str(args.backbone_dir)
    return str(name)


def build_datasets(args: argparse.Namespace, cfg: Dict[str, Any], lb) -> tuple:
    """构造 train/val 分类数据集。

    数据来源两种：
        1. ``data/processed/{train,val}/annotations.json``（推荐，prepare_data 的产物）；
        2. ``--coco`` 指定的单个 COCO json，按 8:2 现场切分。
    分类样本 = 每个检测框裁一个 patch（healthy 类用整图）。
    """
    from src.data.torch_datasets import ImageClassifyDataset, discover_coco_json

    image_size = int(args.image_size or cfg.get("data", {}).get("image_size", 224))
    crop_margin = float(args.crop_margin if args.crop_margin is not None else cfg.get("data", {}).get("crop_margin", 0.1))
    max_per_class = int(args.max_samples_per_class or cfg.get("data", {}).get("max_samples_per_class", 0))
    exclude = list(args.exclude_classes or cfg.get("data", {}).get("exclude_classes", []) or [])

    def _one(json_path: Path, train: bool) -> ImageClassifyDataset:
        return ImageClassifyDataset.from_coco(
            json_path,
            image_roots=args.image_root,
            image_size=image_size,
            crop_margin=crop_margin,
            train=train,
            exclude_class_ids=exclude,
            max_samples=max_per_class,
        )

    if args.coco:
        p = Path(args.coco)
        if not p.is_absolute():
            cand = PROJECT_ROOT / p
            p = cand if cand.exists() else p
        if not p.is_file():
            raise FileNotFoundError(f"--coco 指定的标注文件不存在：{p}")
        full = _one(p, train=False)
        n_val = max(1, int(len(full) * 0.2))
        val = ImageClassifyDataset(full.samples[:n_val], image_size=image_size,
                                   crop_margin=crop_margin, train=False)
        train = ImageClassifyDataset(full.samples[n_val:], image_size=image_size,
                                     crop_margin=crop_margin, train=True)
        return train, val, f"{p}（现场 8:2 切分）"

    root = Path(args.data_root)
    if not root.is_absolute():
        cand = PROJECT_ROOT / root
        root = cand if cand.exists() else root
    train_json = discover_coco_json(root / "train")
    val_json = discover_coco_json(root / "val") or discover_coco_json(root / "test") or train_json
    if train_json is None:
        raise FileNotFoundError(
            f"{root} 下找不到 train/annotations.json。\n"
            "  请先运行：python -m src.data.prepare_data --raw-dir data/raw --out-dir data/processed\n"
            "  或生成合成样例：python -m scripts.make_samples"
        )
    train = _one(train_json, train=True)
    val = _one(val_json, train=False) if val_json is not None else None
    src = f"{train_json}（train）+ {val_json}（val）"
    _ = lb
    return train, val, src


def build_sidecar_for_cls(lb, args, cfg, backbone: str, image_size: int, mean, std, num_classes: int) -> Dict[str, Any]:
    """构造分类任务的 sidecar（outputs.kind=logits）。"""
    from src.train.exporter import build_sidecar

    return build_sidecar(
        task="classify",
        label_book=lb,
        source="dinov2-linear" if "dinov2" in backbone.lower() else ("dinov3-linear" if "dinov3" in backbone.lower() else "dino-linear"),
        imgsz=int(image_size),
        input_mean=mean,
        input_std=std,
        output_kind="logits",
        output_names=["logits"],
        note=(
            f"DINOv2/DINOv3 线性探测（backbone={backbone}, num_classes={num_classes}）；"
            "输入需 resize 到 {size}x{size} 并按 mean/std 归一化，输出为类别 logits（softmax 后取最大）。"
        ),
        extra_meta={
            "backbone": backbone,
            "head": "linear",
            "pooling": str(args.pooling or cfg.get("model", {}).get("pooling", "cls")),
            "num_classes": int(num_classes),
            "peft": str(args.peft or cfg.get("model", {}).get("peft", "none")),
            "class_names": lb.names,
        },
    )


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口：返回退出码（0 成功；2 参数/数据问题；3 依赖缺失；4 训练失败）。"""
    args = build_parser().parse_args(argv)

    # 依赖检查放在最前面：缺重依赖时给中文提示并退出非 0，而不是 ImportError 崩溃栈
    dep_err = _check_deps()
    if dep_err:
        print("[错误] " + dep_err, file=sys.stderr)
        print("\n提示：本脚本需要 torch + transformers；CPU 版 torch 也能跑通合成样例的全流程。", file=sys.stderr)
        return 3

    try:
        lb = load_labels()
    except Exception as exc:
        print(f"[错误] 标签体系加载失败：{exc}", file=sys.stderr)
        return 2

    cfg_path = args.config
    if cfg_path is None:
        default_cfg = PROJECT_ROOT / "configs" / "train_cls.yaml"
        cfg_path = str(default_cfg) if default_cfg.exists() else None
    try:
        cfg = load_config(cfg_path, extra_sets=args.extra_sets)
    except Exception as exc:
        print(f"[错误] 配置加载失败：{exc}", file=sys.stderr)
        return 2

    t_cfg = cfg.get("train", {}) or {}
    m_cfg = cfg.get("model", {}) or {}
    d_cfg = cfg.get("data", {}) or {}
    epochs = int(args.epochs if args.epochs is not None else t_cfg.get("epochs", 10))
    batch_size = int(args.batch_size if args.batch_size is not None else t_cfg.get("batch_size", 16))
    lr = float(args.lr if args.lr is not None else t_cfg.get("lr", 1e-3))
    backbone_lr = float(args.backbone_lr if args.backbone_lr is not None else t_cfg.get("backbone_lr", 1e-5))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else t_cfg.get("weight_decay", 0.05))
    grad_accum = int(args.grad_accum if args.grad_accum is not None else t_cfg.get("grad_accum", 1))
    max_steps = int(args.max_steps_per_epoch if args.max_steps_per_epoch is not None
                    else t_cfg.get("max_steps_per_epoch", 0))
    workers = int(args.workers if args.workers is not None else t_cfg.get("workers", 0))
    seed = int(args.seed if args.seed is not None else t_cfg.get("seed", 42))
    amp = bool(args.amp if args.amp is not None else t_cfg.get("amp", False))
    peft = str(args.peft or m_cfg.get("peft", "none") or "none")
    image_size = int(args.image_size or d_cfg.get("image_size", 224))
    pooling = str(args.pooling or m_cfg.get("pooling", "cls"))
    dropout = float(args.dropout if args.dropout is not None else m_cfg.get("dropout", 0.1))
    class_weight_flag = bool(args.class_weight if args.class_weight is not None else t_cfg.get("class_weight", False))
    unfreeze = bool(args.unfreeze if args.unfreeze is not None else t_cfg.get("unfreeze", False))
    freeze_backbone = (not unfreeze)
    if args.freeze_backbone:
        freeze_backbone = True
    device_req = str(args.device or t_cfg.get("device", "auto"))

    backbone = resolve_backbone(args, cfg)
    out_dir = model_dir(args.name)

    print("=" * 78)
    print("病害分类训练（DINOv2/DINOv3 + 线性探测/ LoRA）")
    print("=" * 78)
    print(f"配置来源：{cfg_path or '内置默认'}")
    print(dump_config({k: v for k, v in cfg.items() if k in ("data", "model", "train")}))
    print(f"backbone：{backbone}")
    print(f"模型目录：{out_dir.resolve()}")
    print(f"标签体系：{lb.num_classes} 类 version={lb.version}")
    for line in lb.summary_lines()[:4]:
        print(line)
    print(f"   ...（完整清单见 configs/labels.yaml）")

    try:
        train_ds, val_ds, data_src = build_datasets(args, cfg, lb)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[错误] 数据准备失败：{exc}", file=sys.stderr)
        return 2

    print(f"\n数据来源：{data_src}")
    print(f"训练样本：{len(train_ds)} 个 patch   验证样本：{len(val_ds) if val_ds else 0} 个")
    counts = train_ds.class_counts()
    print("训练集每类样本数：" + ", ".join(f"[{c}]{lb.name_of(c)}={n}" for c, n in sorted(counts.items())))
    absent = [c for c in range(lb.num_classes) if c not in counts]
    if absent:
        print(f"[警告] 以下类别在训练集里没有样本，分类头对应输出将无法学到（建议补数据或排除该类）："
              f"{[lb.name_of(c) for c in absent][:8]}{' ...' if len(absent) > 8 else ''}")

    if args.dry_run:
        print("\n[完成] --dry-run：数据与配置校验通过，未开始训练。")
        print(f"        backbone={backbone}  image_size={image_size}  pooling={pooling}  peft={peft}")
        return 0

    # ---- 建模型 / 训练 --------------------------------------------------
    try:
        from src.data.torch_datasets import IMAGENET_MEAN, IMAGENET_STD
        from src.train.classifiers import DinoClassifier, train_classifier
        from src.train.common import banner, resolve_device

        banner("环境信息", {"模型名": args.name, "backbone": backbone})
        device = resolve_device(device_req)
        model = DinoClassifier(
            backbone,
            num_classes=lb.num_classes,
            pretrained=True,
            pooling=pooling,
            dropout=dropout,
        )
        if peft == "lora":
            desc = model.apply_lora(r=int(args.lora_r), alpha=int(args.lora_alpha))
            print(f"[信息] {desc}")
            freeze_backbone = False  # LoRA 已冻结原参数，只保留 LoRA 与头可训练
        elif args.unfreeze_layers and unfreeze:
            n = model.unfreeze_backbone(unfreeze_last_n_layers=int(args.unfreeze_layers))
            print(f"[信息] 解冻 backbone 最后 {args.unfreeze_layers} 个模块，共 {n / 1e6:.2f} M 参数")
            freeze_backbone = False
        elif unfreeze:
            freeze_backbone = False
            print("[信息] 全量微调模式（--unfreeze），显存占用会明显上升，8GB 建议配合 --amp 和小 batch")

        cw = None
        if class_weight_flag:
            w = train_ds.class_weights()
            cw = (w + [0.0] * max(0, lb.num_classes - len(w)))[: lb.num_classes]
            print(f"[信息] 类别加权交叉熵权重：{[round(v, 3) for v in cw]}")

        result = train_classifier(
            model,
            train_ds,
            val_ds,
            out_dir=out_dir,
            label_book=lb,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            backbone_lr=backbone_lr,
            weight_decay=weight_decay,
            device=device,
            amp=amp,
            grad_accum=grad_accum,
            num_workers=workers,
            class_weights=cw,
            max_steps_per_epoch=max_steps,
            seed=seed,
            freeze_backbone=freeze_backbone,
        )
    except RuntimeError as exc:
        print(f"\n[错误] 训练失败：{exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # pragma: no cover - 兜底，保证不打出裸栈
        print(f"\n[错误] 训练异常：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    print("\n训练完成：")
    print(f"  最好 epoch：{result['best'].get('epoch')}  macro-F1={result['best'].get('macro_f1')}  "
          f"accuracy={result['best'].get('accuracy')}")
    print(f"  权重：{result['weights']}")
    print(f"  结构：{result['checkpoint']}")
    print(f"  日志：{out_dir / 'train_log.jsonl'}")

    # ---- sidecar（先写，保证推理侧一定有元数据）-------------------------
    try:
        with open(out_dir / "checkpoint.json", "r", encoding="utf-8") as f:
            ckpt_info = json.load(f)
        mean = ckpt_info.get("mean") or list(IMAGENET_MEAN)
        std = ckpt_info.get("std") or list(IMAGENET_STD)
    except Exception:
        from src.data.torch_datasets import IMAGENET_MEAN, IMAGENET_STD

        mean, std = list(IMAGENET_MEAN), list(IMAGENET_STD)

    sidecar = build_sidecar_for_cls(lb, args, cfg, backbone, image_size, mean, std, lb.num_classes)
    from src.train.exporter import save_sidecar, validate_sidecar

    problems = validate_sidecar(sidecar)
    if problems:
        print(f"[警告] sidecar 校验问题：{'；'.join(problems)}")
    sc_path = save_sidecar(sidecar, model_dir=out_dir)
    print(f"  sidecar：{sc_path}")

    # ---- ONNX 导出 ------------------------------------------------------
    export_code = 0
    if args.export_onnx:
        try:
            from src.train.exporter import export_onnx, print_export_result

            res = export_onnx(
                model.net,
                out_dir,
                sidecar,
                opset=int(args.opset),
                device="cpu",
                input_name="pixel_values",
                output_names=["logits"],
            )
            export_code = print_export_result(res)
        except Exception as exc:
            print(f"[警告] ONNX 导出失败（权重与 sidecar 仍然有效）：{type(exc).__name__}: {exc}", file=sys.stderr)
            export_code = 1
    else:
        print("[信息] 未指定 --export-onnx，只写了 sidecar；需要 ONNX 请加 --export-onnx 或单独运行：")
        print(f"        python -m scripts.export_onnx --ckpt-dir {out_dir}")

    # ---- 自检：用推理侧接口跑一遍（这是与 src/infer/ 的对接契约）--------
    print("\n推理侧接口自检（build_predictor + predict）：")
    try:
        import numpy as np

        from src.train.predictors import build_predictor

        predictor = build_predictor(str(out_dir), device=device_req)
        sample = train_ds.samples[0]
        img = _read_bgr(sample["image_path"])
        if img is not None:
            out = predictor.predict([img])[0]
            print(f"  输入：{Path(sample['image_path']).name}（{img.shape}）")
            print(f"  输出：{json.dumps(out, ensure_ascii=False)}")
            print("  字段检查：" + ("通过" if set(out.keys()) == {"disease_name", "confidence", "severity", "class_id", "bbox"} else "字段不匹配！"))
            _ = np
        else:
            print("  [警告] 无法读取样例图片，跳过自检")
    except Exception as exc:
        print(f"  [警告] 自检失败（不影响训练产物）：{type(exc).__name__}: {exc}")

    print("\n下一步：")
    print(f"  1) 评估：python -m scripts.eval --gt data/processed/val/annotations.json --ckpt-dir {out_dir}")
    print(f"  2) 完整导出：python -m scripts.export_onnx --ckpt-dir {out_dir} --task classify --imgsz {image_size}")
    _ = MODELS_DIR, ensure_dir
    return export_code


def _read_bgr(path) -> Any:
    """用 cv2 读 BGR 图（cv2 不在时用 PIL 转成 BGR）。"""
    try:
        import cv2

        return cv2.imread(str(path))
    except Exception:
        try:
            import numpy as np
            from PIL import Image

            arr = np.asarray(Image.open(path).convert("RGB"))
            return arr[:, :, ::-1].copy()
        except Exception:
            return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
