"""主力检测器训练：RT-DETR / RT-DETRv2（WACV2025 分层密集正监督）+ ultralytics / Paddle 后端。

定位：这是替换原 YOLO 的主力链路，负责回答「**病斑在哪里**」。
    默认后端 ``transformers-rtdetrv2`` 用 transformers 的 ``RTDetrV2ForObjectDetection`` 官方实现，
    不依赖 ultralytics；另外提供 ultralytics 版 RT-DETR 与 PaddleDetection 配置生成两条备用路线。

怎么用::

    # 1) 查看参数（缺重依赖时只给中文提示，不抛 ImportError）
    python -m scripts.train_det --help

    # 2) 标准训练（8GB 显存推荐组合：小 batch + 梯度累积 + 冻结 backbone + AMP）
    python -m scripts.train_det --config configs/train_det.yaml \
        --data-root data/processed --backend transformers-rtdetrv2 \
        --name det_rtdetrv2 --epochs 30 --batch-size 2 --grad-accum 8 \
        --freeze-backbone --amp --export-onnx

    # 3) 8 张合成样例冒烟测试（CPU，几十秒验证全链路）
    python -m scripts.train_det --data-root data/processed --epochs 1 \
        --batch-size 1 --max-steps-per-epoch 1 --name det_smoke

    # 4) 用 HuggingFace Trainer 官方范式训练
    python -m scripts.train_det --trainer hf --data-root data/processed --epochs 20 --name det_hf

    # 5) 备用后端：ultralytics RT-DETR（需要 pip install ultralytics）
    python -m scripts.train_det --backend ultralytics-rtdetr --data-root data/processed \
        --epochs 50 --batch-size 4 --name det_yolo

    # 6) 只生成 PaddleDetection 配置与启动脚本（不真跑）
    python -m scripts.train_det --backend paddle --data-root data/processed --name det_paddle

依赖：
    必需 —— torch、transformers、numpy、Pillow、PyYAML
    可选 —— ultralytics（--backend ultralytics-rtdetr）、peft（--peft lora）、
            datasets+pyarrow（--trainer hf）、onnx（--export-onnx 生成 ONNX 文件）
    不需要 —— torchvision / timm / albumentations / pycocotools

产物：
    models/<name>/best.pt            最优权重（state_dict）
    models/<name>/last.pt            最后一轮权重
    models/<name>/checkpoint.json    结构信息（pretrained / num_classes / source）
    models/<name>/model.onnx.json    sidecar 元数据（推理侧按它加载）
    models/<name>/model.onnx         仅 --export-onnx 且环境支持时生成
    models/<name>/train_log.jsonl    每轮指标
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

BACKENDS = ("transformers-rtdetrv2", "ultralytics-rtdetr", "paddle")


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    ap = argparse.ArgumentParser(
        prog="python -m scripts.train_det",
        description="RT-DETR / RT-DETRv2 农作物病害检测训练（transformers 主力 + ultralytics + Paddle）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m scripts.train_det --help\n"
            "  python -m scripts.train_det --data-root data/processed --freeze-backbone --amp --batch-size 2 --grad-accum 8\n"
            "  python -m scripts.train_det --backend ultralytics-rtdetr --data-root data/processed --epochs 50\n"
            "  python -m scripts.train_det --backend paddle --data-root data/processed --name det_paddle\n"
        ),
    )
    ap.add_argument("--config", default=None, help="YAML 配置（默认 configs/train_det.yaml，存在时自动读取）")
    ap.add_argument("--set", dest="extra_sets", action="append", default=[],
                    help="覆盖配置项，形如 train.epochs=30 train.batch_size=2，可重复")
    ap.add_argument("--backend", default=None, choices=list(BACKENDS),
                    help="训练后端，默认 transformers-rtdetrv2（主力）")
    ap.add_argument("--trainer", default=None, choices=["custom", "hf"],
                    help="transformers 后端的训练驱动：custom（自写循环，默认）/ hf（HuggingFace Trainer）")
    ap.add_argument("--data-root", default=str(DATA_DIR / "processed"),
                    help="数据根目录（含 train/val/annotations.json），默认 data/processed")
    ap.add_argument("--train-coco", default=None, help="直接指定训练集 COCO json（覆盖 --data-root）")
    ap.add_argument("--val-coco", default=None, help="直接指定验证集 COCO json")
    ap.add_argument("--image-root", action="append", default=[], help="图片搜索根目录，可重复")
    ap.add_argument("--name", default="det_rtdetrv2", help="模型名，产物写到 models/<name>/")
    ap.add_argument("--variant", default=None, choices=["v2", "v1"], help="RT-DETR 版本：v2（默认）/ v1")
    ap.add_argument("--pretrained", default=None, help="预训练权重仓库名或本地目录（默认 PekingU/rtdetr_v2_r18vd）")
    ap.add_argument("--processor-dir", default=None, help="本地 RTDetrImageProcessor 目录（离线环境）")
    ap.add_argument("--imgsz", type=int, default=None, help="输入边长，默认 640")
    ap.add_argument("--epochs", type=int, default=None, help="训练轮数")
    ap.add_argument("--batch-size", type=int, default=None, help="批大小（8GB 显存建议 2~4）")
    ap.add_argument("--grad-accum", type=int, default=None, help="梯度累积步数（建议 8，等效 batch 16~32）")
    ap.add_argument("--lr", type=float, default=None, help="decoder/head 学习率，默认 1e-4")
    ap.add_argument("--backbone-lr", type=float, default=None, help="backbone 学习率，默认 1e-5")
    ap.add_argument("--weight-decay", type=float, default=None, help="权重衰减，默认 1e-4")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None, help="每轮最多 step（冒烟测试用）")
    ap.add_argument("--device", default=None, help="auto / cpu / cuda，默认 auto")
    ap.add_argument("--amp", action="store_true", default=None, help="混合精度（仅 CUDA 生效，8GB 显存推荐）")
    ap.add_argument("--freeze-backbone", action="store_true", default=None, help="冻结 backbone（省显存，8GB 推荐）")
    # 必须成对提供：`--freeze-backbone` 是 store_true，不传时 default=None 会回退到
    # configs/train_det.yaml 里的 freeze_backbone: true。也就是说**没有"解冻"的开关**
    # ——本机实测踩过：想去掉 --freeze-backbone 来微调，日志却仍显示"已冻结 backbone"，
    # 白跑一轮才发现解冻根本没生效。store_false 且 default=None 才能明确表达"解冻"。
    ap.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                    default=None, help="解冻 backbone 做微调（覆盖配置里的 freeze_backbone: true）")
    ap.add_argument("--peft", default=None, choices=["none", "lora"], help="LoRA 微调（需要 peft），默认 none")
    ap.add_argument("--lora-r", type=int, default=8, help="LoRA rank，默认 8")
    ap.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha，默认 16")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader 线程数")
    ap.add_argument("--seed", type=int, default=None, help="随机种子")
    ap.add_argument("--eval-every", type=int, default=None, help="每多少轮验证一次，默认 1")
    ap.add_argument("--ema", action="store_true", help="预留开关：ultralytics 后端自带 EMA（本参数仅记录到日志）")
    ap.add_argument("--export-onnx", action="store_true", help="训练后导出 ONNX（缺 onnx 包时只写 sidecar）")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset，默认 17")
    # ultralytics 专用
    ap.add_argument("--ultralytics-model", default="rtdetr-l.pt", help="ultralytics 权重，默认 rtdetr-l.pt")
    ap.add_argument("--patience", type=int, default=20, help="ultralytics 早停耐心值，默认 20")
    ap.add_argument("--dry-run", action="store_true", help="只校验数据/配置/模型构建，不训练")
    return ap


def _check_deps(backend: str, trainer: str) -> Optional[str]:
    """按后端检查依赖，返回错误说明（None 表示 OK）。"""
    need = [("torch", "pip install torch --index-url https://download.pytorch.org/whl/cpu"),
            ("numpy", "pip install numpy"),
            ("PIL", "pip install pillow")]
    if backend == "transformers-rtdetrv2":
        need.append(("transformers", "pip install 'transformers>=4.48'"))
    elif backend == "ultralytics-rtdetr":
        need.append(("ultralytics", "pip install ultralytics"))
    missing = []
    for pkg, hint in need:
        try:
            __import__(pkg)
        except Exception:
            missing.append(f"  - {pkg}: {hint}")
    if backend == "paddle":
        # Paddle 路线只生成配置，不要求本机有 paddle
        if not missing:
            return None
    if missing:
        return f"后端 {backend} 缺少依赖：\n" + "\n".join(missing)
    if trainer == "hf":
        try:
            import datasets  # noqa: F401
            import pyarrow  # noqa: F401
        except Exception as exc:
            return (
                "--trainer hf 需要 datasets + pyarrow（transformers.Trainer 的依赖）：\n"
                "  pip install datasets\n"
                "  若 pyarrow 的 DLL 被系统策略拦截，请改用 --trainer custom（本项目默认，功能等价）。\n"
                f"  （探测错误：{type(exc).__name__}: {exc}）"
            )
    return None


def resolve_coco_paths(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Optional[Path]]:
    """定位 train/val 的 COCO json。"""
    from src.data.torch_datasets import discover_coco_json

    def _abs(p) -> Path:
        pp = Path(p)
        if not pp.is_absolute():
            cand = PROJECT_ROOT / pp
            pp = cand if cand.exists() else pp
        return pp

    if args.train_coco:
        return {"train": _abs(args.train_coco), "val": _abs(args.val_coco) if args.val_coco else None}
    root = _abs(args.data_root)
    train = discover_coco_json(root / "train")
    val = discover_coco_json(root / "val") or discover_coco_json(root / "test")
    return {"train": train, "val": val}


def build_sidecar_for_det(lb, args, cfg, source: str, imgsz: int, backend_note: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """构造检测任务的 sidecar（outputs.kind=detect_xyxy_score_cls）。"""
    from src.train.exporter import build_sidecar

    meta = {
        "class_names": lb.names,
        "num_classes": lb.num_classes,
        "backend": str(args.backend or cfg.get("backend", "transformers-rtdetrv2")),
        "trainer": str(args.trainer or cfg.get("trainer", "custom")),
    }
    meta.update(extra or {})
    return build_sidecar(
        task="detect",
        label_book=lb,
        source=source,
        imgsz=int(imgsz),
        input_mean=[0.0, 0.0, 0.0],
        input_std=[255.0, 255.0, 255.0],
        output_kind="detect_xyxy_score_cls",
        output_names=["boxes", "scores", "labels"],
        note=(
            f"{backend_note}；输入为 RGB NCHW，像素缩放到 [0,1]（不做通道标准化，"
            "与 RT-DETR 官方 processor 口径一致）；输出 boxes 为像素 xyxy、scores 置信度、labels 类别 id。"
        ),
        extra_meta=meta,
    )


# ---------------------------------------------------------------------------
def run_transformers_backend(args, cfg, lb, paths: Dict[str, Optional[Path]], imgsz: int,
                             device: str, out_dir: Path) -> Dict[str, Any]:
    """transformers RT-DETR 后端（custom 循环 / HF Trainer）。"""
    from src.data.torch_datasets import RTDetrCocoDataset
    from src.train.detectors import (
        DEFAULT_RTDETR_MODELS,
        HFDetectionDataset,
        apply_lora_detector,
        build_rtdetr_model,
        build_rtdetr_processor,
        freeze_backbone,
        train_rtdetr_hf_trainer,
        train_rtdetr_torch,
    )

    variant = str(args.variant or cfg.get("model", {}).get("variant", "v2"))
    pretrained = str(args.pretrained or cfg.get("model", {}).get("pretrained")
                     or DEFAULT_RTDETR_MODELS[f"{variant}-r18vd" if variant == "v2" else "v1-r50vd"])
    seed = int(args.seed if args.seed is not None else cfg.get("train", {}).get("seed", 42))

    # ---- 数据 ----------------------------------------------------------
    train_ds_raw = RTDetrCocoDataset(paths["train"], image_roots=args.image_root)
    val_ds_raw = RTDetrCocoDataset(paths["val"], image_roots=args.image_root) if paths["val"] else None
    if len(train_ds_raw) == 0:
        raise RuntimeError(
            f"{paths['train']} 中没有可用（有标注且图片存在）的样本。\n"
            "  检查：1) 标注是否已生成（prepare_data）；2) 图片是否能被找到（--image-root）。"
        )
    st = train_ds_raw.label_stats()
    print(f"训练集：{st['num_images']} 张图，{st['num_boxes']} 个框，平均 {st['avg_boxes']} 框/图")
    if val_ds_raw is not None:
        vs = val_ds_raw.label_stats()
        print(f"验证集：{vs['num_images']} 张图，{vs['num_boxes']} 个框")
    train_ds = HFDetectionDataset(train_ds_raw, train=True)
    val_ds = HFDetectionDataset(val_ds_raw, train=False) if val_ds_raw is not None else None

    # ---- 模型 ----------------------------------------------------------
    processor = build_rtdetr_processor(pretrained, imgsz=imgsz, local_dir=args.processor_dir)
    model, used_pretrained = build_rtdetr_model(
        num_classes=lb.num_classes, variant=variant, pretrained=pretrained,
        id2label={c: n for c, n in enumerate(lb.names)},
    )
    peft = str(args.peft or cfg.get("model", {}).get("peft", "none") or "none")
    if peft == "lora":
        print("[信息] " + apply_lora_detector(model, r=int(args.lora_r), alpha=int(args.lora_alpha)))
    elif bool(args.freeze_backbone if args.freeze_backbone is not None else cfg.get("model", {}).get("freeze_backbone", False)):
        frozen, total = freeze_backbone(model)
        print(f"[信息] 已冻结 backbone：{frozen / 1e6:.2f} M / {total / 1e6:.2f} M 参数（省显存）")

    train_cfg = cfg.get("train", {}) or {}
    kwargs = dict(
        out_dir=out_dir,
        label_book=lb,
        epochs=int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 30)),
        batch_size=int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 2)),
        grad_accum=int(args.grad_accum if args.grad_accum is not None else train_cfg.get("grad_accum", 8)),
        lr=float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-4)),
        backbone_lr=float(args.backbone_lr if args.backbone_lr is not None else train_cfg.get("backbone_lr", 1e-5)),
        weight_decay=float(args.weight_decay if args.weight_decay is not None else train_cfg.get("weight_decay", 1e-4)),
        device=device,
        amp=bool(args.amp if args.amp is not None else train_cfg.get("amp", False)),
        num_workers=int(args.workers if args.workers is not None else train_cfg.get("workers", 0)),
        seed=seed,
        pretrained_name=str(used_pretrained),
    )
    trainer = str(args.trainer or cfg.get("trainer", "custom"))
    if trainer == "hf":
        res = train_rtdetr_hf_trainer(
            model, processor, train_ds, val_ds,
            max_steps=int(train_cfg.get("max_steps", -1)),
            **kwargs,
        )
    else:
        res = train_rtdetr_torch(
            model, processor, train_ds, val_ds,
            max_steps_per_epoch=int(args.max_steps_per_epoch if args.max_steps_per_epoch is not None
                                    else train_cfg.get("max_steps_per_epoch", 0)),
            eval_every_epochs=int(args.eval_every if args.eval_every is not None else train_cfg.get("eval_every", 1)),
            save_best_by=str(train_cfg.get("save_best_by", "loss")),
            **kwargs,
        )
    res["model"] = model
    res["processor"] = processor
    res["pretrained"] = used_pretrained
    res["source"] = "rtdetrv2" if variant == "v2" else "rtdetr"
    res["imgsz"] = imgsz
    res["train_ds_raw"] = train_ds_raw
    return res


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口：返回退出码（0 成功；2 参数/数据问题；3 依赖缺失；4 训练失败）。"""
    args = build_parser().parse_args(argv)

    cfg_path = args.config
    if cfg_path is None:
        default_cfg = PROJECT_ROOT / "configs" / "train_det.yaml"
        cfg_path = str(default_cfg) if default_cfg.exists() else None
    try:
        cfg = load_config(cfg_path, extra_sets=args.extra_sets)
    except Exception as exc:
        print(f"[错误] 配置加载失败：{exc}", file=sys.stderr)
        return 2

    backend = str(args.backend or cfg.get("backend", "transformers-rtdetrv2"))
    trainer = str(args.trainer or cfg.get("trainer", "custom"))
    dep_err = _check_deps(backend, trainer)
    if dep_err:
        print("[错误] " + dep_err, file=sys.stderr)
        print("\n提示：transformers 版 RT-DETRv2 是主力路线；ultralytics 只是备用后端。", file=sys.stderr)
        return 3

    try:
        lb = load_labels()
    except Exception as exc:
        print(f"[错误] 标签体系加载失败：{exc}", file=sys.stderr)
        return 2

    imgsz = int(args.imgsz or cfg.get("data", {}).get("imgsz") or 640)
    out_dir = model_dir(args.name)
    print("=" * 78)
    print(f"RT-DETR 病害检测训练（backend={backend}, trainer={trainer}）")
    print("=" * 78)
    print(f"配置来源：{cfg_path or '内置默认'}")
    print(dump_config({k: v for k, v in cfg.items() if k in ("data", "model", "train")}))
    print(f"模型目录：{out_dir.resolve()}")
    print(f"标签体系：{lb.num_classes} 类 version={lb.version}（class_id 与推理侧严格一致）")

    paths = resolve_coco_paths(args, cfg)
    if paths["train"] is None and backend != "paddle":
        print(
            f"[错误] {args.data_root} 下找不到 train/annotations.json。\n"
            "  请先运行：python -m src.data.prepare_data --raw-dir data/raw --out-dir data/processed\n"
            "  或生成合成样例：python -m scripts.make_samples",
            file=sys.stderr,
        )
        return 2
    print(f"训练标注：{paths['train']}")
    print(f"验证标注：{paths['val']}")

    if backend == "paddle":
        from src.train.detectors import write_paddle_configs

        try:
            paths_out = write_paddle_configs(
                out_dir / "paddle", lb, data_root=Path(args.data_root),
                epochs=int(args.epochs or 80),
                batch_size=int(args.batch_size or 4),
                imgsz=imgsz,
                lr=float(args.lr or 1e-4),
            )
        except Exception as exc:
            print(f"[错误] 生成 PaddleDetection 配置失败：{exc}", file=sys.stderr)
            return 4
        sidecar = build_sidecar_for_det(lb, args, cfg, "rtdetrv2",
                                        imgsz, "PaddleDetection RT-DETR（配置已生成，训练需在 PaddleDetection 仓库执行）",
                                        extra={"backend_impl": "paddle"})
        from src.train.exporter import save_sidecar
        print(f"[完成] sidecar 已写出：{save_sidecar(sidecar, model_dir=out_dir)}")
        print("[完成] PaddleDetection 配置：" + ", ".join(str(p) for p in paths_out.values()))
        print("        后续：把 rtdetr_v2_custom.yml 复制到 PaddleDetection/configs/rtdetr/ 后运行 train.bat/train.sh")
        return 0

    from src.train.common import banner, resolve_device

    banner("环境信息", {"模型名": args.name, "backend": backend, "imgsz": imgsz})
    device = resolve_device(str(args.device or cfg.get("train", {}).get("device", "auto")))

    if args.dry_run:
        if backend == "ultralytics-rtdetr":
            from src.train.detectors import prepare_ultralytics_dataset

            yaml_path = prepare_ultralytics_dataset(paths["train"], paths["val"], out_dir / "yolo_dataset", lb)
            print(f"[完成] --dry-run：已导出 ultralytics 数据集 -> {yaml_path}（未开始训练）")
            return 0
        try:
            from src.train.detectors import build_rtdetr_model, build_rtdetr_processor
            from src.data.torch_datasets import RTDetrCocoDataset

            ds = RTDetrCocoDataset(paths["train"], image_roots=args.image_root, with_image=False)
            processor = build_rtdetr_processor(
                str(args.pretrained or cfg.get("model", {}).get("pretrained") or "PekingU/rtdetr_v2_r18vd"),
                imgsz=imgsz, local_dir=args.processor_dir,
            )
            print(f"[完成] --dry-run：数据集 {len(ds)} 张、processor 就绪（size={processor.size}），未训练。")
            return 0
        except Exception as exc:
            print(f"[错误] --dry-run 校验失败：{exc}", file=sys.stderr)
            return 4

    result: Dict[str, Any] = {}
    try:
        if backend == "ultralytics-rtdetr":
            from src.train.detectors import prepare_ultralytics_dataset, train_ultralytics_rtdetr

            data_yaml = prepare_ultralytics_dataset(paths["train"], paths["val"], out_dir / "yolo_dataset", lb)
            res = train_ultralytics_rtdetr(
                data_yaml, out_dir,
                model_name=str(args.ultralytics_model),
                epochs=int(args.epochs or cfg.get("train", {}).get("epochs", 50)),
                imgsz=imgsz,
                batch_size=int(args.batch_size or cfg.get("train", {}).get("batch_size", 4)),
                device=device,
                amp=bool(args.amp),
                workers=int(args.workers or 0),
                seed=int(args.seed or 42),
                patience=int(args.patience),
            )
            result = {"weights": res["weights"], "source": "ultralytics-rtdetr", "imgsz": imgsz,
                      "pretrained": str(args.ultralytics_model), "model": None}
        else:
            result = run_transformers_backend(args, cfg, lb, paths, imgsz, device, out_dir)
    except RuntimeError as exc:
        print(f"\n[错误] 训练失败：{exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # pragma: no cover
        print(f"\n[错误] 训练异常：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    print("\n训练完成：")
    if "best" in result:
        print(f"  最好轮次：{result['best'].get('epoch')}  指标={result['best'].get('metrics', {})}")
    print(f"  权重：{result.get('weights')}")
    print(f"  日志：{out_dir / 'train_log.jsonl'}")

    sidecar = build_sidecar_for_det(
        lb, args, cfg, str(result.get("source", "rtdetrv2")), imgsz,
        f"{backend} 训练的 RT-DETR 检测器（pretrained={result.get('pretrained')}）",
        extra={"pretrained": str(result.get("pretrained") or ""), "backend_impl": backend},
    )
    from src.train.exporter import print_export_result, save_sidecar, validate_sidecar

    problems = validate_sidecar(sidecar)
    if problems:
        print(f"[警告] sidecar 校验问题：{'；'.join(problems)}")
    print(f"  sidecar：{save_sidecar(sidecar, model_dir=out_dir)}")

    export_code = 0
    model = result.get("model")
    if args.export_onnx:
        from src.train.exporter import export_onnx

        if model is None:
            wpath = Path(str(result.get("weights") or ""))
            if wpath.is_file() and "ultralytics" in backend:
                import shutil

                shutil.copy2(wpath, out_dir / "best.pt")
                print(f"[信息] 已把 ultralytics 权重复制到 {out_dir / 'best.pt'}"
                      "（导出 ONNX 请在 ultralytics 环境执行 model.export(format='onnx')）")
            res = {"ok": True, "exported": False, "kind": "skipped",
                   "reason": "ultralytics 后端请用 ultralytics 自带导出：from ultralytics import RTDETR; "
                             "RTDETR('best.pt').export(format='onnx')，导出后按 sidecar 字段补齐 meta。"}
            export_code = print_export_result(res)
        else:
            res = export_onnx(
                model, out_dir, sidecar, opset=int(args.opset), device="cpu",
                input_name="images", output_names=["boxes", "scores", "labels"],
            )
            export_code = print_export_result(res)
    else:
        print("[信息] 未指定 --export-onnx，只写了 sidecar；需要 ONNX 请加 --export-onnx 或单独运行：")
        print(f"        python -m scripts.export_onnx --ckpt-dir {out_dir} --task detect --imgsz {imgsz}")

    # ---- 推理侧契约自检 -------------------------------------------------
    if (out_dir / "best.pt").is_file():
        print("\n推理侧接口自检（build_predictor + predict）：")
        try:
            import numpy as np

            from src.train.predictors import build_predictor

            predictor = build_predictor(str(out_dir), device=str(args.device or "auto"))
            ds_raw = result.get("train_ds_raw")
            if ds_raw is not None and len(ds_raw) > 0:
                item = ds_raw[0]
                img = np.asarray(item["image"])[:, :, ::-1].copy()  # RGB -> BGR，模拟 OpenCV 读入
                out = predictor.predict([img])[0]
                print(f"  输入：{item['file_name']}（{img.shape}）")
                print(f"  输出：{json.dumps(out, ensure_ascii=False)}")
                ok = set(out.keys()) == {"disease_name", "confidence", "severity", "class_id", "bbox"}
                print("  字段检查：" + ("通过" if ok else "字段不匹配！"))
                dbg = predictor.predict_debug([img])[0]
                print(f"  调试信息：检出 {dbg['num_boxes']} 个框，病斑面积占比 {dbg['disease_area_ratio']}")
        except Exception as exc:
            print(f"  [警告] 自检失败（不影响训练产物）：{type(exc).__name__}: {exc}")

    print("\n下一步：")
    print(f"  1) 评估：python -m scripts.eval --gt {paths['val'] or paths['train']} --ckpt-dir {out_dir}")
    print(f"  2) 完整导出：python -m scripts.export_onnx --ckpt-dir {out_dir} --task detect --imgsz {imgsz}")
    _ = MODELS_DIR, ensure_dir
    return export_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
