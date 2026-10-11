"""开放词汇 / 零样本检测训练：YOLO-World + Grounding DINO。

干什么（对应大赛强调的「创新性」与「类别可扩展」）：
    - **Grounding DINO**（transformers）：直接用中文/英文病名作文本提示，零样本出框，
      新增病害类别**不需要重新标注**，还能反过来给无标注航拍图自动预标注（见 scripts/autolabel.py）；
    - **YOLO-World**（ultralytics）：把病害名设成自定义词表后微调，推理时改词表即可扩展类别。

怎么用::

    # 1) 查看参数
    python -m scripts.train_openvocab --help

    # 2) YOLO-World：导出数据集 + 生成训练命令（缺 ultralytics 时只生成配置与命令）
    python -m scripts.train_openvocab --method yoloworld --data-root data/processed \
        --epochs 50 --batch-size 4 --name ov_yoloworld

    # 3) Grounding DINO 零样本演示（不需要训练，直接给病名提示出框；需联网首次下载权重）
    python -m scripts.train_openvocab --method grounding-dino \
        --images data/samples/images --prompt-lang en --name ov_grounding

    # 4) 只生成/校验配置，不做任何推理或训练
    python -m scripts.train_openvocab --method yoloworld --dry-run --name ov_cfg

依赖：
    transformers 路线 —— torch + transformers + numpy + Pillow
    YOLO-World 路线 —— ultralytics（惰性 import；缺包时只输出配置与命令，退出码非 0 并给中文提示）
    不需要 —— timm / albumentations / pycocotools

产物：
    models/<name>/model.onnx.json        sidecar 元数据（推理侧按它加载）
    models/<name>/best.pt                YOLO-World 微调权重（ultralytics 路线）
    models/<name>/yoloworld/data.yaml    ultralytics 数据集配置
    models/<name>/train_cmd.txt          可直接复制执行的训练命令（便于在别的机器上跑）
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

METHODS = ("yoloworld", "grounding-dino")


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    ap = argparse.ArgumentParser(
        prog="python -m scripts.train_openvocab",
        description="开放词汇/零样本病害检测：YOLO-World 微调 与 Grounding DINO 文本提示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m scripts.train_openvocab --method yoloworld --data-root data/processed --epochs 50\n"
            "  python -m scripts.train_openvocab --method grounding-dino --images data/samples/images\n"
        ),
    )
    ap.add_argument("--config", default=None, help="YAML 配置（默认 configs/train_openvocab.yaml）")
    ap.add_argument("--set", dest="extra_sets", action="append", default=[], help="覆盖配置项 key=value，可重复")
    ap.add_argument("--method", default=None, choices=list(METHODS),
                    help="方法：yoloworld（词表微调）/ grounding-dino（零样本演示）")
    ap.add_argument("--data-root", default=str(DATA_DIR / "processed"), help="数据根目录，默认 data/processed")
    ap.add_argument("--images", default=str(DATA_DIR / "samples" / "images"),
                    help="Grounding DINO 演示/预标注用的图片目录")
    ap.add_argument("--name", default=None, help="模型名，产物写到 models/<name>/（默认按方法命名）")
    ap.add_argument("--model", default=None, help="预训练权重：YOLO-World 默认 yolov8s-worldv2.pt")
    ap.add_argument("--local-dir", default=None, help="本地权重目录（离线环境）")
    ap.add_argument("--prompt-lang", default=None, choices=["en", "cn"], help="文本提示语言，默认 en")
    ap.add_argument("--prompts", nargs="*", default=None, help="自定义文本提示（覆盖 labels.yaml 生成的提示）")
    ap.add_argument("--box-threshold", type=float, default=None, help="Grounding DINO 框阈值，默认 0.3")
    ap.add_argument("--text-threshold", type=float, default=None, help="Grounding DINO 文本阈值，默认 0.25")
    ap.add_argument("--max-images", type=int, default=None, help="零样本演示最多处理多少张图，默认 8")
    ap.add_argument("--epochs", type=int, default=None, help="YOLO-World 训练轮数")
    ap.add_argument("--batch-size", type=int, default=None, help="YOLO-World 批大小")
    ap.add_argument("--imgsz", type=int, default=None, help="输入边长，默认 640")
    ap.add_argument("--device", default=None, help="auto / cpu / cuda")
    ap.add_argument("--freeze", type=int, default=None, help="YOLO-World 冻结前 N 层（省显存）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子")
    ap.add_argument("--dry-run", action="store_true", help="只生成配置与命令，不训练/不推理")
    return ap


def build_sidecar_for_openvocab(lb, method: str, imgsz: int, model_name: str,
                                prompt_lang: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """构造开放词汇方法的 sidecar。"""
    from src.train.exporter import build_sidecar

    source = "yoloworld" if method == "yoloworld" else "grounding-dino"
    meta = {
        "class_names": lb.names,
        "num_classes": lb.num_classes,
        "prompt_lang": prompt_lang,
        "prompts": lb.openvocab_prompts("cn" if prompt_lang == "cn" else "en"),
        "source_model": model_name,
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
            f"开放词汇检测（{method}）：用文本提示识别病害类别，新增类别无需重新标注。"
            "boxes 为像素 xyxy，labels 已映射到 labels.yaml 的 class_id。"
        ),
        extra_meta=meta,
    )


def run_yoloworld(args, cfg, lb, out_dir: Path, imgsz: int) -> int:
    """YOLO-World 路线：导出数据集 + 设置词表 + 训练（缺 ultralytics 时给出命令）。"""
    from src.data.torch_datasets import discover_coco_json

    root = Path(args.data_root)
    if not root.is_absolute():
        cand = PROJECT_ROOT / root
        root = cand if cand.exists() else root
    train_json = discover_coco_json(root / "train")
    val_json = discover_coco_json(root / "val") or discover_coco_json(root / "test")
    if train_json is None:
        print(f"[错误] {root} 下找不到 train/annotations.json，请先跑 src.data.prepare_data", file=sys.stderr)
        return 2

    model_name = str(args.model or cfg.get("model", {}).get("yoloworld", "yolov8s-worldv2.pt"))
    epochs = int(args.epochs or cfg.get("train", {}).get("epochs", 50))
    batch = int(args.batch_size or cfg.get("train", {}).get("batch_size", 4))
    freeze = int(args.freeze if args.freeze is not None else cfg.get("train", {}).get("freeze", 0))
    device = str(args.device or cfg.get("train", {}).get("device", "auto"))
    seed = int(args.seed or cfg.get("train", {}).get("seed", 42))
    lang = str(args.prompt_lang or cfg.get("prompts", {}).get("lang", "en"))

    from src.train.openvocab import prepare_yoloworld_dataset

    ds_dir = out_dir / "yoloworld"
    data_yaml = prepare_yoloworld_dataset(train_json, val_json or train_json, ds_dir, lb)
    print(f"数据集已导出：{data_yaml}")

    vocab = args.prompts or lb.openvocab_prompts("cn" if lang == "cn" else "en")
    print(f"YOLO-World 词表（{lang}，{len(vocab)} 项）：{vocab}")

    cmd = (
        "yolo train model=" + model_name + " data=" + str(data_yaml)
        + f" epochs={epochs} imgsz={imgsz} batch={batch} device={'0' if device.startswith('cuda') else 'cpu'}"
        + (f" freeze={freeze}" if freeze else "")
        + f" project={out_dir} name=yoloworld exist_ok=True seed={seed}"
    )
    cmd_file = out_dir / "train_cmd.txt"
    cmd_file.write_text(
        "# YOLO-World 训练命令（在装有 ultralytics 的环境执行）\n"
        "# 也可以用 Python API：\n"
        "#   from ultralytics import YOLOWorld\n"
        f"#   m = YOLOWorld('{model_name}'); m.set_classes({json.dumps(vocab, ensure_ascii=False)})\n"
        f"#   m.train(data='{data_yaml}', epochs={epochs}, imgsz={imgsz}, batch={batch})\n"
        + cmd + "\n",
        encoding="utf-8",
    )
    print(f"训练命令已写入：{cmd_file}")

    if args.dry_run:
        print("[完成] --dry-run：仅生成数据集与训练命令，未训练。")
        return 0

    from src.train.openvocab import train_yolo_world

    try:
        res = train_yolo_world(
            data_yaml, out_dir, lb, model_name=model_name, epochs=epochs,
            imgsz=imgsz, batch_size=batch, device=device, lang=lang,
            freeze=freeze, seed=seed,
        )
    except RuntimeError as exc:
        print(f"\n[错误] YOLO-World 训练失败：{exc}", file=sys.stderr)
        print(f"        可手动执行：{cmd}", file=sys.stderr)
        return 3
    print(f"[完成] YOLO-World 权重：{res['weights']}")
    return 0


def run_grounding(args, cfg, lb, out_dir: Path, imgsz: int) -> int:
    """Grounding DINO 路线：零样本演示（可选：对图片目录批量出框并落盘伪标签）。"""
    from src.data.torch_datasets import SimpleImageFolder
    from src.train.common import resolve_device
    from src.train.openvocab import DEFAULT_GROUNDING_DINO, build_grounding_dino, detect_with_grounding_dino

    model_name = str(args.model or cfg.get("model", {}).get("grounding_dino", DEFAULT_GROUNDING_DINO))
    lang = str(args.prompt_lang or cfg.get("prompts", {}).get("lang", "en"))
    prompts = args.prompts or lb.openvocab_prompts("cn" if lang == "cn" else "en")
    box_thr = float(args.box_threshold if args.box_threshold is not None else cfg.get("infer", {}).get("box_threshold", 0.3))
    text_thr = float(args.text_threshold if args.text_threshold is not None else cfg.get("infer", {}).get("text_threshold", 0.25))
    max_images = int(args.max_images if args.max_images is not None else cfg.get("infer", {}).get("max_images", 8))

    print(f"文本提示（{lang}，{len(prompts)} 项）：{prompts}")
    print(f"阈值：box={box_thr}  text={text_thr}  最多处理 {max_images} 张图")
    if args.dry_run:
        print("[完成] --dry-run：仅打印提示词与阈值，未加载模型。")
        return 0

    device = resolve_device(str(args.device or cfg.get("infer", {}).get("device", "auto")))
    try:
        model, processor, used = build_grounding_dino(model_name, device=device, local_dir=args.local_dir)
    except RuntimeError as exc:
        print(f"\n[错误] {exc}", file=sys.stderr)
        return 3

    try:
        folder = SimpleImageFolder(args.images)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    paths = list(folder.paths)[:max_images] if max_images > 0 else list(folder.paths)
    if not paths:
        print(f"[错误] {args.images} 下没有图片", file=sys.stderr)
        return 2

    try:
        from PIL import Image
    except Exception as exc:
        print(f"[错误] 需要 Pillow 读取图片：{exc}", file=sys.stderr)
        return 3

    print(f"\n开始零样本检测（{len(paths)} 张）：")
    total = 0
    summary: List[Dict[str, Any]] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        dets = detect_with_grounding_dino(
            model, processor, img, prompts, device=device,
            box_threshold=box_thr, text_threshold=text_thr,
        )
        total += len(dets)
        labels = ", ".join(f"{d['label']}({d['score']:.2f})" for d in dets[:4])
        print(f"  {Path(p).name}：{len(dets)} 个框  {labels}{' ...' if len(dets) > 4 else ''}")
        summary.append({"image": Path(p).name, "num_boxes": len(dets),
                        "boxes": [{"label": d["label"], "score": d["score"], "bbox": d["bbox"]} for d in dets]})

    report = out_dir / "zeroshot_preview.json"
    with open(report, "w", encoding="utf-8") as f:
        json.dump({"model": used, "prompts": prompts, "device": device,
                   "box_threshold": box_thr, "text_threshold": text_thr,
                   "total_boxes": total, "images": summary}, f, ensure_ascii=False, indent=2)
    print(f"\n零样本演示完成：共 {total} 个框，明细写入 {report}")
    print("提示：要把这些零样本结果变成训练标签，请运行：")
    print(f"      python -m scripts.autolabel --backend grounding-dino --images {args.images} "
          f"--out-dir data/pseudo --prompt-lang {lang}")
    _ = imgsz
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口：返回退出码（0 成功；2 参数/数据问题；3 依赖缺失；4 运行失败）。"""
    args = build_parser().parse_args(argv)

    cfg_path = args.config
    if cfg_path is None:
        default_cfg = PROJECT_ROOT / "configs" / "train_openvocab.yaml"
        cfg_path = str(default_cfg) if default_cfg.exists() else None
    try:
        cfg = load_config(cfg_path, extra_sets=args.extra_sets)
    except Exception as exc:
        print(f"[错误] 配置加载失败：{exc}", file=sys.stderr)
        return 2

    method = str(args.method or cfg.get("method", "yoloworld"))
    name = args.name or str(cfg.get("name") or ("ov_yoloworld" if method == "yoloworld" else "ov_grounding"))
    imgsz = int(args.imgsz or cfg.get("data", {}).get("imgsz") or 640)
    out_dir = model_dir(name)

    try:
        lb = load_labels()
    except Exception as exc:
        print(f"[错误] 标签体系加载失败：{exc}", file=sys.stderr)
        return 2

    print("=" * 78)
    print(f"开放词汇检测（method={method}）")
    print("=" * 78)
    print(f"配置来源：{cfg_path or '内置默认'}")
    print(dump_config({k: v for k, v in cfg.items() if k in ("data", "model", "train", "prompts", "infer")}))
    print(f"模型目录：{out_dir.resolve()}")
    print(f"标签体系：{lb.num_classes} 类（新增类别只需改 configs/labels.yaml，无需重新标注即可用零样本兜底）")

    # sidecar 先写好：即使后续训练/推理失败，推理侧也有可读的元数据
    lang = str(args.prompt_lang or cfg.get("prompts", {}).get("lang", "en"))
    model_name = str(args.model or (cfg.get("model", {}) or {}).get(
        "yoloworld" if method == "yoloworld" else "grounding_dino", ""))
    sidecar = build_sidecar_for_openvocab(
        lb, method, imgsz, model_name or "unknown", lang,
        extra={"box_threshold": float(args.box_threshold if args.box_threshold is not None
                                      else cfg.get("infer", {}).get("box_threshold", 0.3))},
    )
    from src.train.exporter import save_sidecar, validate_sidecar

    problems = validate_sidecar(sidecar)
    if problems:
        print(f"[警告] sidecar 校验问题：{'；'.join(problems)}")
    print(f"sidecar：{save_sidecar(sidecar, model_dir=out_dir)}")

    try:
        if method == "yoloworld":
            return run_yoloworld(args, cfg, lb, out_dir, imgsz)
        return run_grounding(args, cfg, lb, out_dir, imgsz)
    except Exception as exc:  # pragma: no cover - 兜底，避免裸栈
        print(f"\n[错误] 运行异常：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
