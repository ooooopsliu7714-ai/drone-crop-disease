"""自动预标注：给无标注航拍图生成伪标签，供农技端人工复核（AI 初筛 + 人工复核闭环）。

干什么：
    1. 用 **Grounding DINO** 文本提示零样本出框（新病害类别不用先标注），或
       用**已训练的检测模型**（``models/<name>``，走 ``src.train.predictors``）推理；
    2. 把结果写成 YOLO 伪标签 ``data/pseudo/<图片名>.txt``：``cls cx cy w h score label_name``
       （第 6 列置信度、第 7 列原始类别名给人工复核排序用；训练前可 ``--strip-score`` 去掉两列）；
    3. 同时输出 COCO 格式 ``data/pseudo/annotations.json`` 与中文复核报告
       ``data/pseudo/autolabel_report.json``（每图框数、类别分布、平均置信度、低置信优先复核清单）。

怎么用::

    # 1) 用 Grounding DINO 零样本预标注（推荐，冷启动无需任何标注）
    python -m scripts.autolabel --backend grounding-dino --images data/samples/images \
        --out-dir data/pseudo --prompt-lang en --box-threshold 0.3

    # 2) 用已训练模型预标注（RT-DETR / DINO 分类产物均可）
    python -m scripts.autolabel --backend ckpt --ckpt-dir models/det_rtdetrv2 \
        --images data/raw/images --out-dir data/pseudo --conf 0.35

    # 3) 生成训练可直接用的纯 YOLO 标签（去掉 score 与类别名两列）
    python -m scripts.autolabel --backend ckpt --ckpt-dir models/det_rtdetrv2 \
        --images data/raw/images --out-dir data/pseudo --strip-score

依赖：
    grounding-dino 后端 —— torch + transformers + numpy + Pillow
    ckpt 后端          —— torch + transformers（或 ultralytics，取决于产物的 meta.source）
    都不需要 pycocotools / onnxruntime。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.labels import load_labels
from src.core.paths import DATA_DIR, MODELS_DIR, PROJECT_ROOT, ensure_dir

BACKENDS = ("grounding-dino", "ckpt")


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    ap = argparse.ArgumentParser(
        prog="python -m scripts.autolabel",
        description="自动预标注：Grounding DINO 零样本 或 已训模型推理 -> data/pseudo/*.txt（供人工复核）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m scripts.autolabel --backend grounding-dino --images data/samples/images\n"
            "  python -m scripts.autolabel --backend ckpt --ckpt-dir models/det_rtdetrv2 --images data/raw/images\n"
        ),
    )
    ap.add_argument("--backend", default="grounding-dino", choices=list(BACKENDS),
                    help="伪标签来源：grounding-dino（零样本）/ ckpt（已训模型）")
    ap.add_argument("--images", required=True, help="待标注图片目录（递归遍历）")
    ap.add_argument("--out-dir", default=str(DATA_DIR / "pseudo"), help="输出目录，默认 data/pseudo")
    ap.add_argument("--ckpt-dir", default=None, help="--backend ckpt 时的模型目录 models/<name>")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda，默认 auto")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    ap.add_argument("--box-threshold", type=float, default=None, help="Grounding DINO 框阈值（默认取 --conf）")
    ap.add_argument("--text-threshold", type=float, default=0.25, help="Grounding DINO 文本阈值，默认 0.25")
    ap.add_argument("--prompt-lang", default="en", choices=["en", "cn"], help="文本提示语言，默认 en")
    ap.add_argument("--prompts", nargs="*", default=None, help="自定义文本提示（覆盖 labels.yaml 生成的提示）")
    ap.add_argument("--local-dir", default=None, help="Grounding DINO 本地权重目录（离线）")
    ap.add_argument("--model", default=None, help="Grounding DINO 权重名，默认 IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--max-images", type=int, default=0, help="最多处理多少张（0=全部）")
    ap.add_argument("--max-boxes-per-image", type=int, default=100, help="每图最多保留多少框，默认 100")
    ap.add_argument("--strip-score", action="store_true", help="输出纯 YOLO 标签（只留 cls cx cy w h）")
    ap.add_argument("--no-coco", action="store_true", help="不生成 data/pseudo/annotations.json")
    ap.add_argument("--dry-run", action="store_true", help="只列出将处理的图片，不加载模型")
    return ap


def _read_rgb(path: Path):
    """读成 RGB numpy 数组（cv2 优先，其次 PIL）。"""
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is not None:
            return img[:, :, ::-1].copy()  # BGR -> RGB
    except Exception:
        pass
    try:
        import numpy as np
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _read_bgr(path: Path):
    """读成 BGR uint8（推理侧 predict 的约定格式）。"""
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is not None:
            return img
    except Exception:
        pass
    try:
        import numpy as np
        from PIL import Image

        arr = np.asarray(Image.open(path).convert("RGB"))
        return arr[:, :, ::-1].copy()
    except Exception:
        return None


def _dets_from_ckpt(predictor, image_bgr, lb) -> List[Dict[str, Any]]:
    """用已训模型的 predict_debug 拿到框，转成 {'label','score','bbox'} 结构。"""
    res = predictor.predict_debug([image_bgr])[0]
    out: List[Dict[str, Any]] = []
    for b in res.get("boxes", []):
        cid = int(b["class_id"])
        out.append({"label": lb.name_of(cid), "score": float(b["confidence"]), "bbox": list(b["bbox"])})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口：返回退出码（0 成功；2 参数/数据问题；3 依赖缺失；4 运行失败）。"""
    args = build_parser().parse_args(argv)
    try:
        lb = load_labels()
    except Exception as exc:
        print(f"[错误] 标签体系加载失败：{exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 收集图片
    from src.data.torch_datasets import SimpleImageFolder

    try:
        folder = SimpleImageFolder(args.images)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    paths = list(folder.paths)
    if args.max_images and args.max_images > 0:
        paths = paths[: int(args.max_images)]
    if not paths:
        print(f"[错误] {args.images} 下没有图片", file=sys.stderr)
        return 2

    print("=" * 78)
    print("自动预标注（AI 初筛 + 人工复核闭环）")
    print("=" * 78)
    print(f"后端：{args.backend}   图片：{len(paths)} 张（{args.images}）")
    print(f"输出：{out_dir.resolve()}")
    print(f"标签体系：{lb.num_classes} 类（class_id 严格按 configs/labels.yaml）")
    if args.dry_run:
        for p in paths[:20]:
            print(f"  - {p}")
        if len(paths) > 20:
            print(f"  ...（其余 {len(paths) - 20} 张）")
        print("[完成] --dry-run：未加载模型。")
        return 0

    # ---- 准备后端 -------------------------------------------------------
    predictor = None
    gd_model = gd_proc = None
    prompts: List[str] = []
    model_desc = ""
    resolved_device = "cpu"
    if args.backend == "grounding-dino":
        from src.train.common import resolve_device
        from src.train.openvocab import DEFAULT_GROUNDING_DINO, build_grounding_dino

        prompts = args.prompts or lb.openvocab_prompts(args.prompt_lang)
        model_name = str(args.model or DEFAULT_GROUNDING_DINO)
        resolved_device = resolve_device(args.device)
        try:
            gd_model, gd_proc, model_desc = build_grounding_dino(model_name, device=resolved_device, local_dir=args.local_dir)
        except RuntimeError as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            return 3
        print(f"文本提示（{args.prompt_lang}，{len(prompts)} 项）：{prompts}")
    else:
        if not args.ckpt_dir:
            print("[错误] --backend ckpt 需要 --ckpt-dir 指定模型目录", file=sys.stderr)
            return 2
        try:
            from src.train.common import resolve_device
            from src.train.predictors import build_predictor

            resolved_device = resolve_device(args.device)
            predictor = build_predictor(args.ckpt_dir, device=resolved_device)
            model_desc = str(Path(args.ckpt_dir).resolve())
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[错误] 加载模型失败：{exc}", file=sys.stderr)
            return 3
        print(f"模型：{model_desc}")

    # ---- 逐图推理 -------------------------------------------------------
    from src.train.openvocab import boxes_to_coco, detect_with_grounding_dino, save_pseudo_labels

    coco: Dict[str, Any] = {
        "info": {"description": "autolabel 伪标签（待人工复核）", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": c["class_id"], "name": c["disease_name"], "name_cn": c["disease_name_cn"]}
            for c in lb.sidecar_classes()
        ],
    }
    entries: List[Dict[str, Any]] = []
    ann_id = 1
    n_failed = 0
    box_thr = float(args.box_threshold if args.box_threshold is not None else args.conf)

    print(f"\n开始推理（box_threshold={box_thr}）：")
    for i, p in enumerate(paths, start=1):
        try:
            if gd_model is not None:
                rgb = _read_rgb(p)
                if rgb is None:
                    print(f"  [警告] 读不到图片：{p.name}")
                    n_failed += 1
                    continue
                dets = detect_with_grounding_dino(
                    gd_model, gd_proc, rgb, prompts,
                    device=resolved_device,
                    box_threshold=box_thr, text_threshold=float(args.text_threshold),
                    max_boxes=int(args.max_boxes_per_image),
                )
            else:
                bgr = _read_bgr(p)
                if bgr is None:
                    print(f"  [警告] 读不到图片：{p.name}")
                    n_failed += 1
                    continue
                dets = _dets_from_ckpt(predictor, bgr, lb)
                dets = [d for d in dets if d["score"] >= float(args.conf)][: int(args.max_boxes_per_image)]
        except Exception as exc:
            print(f"  [警告] {p.name} 推理失败：{type(exc).__name__}: {str(exc)[:120]}")
            n_failed += 1
            continue

        h, w = _image_size(p)
        rec = {"id": i, "file_name": p.name, "width": w, "height": h, "_source_path": str(p)}
        coco["images"].append(rec)

        # YOLO 伪标签
        if args.strip_score:
            lines = []
            name2id = lb.name_to_id_map()
            for d in dets:
                key = str(d["label"]).strip().lower().replace(" ", "_")
                cid = name2id.get(key, lb.healthy_id())
                x1, y1, x2, y2 = (float(v) for v in d["bbox"][:4])
                x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
                y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
                bw, bh = x2 - x1, y2 - y1
                if bw <= 1 or bh <= 1:
                    continue
                lines.append(f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{bw / w:.6f} {bh / h:.6f}")
            (out_dir / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            save_pseudo_labels(out_dir, p.stem, dets, (w, h), lb)

        # COCO 伪标签
        if not args.no_coco:
            new_anns = boxes_to_coco(dets, rec, lb, ann_id_start=ann_id)
            for a in new_anns:
                a.pop("_raw_label", None)
            coco["annotations"].extend(new_anns)
            ann_id += len(new_anns)

        counts: Dict[int, int] = {}
        for d in dets:
            key = str(d["label"]).strip().lower().replace(" ", "_")
            cid = lb.name_to_id_map().get(key, lb.healthy_id())
            counts[cid] = counts.get(cid, 0) + 1
        mean_score = round(sum(float(d["score"]) for d in dets) / max(1, len(dets)), 4)
        entries.append({
            "image": p.name,
            "num_boxes": len(dets),
            "mean_score": mean_score,
            "class_counts": counts,
            "low_conf_boxes": sum(1 for d in dets if float(d["score"]) < 0.45),
        })
        if i <= 15 or i % 20 == 0:
            print(f"  [{i}/{len(paths)}] {p.name}：{len(dets)} 个框，平均置信度 {mean_score}")

    if not args.no_coco:
        coco_path = out_dir / "annotations.json"
        with open(coco_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)
        print(f"\nCOCO 伪标签：{coco_path}（{len(coco['annotations'])} 个框）")

    from src.train.openvocab import write_autolabel_report

    report = write_autolabel_report(
        out_dir / "autolabel_report.json", entries, lb, model_desc,
        {"box_threshold": box_thr, "text_threshold": float(args.text_threshold), "conf": float(args.conf)},
    )

    total_boxes = sum(e["num_boxes"] for e in entries)
    low = sorted(entries, key=lambda e: e["mean_score"])[:5]
    print("=" * 78)
    print("自动标注完成")
    print("=" * 78)
    print(f"处理图片：{len(entries)} 张（失败 {n_failed} 张）  伪标签框：{total_boxes} 个")
    print(f"YOLO 伪标签目录：{out_dir.resolve()}")
    print(f"复核报告：{report}")
    if low:
        print("\n建议优先复核（平均置信度最低的 5 张）：")
        for e in low:
            print(f"  - {e['image']}：{e['num_boxes']} 框，平均置信度 {e['mean_score']}，"
                  f"低置信框 {e['low_conf_boxes']} 个")
    print("\n人工复核流程（对应项目「AI 初筛 + 人工复核」闭环）：")
    print("  1) 农技端打开图片与 data/pseudo/<同名>.txt，逐框确认/删除/改类别；")
    print("  2) 复核通过后把 txt 与图片放进 data/raw/labels 与 data/raw/images；")
    print("  3) 重新跑 python -m src.data.prepare_data 生成正式训练集。")
    return 0


def _image_size(path: Path):
    """读图片宽高（PIL -> cv2 -> 兜底 640）。"""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.height), int(im.width)
    except Exception:
        pass
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is not None:
            return int(img.shape[0]), int(img.shape[1])
    except Exception:
        pass
    return 640, 640


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
