"""训练期间的健康检查：预测框尺寸 / 数量 vs 真值（快速判断"模型是否学对了方向"）。

为什么需要它：本机曾发生"训练目标编码错误"的事故 —— val_loss 从 22.5 正常降到 7.0，
但模型学到的是"贴在图左上角、0.2 像素大的框"，mAP 恒为 0。
**只看 loss 会误判，必须看预测框的尺寸分布**。

用法：
    python scripts/check_det_health.py --ckpt-dir models/det_rtdetrv2 --n 6
    python scripts/check_det_health.py --ckpt-dir models/det_rtdetrv2 --device cpu
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检测模型健康检查（框尺寸/数量 vs 真值）")
    ap.add_argument("--ckpt-dir", default="models/det_rtdetrv2", help="模型目录")
    ap.add_argument("--data-root", default="data/processed_stress2", help="数据集根（读 val 真值做对照）")
    ap.add_argument("--split", default="val", help="用哪个划分的图片")
    ap.add_argument("--n", type=int, default=6, help="检查多少张图")
    ap.add_argument("--device", default="cpu", help="推理设备（训练中建议用 cpu，避免抢显存）")
    args = ap.parse_args(argv)

    import cv2

    from src.train.predictors import build_predictor

    ckpt = ROOT / args.ckpt_dir
    if not (ckpt / "best.pt").is_file():
        print(f"[错误] 找不到 {ckpt / 'best.pt'}（还没开始训练？）")
        return 2

    # sidecar 是训练**结束时**才写的；训练中途做体检时先按 checkpoint.json 补一份，
    # 否则 build_predictor 会拒绝加载（本机实测踩过）。
    if not (ckpt / "model.onnx.json").is_file():
        try:
            import argparse as _ap
            import json as _json

            from src.core.labels import load_labels
            from src.train.exporter import save_sidecar
            from scripts.export_onnx import infer_sidecar_from_ckpt

            info = {}
            if (ckpt / "checkpoint.json").is_file():
                info = _json.loads((ckpt / "checkpoint.json").read_text(encoding="utf-8-sig"))
            ns = _ap.Namespace(
                task="detect",
                source=info.get("source"),
                imgsz=info.get("imgsz") or info.get("image_size"),
                note=None,
                mean=None, std=None,
                output_kind=None, output_names=None,
                device=None, backend_impl=None,
            )
            sc = infer_sidecar_from_ckpt(ckpt, load_labels(), ns)
            save_sidecar(sc, model_dir=ckpt)
            print(f"[信息] 已按 checkpoint.json 补写 sidecar：{ckpt / 'model.onnx.json'}")
        except Exception as exc:
            print(f"[警告] 补写 sidecar 失败（{type(exc).__name__}: {exc}）")
            print("       可手动执行：python -m scripts.export_onnx --ckpt-dir "
                  f"{args.ckpt_dir} --sidecar-only --task detect")

    data_root = ROOT / args.data_root
    img_dir = data_root / args.split / "images"
    coco_path = data_root / args.split / "annotations.json"
    files = sorted(glob.glob(str(img_dir / "*.jpg")))[: max(1, args.n)]
    if not files:
        print(f"[错误] {img_dir} 下没有图片")
        return 2

    # 真值框尺寸分布作为对照
    gt_med_w = gt_med_h = float("nan")
    if coco_path.is_file():
        coco = json.loads(coco_path.read_text(encoding="utf-8-sig"))
        ws = [float(a["bbox"][2]) for a in coco["annotations"]]
        hs = [float(a["bbox"][3]) for a in coco["annotations"]]
        if ws:
            gt_med_w, gt_med_h = statistics.median(ws), statistics.median(hs)

    predictor = build_predictor(str(ckpt), device=args.device)
    print("=" * 74)
    print(f"模型：{args.ckpt_dir}   设备：{args.device}")
    print(f"真值框尺寸中位数：{gt_med_w:.0f} x {gt_med_h:.0f} 像素（{args.split} 集）")
    print("=" * 74)

    pred_sizes: list[tuple[float, float]] = []
    n_box_total = 0
    for f in files:
        img = cv2.imread(f)
        raw = predictor._infer_one(img)          # 绕开阈值过滤，看模型原始输出
        n_box_total += len(raw)
        sizes = []
        for b in raw:
            x1, y1, x2, y2 = (float(v) for v in b["bbox"][:4])
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            pred_sizes.append((w, h))
            sizes.append(f"{w:.0f}x{h:.0f}")
        print(f"  {os.path.basename(f)[:26]:28s} {len(raw):3d} 框   前几个: {', '.join(sizes[:4])}")

    print("-" * 74)
    if not pred_sizes:
        print("✗ 模型没有输出任何框")
        return 1
    pw = statistics.median([s[0] for s in pred_sizes])
    ph = statistics.median([s[1] for s in pred_sizes])
    print(f"预测框数量：{n_box_total}（{len(files)} 张图，平均 {n_box_total/len(files):.1f} 个/图）")
    print(f"预测框尺寸中位数：{pw:.0f} x {ph:.0f} 像素")

    # 判定：预测框尺寸应与真值同量级；退化到个位数像素 = 训练目标编码有问题
    ok = True
    if pw < 10 or ph < 10:
        ok = False
        print("\n✗ 预测框退化成个位数像素 —— 典型症状：训练目标编码错误")
        print("  参见 docs/dataset-drone-plant-stress2.md 第 5.1 节；")
        print("  跑 python tests/test_det_targets.py 可验证目标编码是否正确。")
    elif gt_med_w == gt_med_w and (pw < gt_med_w * 0.2 or pw > gt_med_w * 5):
        print(f"\n⚠️ 预测框尺寸与真值差距较大（{pw:.0f} vs {gt_med_w:.0f} 像素），可能还没收敛")
    else:
        print(f"\n✓ 预测框尺寸与真值同量级（{pw:.0f} vs {gt_med_w:.0f} 像素），方向正确")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
