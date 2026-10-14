"""分类模型的泄漏影响评估：在"无 train 副本"的干净图块上重算指标。

为什么必须做：分类模型在 val 上拿到 **macro-F1 = 0.9998**，而这个数据集实测
**28.4% 的 val 图在 train 里有像素级副本**（MAE < 6，只差 JPEG 重压缩）。
分类是在**按 GT 框裁出的图块**上训练的——若整张图重复，则同一批图块几乎原样出现在
训练集里，val 指标必然虚高。只有剔除这些图后重算，才是有意义的泛化能力。

用法：
    python scripts/check_cls_leakage.py --data-root data/processed_stress2 \
        --ckpt-dir models/cls_dinov2 --clean-list runs/clean_val.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="分类模型在无泄漏子集上的诚实指标")
    ap.add_argument("--data-root", default="data/processed_stress2")
    ap.add_argument("--ckpt-dir", default="models/cls_dinov2")
    ap.add_argument("--clean-list", default="runs/clean_val.txt",
                    help="check_leakage_pixel.py --emit-clean-list 产出的干净图名单")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)

    import torch

    from src.data.torch_datasets import ImageClassifyDataset
    from src.train.classifiers import evaluate_classifier
    from src.train.predictors import build_predictor

    root = ROOT / args.data_root
    val_json = root / "val" / "annotations.json"
    if not val_json.is_file():
        print(f"[错误] 缺少 {val_json}")
        return 2

    # 复用项目自己的 build_predictor：它会读 sidecar 并组装出与训练一致的模型
    pred = build_predictor(str(ROOT / args.ckpt_dir), device=args.device)
    model = getattr(pred, "clf", None) or getattr(pred, "model", None)
    if model is None:
        print(f"[错误] 从 {args.ckpt_dir} 没拿到分类模型（clf/model 都为空）")
        return 2
    device = getattr(pred, "device", args.device)
    image_size = int(getattr(pred, "imgsz", 224) or 224)

    full = ImageClassifyDataset.from_coco(
        val_json,
        image_roots=[root / "val" / "images"],
        image_size=image_size,
        train=False,
    )
    print(f"完整 val 图块数：{len(full.samples)}")

    # 干净名单：只保留来自"无 train 副本"图片的图块
    keep_imgs: set[str] | None = None
    cl = Path(args.clean_list)
    if cl.is_file():
        keep_imgs = {
            Path(line.strip()).name
            for line in cl.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.startswith("#")
        }
        print(f"干净名单：{len(keep_imgs)} 张图")

    def subset(which: str) -> ImageClassifyDataset:
        """which: 'all' | 'clean'（在干净名单里）| 'leaked'（不在名单里）。

        注意 don't 把 clean/leaked 写反：clean_list 里列的是**没有 train 副本**的图，
        所以 `name in keep_imgs` 为真 = 干净。早期版本用布尔 keep 传参导致两者颠倒，
        会把"干净子集"的结果标成"含泄漏"，结论虽然同向但标签是错的。
        """
        if keep_imgs is None or which == "all":
            samples = full.samples
        else:
            want_clean = which == "clean"
            samples = [
                s for s in full.samples
                if (Path(str(s["image_path"])).name in keep_imgs) == want_clean
            ]
        return ImageClassifyDataset(
            samples,
            image_size=image_size,
            crop_margin=getattr(full, "crop_margin", 0.1),
            train=False,
            mean=getattr(full, "mean", (0.485, 0.456, 0.406)),
            std=getattr(full, "std", (0.229, 0.224, 0.225)),
        )

    print("=" * 74)
    print(f"分类模型：{args.ckpt_dir}   设备：{device}   图块尺寸：{image_size}")
    print("=" * 74)

    results = {}
    for tag, which in (
        ("干净图块（无 train 副本）", "clean"),
        ("泄漏图块（train 里有副本）", "leaked"),
        ("全部 val 图块", "all"),
    ):
        ds = subset(which)
        if len(ds) == 0:
            print(f"  {tag}: 无样本，跳过")
            continue
        m = evaluate_classifier(model, ds, device=device, max_batches=0)
        results[tag] = m
        print(f"\n  【{tag}】{len(ds)} 个图块")
        print(f"     accuracy = {m.get('accuracy', 0):.4f}")
        print(f"     macro_F1 = {m.get('macro_f1', 0):.4f}")
        for pc in (m.get("per_class") or [])[:3]:
            print(f"       class {pc.get('class_id')}: P={pc.get('precision',0):.3f} "
                  f"R={pc.get('recall',0):.3f} F1={pc.get('f1',0):.3f} (n={pc.get('support',0)})")

    clean = results.get("干净图块（无 train 副本）")
    leaked = results.get("泄漏图块（train 里有副本）")
    if clean and leaked:
        d_f1 = leaked.get("macro_f1", 0) - clean.get("macro_f1", 0)
        print("\n" + "=" * 74)
        print(f"macro_F1 差距 = {d_f1:+.4f}（泄漏图块 - 干净图块）")
        print(f"**报告应采用的诚实数字：macro_F1 = {clean.get('macro_f1', 0):.4f}，"
              f"accuracy = {clean.get('accuracy', 0):.4f}**（干净图块上）")
        if d_f1 > 0.05:
            print("✗ 泄漏对分类指标影响显著：必须用干净子集的数字。")
        elif d_f1 > 0.01:
            print("⚠️ 泄漏有可测影响，建议报告干净子集数字并说明。")
        else:
            print("✓ 泄漏影响很小 —— 0.9998 不是靠重复图刷出来的，")
            print("  而是'健康 vs 受胁迫'的图块分类本身很容易（两类视觉差异大）。")
        print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
