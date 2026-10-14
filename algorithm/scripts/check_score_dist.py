"""诊断：模型输出的置信度分布（判断"分数塌陷"还是"阈值设太高"）。

背景：本机实测到模型训练后 >0.25 的框从 52 框/图 掉到 0.2 框/图，
但框坐标一直正确。需要区分两种可能：
  ① 分数整体下移（多数落在 0.05~0.25），说明是阈值/置信度标定问题；
  ② 分数整体接近 0，说明模型塌陷（学成了"全部无目标"）。

用法：python scripts/check_score_dist.py --ckpt-dir models/det_rtdetrv2 --n 4
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="模型输出置信度分布诊断")
    ap.add_argument("--ckpt-dir", default="models/det_rtdetrv2")
    ap.add_argument("--data-root", default="data/processed_stress2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    import cv2
    import torch

    from src.train.predictors import _bgr_to_pil, build_predictor
    from src.train.sidecar_bootstrap import ensure_sidecar

    # sidecar 要训练结束才写；中途体检时先按 checkpoint.json 补一份
    ok, msg = ensure_sidecar(args.ckpt_dir, task="detect")
    print(f"[信息] {msg}")
    if not ok:
        return 2

    predictor = build_predictor(args.ckpt_dir, device=args.device)
    model = predictor.model
    processor = predictor.processor
    model.eval()

    files = sorted(glob.glob(str(ROOT / args.data_root / args.split / "images" / "*.jpg")))[: args.n]
    if not files:
        print("[错误] 找不到图片")
        return 2

    # 训练时输出了多少 GT 框（平均），用于对照
    print("=" * 74)
    print("每张图：全部分数分布（模型共 300 个 query）")
    print("=" * 74)
    for f in files:
        img = cv2.imread(f)
        pil = _bgr_to_pil(img)
        inputs = processor(images=pil, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            outs = model(**inputs)
        logits = outs.logits[0]                       # (Q, C)
        probs = torch.sigmoid(logits)                 # RT-DETR 用 sigmoid 打分
        best = probs.max(dim=-1).values.cpu().numpy()  # 每个 query 的最高类得分
        top = np.sort(best)[::-1][:10]
        print(f"  {Path(f).name[:24]:26s} max={best.max():.4f}  "
              f">0.25:{(best > 0.25).sum():3d}  >0.1:{(best > 0.1).sum():3d}  "
              f">0.05:{(best > 0.05).sum():3d}  >0.01:{(best > 0.01).sum():3d}")
        print(f"      top10 分数: {', '.join(f'{v:.4f}' for v in top)}")

    print("-" * 74)
    print("判读方式：")
    print("  · top10 分布在 0.1~0.4  -> 分数整体偏低但可用（阈值/标定问题）")
    print("  · top10 都 < 0.05       -> 模型塌陷（学成'全部无目标'），需换训练配置")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
