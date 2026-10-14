"""生成 PlantSeg 10 类病种检测的可视化图（真值 vs 预测并排）。

为什么要单独写：`tools/make_figures.py` 的配色是给 2 类体系写死的
（`CLS_COLOR_HEX = {"healthy":..., "stressed":...}`），10 类病种下会全部落到默认色，
图上分不出类别。这里按类别数生成调色板。

用法：
    python tools/make_figures_plantseg.py --n 4 --out runs/figures
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CN_FONTS = ["Microsoft YaHei", "SimHei", "DengXian", "SimSun", "DejaVu Sans"]
#: 10 类病种配色（matplotlib hex）；BGR 由 hex 转出
PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/processed_plantseg")
    ap.add_argument("--det-dir", default="models/det_rtdetrv2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="runs/figures")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    import cv2
    from matplotlib import font_manager, pyplot as plt, rcParams

    have = {f.name for f in font_manager.fontManager.ttflist}
    for nm in CN_FONTS:
        if nm in have:
            rcParams["font.sans-serif"] = [nm]
            break
    rcParams["axes.unicode_minus"] = False

    from src.labels import LABELS
    from src.train.predictors import build_predictor

    root = ROOT / args.data_root
    coco = json.loads((root / args.split / "annotations.json").read_text(encoding="utf-8"))
    names = [c["name"] for c in coco["categories"]]
    color_of = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
    anns: dict[int, list] = {}
    for a in coco["annotations"]:
        anns.setdefault(int(a["image_id"]), []).append(a)

    cands = [im for im in coco["images"] if anns.get(int(im["id"]))]
    random.seed(11)
    picks = random.sample(cands, min(args.n, len(cands)))

    predictor = build_predictor(args.det_dir, device=args.device)
    print(f"[信息] 预测器阈值 conf={getattr(predictor, 'conf_threshold', None)}  类别数={len(names)}")

    fig, axes = plt.subplots(len(picks), 2, figsize=(13.5, 3.6 * len(picks)))
    if len(picks) == 1:
        axes = np.array([axes])

    for r, im in enumerate(picks):
        nm = Path(str(im["file_name"])).name
        p = root / args.split / "images" / nm
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue

        left = bgr.copy()
        for a in anns[int(im["id"])]:
            x, y, w, h = (float(v) for v in a["bbox"][:4])
            cn = names[int(a["category_id"])]
            cv2.rectangle(left, (int(x), int(y)), (int(x + w), int(y + h)), _hex_to_bgr(color_of[cn]), 2)

        right = bgr.copy()
        res = predictor.predict([bgr])[0]
        dets = res.get("detections") or []
        for b in dets:
            x1, y1, x2, y2 = (int(v) for v in b["bbox"][:4])
            cn = str(b.get("disease_name", "?"))
            cv2.rectangle(right, (x1, y1), (x2, y2), _hex_to_bgr(color_of.get(cn, "#ffffff")), 2)

        import cv2 as _cv
        axes[r][0].imshow(_cv.cvtColor(left, _cv.COLOR_BGR2RGB))
        axes[r][0].set_title(f"真值  {nm[:26]}\n{len(anns[int(im['id'])])} 个标注框", fontsize=9)
        axes[r][1].imshow(_cv.cvtColor(right, _cv.COLOR_BGR2RGB))
        axes[r][1].set_title(
            f"预测  {len(dets)} 个框（{res.get('disease_name')} / {res.get('severity')}）", fontsize=9)
        for c in (0, 1):
            axes[r][c].axis("off")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=color_of[n], label=n) for n in names]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("PlantSeg 10 类病种检测：真值 vs 模型预测（按源 URL 重划，泄漏 2.5%）",
                 fontsize=13, y=0.997)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    out = ROOT / args.out / "10类病种_检测效果.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
