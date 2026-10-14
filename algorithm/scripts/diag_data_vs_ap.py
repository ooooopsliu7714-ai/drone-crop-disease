"""诊断：每类「数据量 / 目标尺寸」与「AP」的相关性。

用途：回答"再找数据集来训练能不能改善精度"——
如果 AP 与数据量强正相关，加数据有效；如果无关甚至负相关，加数据没用。

用法：python scripts/diag_data_vs_ap.py --eval runs/eval_768.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-coco", default="data/processed_plantseg/train/annotations.json")
    ap.add_argument("--eval", default="runs/eval_768.json")
    args = ap.parse_args(argv)

    tr = json.loads((ROOT / args.train_coco).read_text(encoding="utf-8"))
    names = [c["name"] for c in tr["categories"]]
    boxes: dict[str, list] = defaultdict(list)
    imgs: dict[str, set] = defaultdict(set)
    for a in tr["annotations"]:
        n = names[int(a["category_id"])]
        boxes[n].append(a["bbox"])
        imgs[n].add(int(a["image_id"]))

    ev = json.loads((ROOT / args.eval).read_text(encoding="utf-8"))
    ap50: dict[str, float] = {}
    # per_class 可能是「列表 of dict」也可能是「dict[名 -> dict]」，两种都要能吃下
    raw = ev.get("per_class")
    items = raw.items() if isinstance(raw, dict) else ((None, r) for r in (raw or []))
    for key, r in items:
        if not isinstance(r, dict):
            continue
        nm = (r.get("name") or r.get("disease_name") or key
              or names[int(r.get("class_id", r.get("cid", 0)))])
        v = r.get("ap50", r.get("ap", r.get("AP50")))
        if v is not None:
            ap50[str(nm)] = float(v)

    print(f"{'类别':<32}{'训练图':>7}{'训练框':>8}{'框中位面积':>12}{'valAP50':>10}")
    print("-" * 72)
    rows = []
    for n in names:
        areas = [b[2] * b[3] for b in boxes[n]]
        med = statistics.median(areas) if areas else 0.0
        a = ap50.get(n)
        print(f"{n:<32}{len(imgs[n]):>7}{len(boxes[n]):>8}{med:>12.0f}"
              f"{(f'{a:.4f}' if a is not None else '?'):>10}")
        if a is not None:
            rows.append((len(imgs[n]), len(boxes[n]), med, a, n))

    print()
    print("相关系数 (Pearson)：")
    c_img = pearson([r[0] for r in rows], [r[3] for r in rows])
    c_box = pearson([r[1] for r in rows], [r[3] for r in rows])
    c_area = pearson([r[2] for r in rows], [r[3] for r in rows])
    print(f"  训练图数   vs AP50 : {c_img:+.3f}   "
          f"{'数据多→精度高' if c_img > 0.3 else '无正相关（加数据未必有用）'}")
    print(f"  训练框数   vs AP50 : {c_box:+.3f}")
    print(f"  框中位面积 vs AP50 : {c_area:+.3f}   "
          f"{'目标越大越好检（尺寸是主因）' if c_area > 0.5 else ''}")

    print()
    top = sorted(rows, key=lambda r: -r[3])[:3]
    bot = sorted(rows, key=lambda r: r[3])[:3]
    print("最易检的 3 类：", "、".join(f"{r[4]}({r[0]}图/AP{r[3]:.2f})" for r in top))
    print("最难检的 3 类：", "、".join(f"{r[4]}({r[0]}图/AP{r[3]:.2f})" for r in bot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
