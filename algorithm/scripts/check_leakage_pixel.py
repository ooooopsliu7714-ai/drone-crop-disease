"""泄漏检查第二阶段：对候选图对做**像素级验证**，区分"真副本"与"同一块田的不同帧"。

为什么需要第二阶段：dHash 只有 64 位、在 8×8 分辨率上比较，同一块地的不同航拍帧
也可能得到距离 0（大尺度结构相似）。本机实测：中位距离 = 0、90.9% 的 val 图都能在
train 里找到"dHash 距离<=2"的图——但这个数字**既可能是真泄漏，也可能只是同类场景**。
只有像素级比对才能定性，否则会误报（把好数据判成泄漏）或漏报（放过真泄漏）。

判据（真副本应同时满足）：
  · MAE（平均绝对像素差）很小；
  · 允许 ±shift 像素平移后仍然很小（增广可能带轻微裁剪/平移）；
  · 结构相关系数很高。

用法：
    python scripts/check_leakage_pixel.py --data-root data/processed_stress2 --top 120
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: MAE 判据：真副本（含 JPEG 重压缩/轻微增广）实测应远低于此值
MAE_DUP = 6.0
#: 平移搜索范围（像素）
SHIFTS = (-8, -4, 0, 4, 8)


def _load_gray(path: Path, size: int = 256):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def _best_mae(a: np.ndarray, b: np.ndarray) -> tuple[float, tuple[int, int]]:
    """在 ±SHIFTS 平移范围内找最小 MAE（增广可能带轻微位移）。"""
    best, best_s = 1e9, (0, 0)
    h, w = a.shape
    m = 16
    for dy in SHIFTS:
        for dx in SHIFTS:
            aa = a[m : h - m, m : w - m]
            bb = b[m + dy : h - m + dy, m + dx : w - m + dx]
            mae = float(np.mean(np.abs(aa - bb)))
            if mae < best:
                best, best_s = mae, (dy, dx)
    return best, best_s


def _sample_mode(args) -> int:
    """估计真实泄漏率：随机抽 holdout 图，各自找 train 最近邻并做像素验证。

    为什么不能只看最可疑的那批：第一阶段按"dHash 小 + 颜色像"筛出来的图对
    天然都是真副本，用它们的比例（6.1%）当泄漏率是**有偏**的。
    正确做法是随机抽样，对样本里每一张都做像素验证，才能得到无偏估计。
    """
    import random

    import numpy as np

    root = ROOT / args.data_root
    tr = sorted((root / "train" / "images").glob("*.jpg"))
    ho: list[tuple[str, Path]] = []
    for sp in ("val", "test"):
        for p in sorted((root / sp / "images").glob("*.jpg")):
            ho.append((sp, p))
    if not tr or not ho:
        print("[错误] 需要 train 与 val/test 目录")
        return 2

    random.seed(args.seed)
    if args.emit_clean_list:
        # 全量模式：逐张判定，输出"干净"（无 train 副本）名单
        sample = ho
        print(f"[信息] 全量模式：将检查全部 {len(sample)} 张 holdout 图以产出干净名单")
    else:
        sample = random.sample(ho, min(args.sample, len(ho)))

    # train 的 dHash 索引（复用第一阶段脚本的哈希实现，避免两份实现漂移）
    from PIL import Image

    from scripts.check_leakage import _dhash

    def _dh(p: Path):
        g = Image.open(p).convert("L")
        return _dhash(g)

    train_bits = [(_dh(p), p) for p in tr]

    print("=" * 78)
    print(f"真实泄漏率估计：随机 {len(sample)} 张 holdout 图 × train {len(tr)} 张")
    print(f"判据：允许 ±8 像素平移后 MAE < {MAE_DUP}")
    print("=" * 78)

    maes, dups = [], []
    for sp, p in sample:
        bits = _dh(p)
        # 先按 dHash 取前 20 个候选，再做像素验证
        cand = sorted(train_bits, key=lambda t: int(np.count_nonzero(bits != t[0])))[:20]
        ga = _load_gray(p)
        if ga is None:
            continue
        best = (1e9, None, None)
        for tbits, tp in cand:
            gb = _load_gray(tp)
            if gb is None:
                continue
            mae, sh = _best_mae(ga, gb)
            if mae < best[0]:
                best = (mae, tp, sh)
        mae = best[0]
        maes.append(mae)
        if mae < MAE_DUP:
            dups.append({"holdout": f"{sp}:{p.name}", "train": best[1].name if best[1] else "?",
                         "mae": round(mae, 2), "shift": list(best[2] or (0, 0))})

    a = np.array(maes)
    print(f"\n  最近邻 MAE 分布：min={a.min():.2f}  中位={np.median(a):.2f}  "
          f"均值={a.mean():.2f}  max={a.max():.2f}")
    for t in (MAE_DUP, 10, 20, 40):
        n = int((a < t).sum())
        print(f"    MAE < {t:5.1f}：{n:4d} / {len(a)}  ({n/len(a)*100:.1f}%)")

    rate = len(dups) / max(1, len(a))
    print("\n" + "=" * 78)
    print(f"真实泄漏率估计：**{len(dups)} / {len(a)} = {rate*100:.1f}%**")

    # 输出"干净"名单：这些图在 train 里没有像素级副本，用它们评估才是诚实的
    if args.emit_clean_list:
        leaked_names = {d["holdout"].split(":", 1)[1] for d in dups}
        clean = [f"{sp}/{p.name}" for sp, p in sample if p.name not in leaked_names]
        cl = Path(args.emit_clean_list)
        cl.parent.mkdir(parents=True, exist_ok=True)
        cl.write_text("\n".join(clean) + "\n", encoding="utf-8")
        print(f"  干净名单已写入：{cl}（{len(clean)} 张，占 {len(clean)/max(1,len(sample))*100:.1f}%）")
        print("  用它对模型出诚实指标：")
        print(f"    python -m scripts.eval --gt {args.data_root}/val/annotations.json "
              f"--ckpt-dir models/det_rtdetrv2 --only-images {cl}")
    if dups:
        print("  样例：")
        for d in sorted(dups, key=lambda x: x["mae"])[:5]:
            print(f"    MAE={d['mae']:6.2f}  {d['holdout']}  <->  train:{d['train']}")
    print(f"  → 该比例直接对应 val 指标被高估的上限：约 {rate*100:.1f}% 的样本是'见过的画面'。")
    print("=" * 78)

    out = Path(args.out_json)
    out.write_text(json.dumps({
        "mode": "random_sample",
        "sampled": len(a),
        "duplicates": len(dups),
        "leak_rate": rate,
        "mae_stats": {"min": float(a.min()), "median": float(np.median(a)),
                      "mean": float(a.mean()), "max": float(a.max())},
        "examples": sorted(dups, key=lambda x: x["mae"])[:30],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入：{out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="泄漏候选图对的像素级验证")
    ap.add_argument("--data-root", default="data/processed_stress2")
    ap.add_argument("--report", default="runs/leakage_report.json", help="第一阶段报告")
    ap.add_argument("--top", type=int, default=120, help="验证最可疑的前 N 对")
    ap.add_argument("--sample", type=int, default=0,
                    help="改为此模式：随机抽 N 张 holdout 图，各自找 train 最近邻再做像素验证，"
                         "用于**估计真实泄漏率**（而不是只看最可疑的那批）")
    ap.add_argument("--emit-clean-list", default=None,
                    help="把**没有 train 副本**的 holdout 图名单写到该文件（供 eval.py 出诚实指标）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="runs/leakage_pixel_report.json")
    args = ap.parse_args(argv)

    # 注意：这里必须同时判断 emit_clean_list，否则默认 sample=0 会掉进下面的
    # --top 模式（只验证"最可疑的 N 对"），跑出来的泄漏率依然是有偏的。
    if args.sample or args.emit_clean_list:
        return _sample_mode(args)

    rep_path = Path(args.report)
    if not rep_path.is_file():
        print(f"[错误] 找不到第一阶段报告 {rep_path}（先跑 check_leakage.py）")
        return 2
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    root = ROOT / args.data_root

    pairs = rep.get("examples", [])[: args.top]
    if not pairs:
        print("[信息] 第一阶段报告里没有可疑图对")
        return 0

    print("=" * 78)
    print(f"像素级验证：{len(pairs)} 对候选")
    print(f"判据：允许 ±8 像素平移后 MAE < {MAE_DUP} 视为真副本")
    print("=" * 78)

    dup, similar, distinct = [], [], []
    for item in pairs:
        hs, hn = item["holdout"].split(":", 1)
        ts, tn = item["train"].split(":", 1)
        pa, pb = root / hs / "images" / hn, root / ts / "images" / tn
        ga, gb = _load_gray(pa), _load_gray(pb)
        if ga is None or gb is None:
            continue
        mae, sh = _best_mae(ga, gb)
        rec = {"holdout": item["holdout"], "train": item["train"],
               "dhash_ham": item["hamming"], "color_cos": item["color_cos"],
               "mae": round(mae, 2), "shift": list(sh)}
        if mae < MAE_DUP:
            dup.append(rec)
        elif mae < MAE_DUP * 3:
            similar.append(rec)
        else:
            distinct.append(rec)

    print(f"\n  真副本      (MAE < {MAE_DUP})            : {len(dup)} 对")
    print(f"  较相似      ({MAE_DUP} <= MAE < {MAE_DUP*3}) : {len(similar)} 对")
    print(f"  明显不同    (MAE >= {MAE_DUP*3})          : {len(distinct)} 对")

    for tag, group in (("真副本", dup), ("较相似", similar)):
        if not group:
            continue
        print(f"\n  【{tag}】按 MAE 升序前 8 对：")
        for r in sorted(group, key=lambda x: x["mae"])[:8]:
            print(f"      MAE={r['mae']:6.2f} 平移={tuple(r['shift'])}  "
                  f"dHash={r['dhash_ham']} cos={r['color_cos']:.4f}")
            print(f"        {r['holdout']}")
            print(f"        {r['train']}")

    print("\n" + "=" * 78)
    n_holdout = int(rep.get("holdout_total") or 0) or 1
    if dup:
        print(f"✗ 确认泄漏：{len(dup)} 对真副本（占 holdout 的 {len(dup)/n_holdout*100:.1f}%）")
        print("  这些图在 train 与 val/test 之间几乎是同一画面，val 指标会虚高。")
    elif len(similar) > len(dup):
        print("⚠️ 未发现像素级真副本，但存在较多'高度相似'图（可能是同一块田的相邻帧）。")
        print("  这类相似会**高估**指标，但程度远低于真副本；建议在报告中说明，")
        print("  或按'源画面'重新划分以保证严谨。")
    else:
        print("✓ 未发现像素级真副本：此前 dHash 的高命中率来自同类场景的全局结构相似，")
        print("  不是数据泄漏。")

    out = Path(args.out_json)
    out.write_text(json.dumps({
        "checked": len(pairs),
        "duplicates": dup,
        "similar": similar,
        "distinct_count": len(distinct),
        "mae_dup_threshold": MAE_DUP,
        "verdict": "confirmed_leak" if dup else ("similar_only" if similar else "no_leak"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
