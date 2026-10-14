"""数据泄漏检查：train / val / test 之间是否存在重复或近似重复图像。

为什么必须查（本机事故背景）：
    该数据集的分类模型在 val 上拿到 **macro-F1 = 0.9998**，这个数字高得可疑。
    文件名的 `.rf.<hash>` 后缀说明数据来自 Roboflow 导出，而 Roboflow 常带增广副本；
    同时 `Image_001 / 004 / 005` 这样的连号很可能是**同一段航拍的相邻帧**。
    一旦相邻帧或增广副本被随机分到 train 与 val，val 就等价于"训练时见过的画面"，
    指标会虚高。这类问题在答辩时被问出来，比低指标更致命。

检查三类问题：
    1. **字节级完全相同**（MD5）：最直接的重复。
    2. **近似重复**（dHash + 颜色直方图）：缩放/重压缩/轻微裁剪的副本。
    3. **跨划分命中**：val/test 的图是否能在 train 里找到近似副本。

用法：
    python scripts/check_leakage.py --data-root data/processed_stress2
    python scripts/check_leakage.py --data-root data/processed_stress2 --threshold 6 --examples 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: dHash 汉明距离阈值：<= 该值判为近似重复（0 = 完全相同，越大越宽松）
DEFAULT_HAMMING = 6


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash(gray, hash_size: int = 8) -> np.ndarray:
    """差值哈希：比较相邻像素亮度，对亮度/对比度变化稳健。"""
    from PIL import Image

    small = gray.resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(small, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _color_hist(bgr, bins: int = 16) -> np.ndarray:
    """HSV 颜色直方图（归一化）：航拍相邻帧的颜色分布几乎一致。"""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    h = h.astype(np.float32).flatten()
    n = np.linalg.norm(h)
    return h / n if n > 0 else h


def _features(path: Path):
    import cv2
    from PIL import Image

    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    gray = Image.open(path).convert("L")
    return _dhash(gray), _color_hist(bgr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="数据泄漏检查（重复/近似重复图像）")
    ap.add_argument("--data-root", default="data/processed_stress2")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--threshold", type=int, default=DEFAULT_HAMMING, help="dHash 汉明距离阈值")
    ap.add_argument("--examples", type=int, default=6, help="打印多少组样例")
    ap.add_argument("--out-json", default="runs/leakage_report.json")
    args = ap.parse_args(argv)

    root = ROOT / args.data_root
    items: list[tuple[str, Path]] = []
    for sp in args.splits:
        d = root / sp / "images"
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")):
            items.append((sp, p))
    if not items:
        print(f"[错误] {root} 下没找到图片（检查 --data-root 与 --splits）")
        return 2

    print("=" * 78)
    print(f"数据泄漏检查：{args.data_root}")
    print(f"图片总数 {len(items)}，划分 {args.splits}，dHash 阈值 <= {args.threshold}")
    print("=" * 78)

    # ---- 1. 字节级重复 -------------------------------------------------
    md5_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sp, p in items:
        md5_map[_md5(p)].append((sp, p.name))
    exact_groups = [v for v in md5_map.values() if len(v) > 1]
    cross_exact = [g for g in exact_groups if len({s for s, _ in g}) > 1]
    print(f"\n【1】字节级完全相同：{len(exact_groups)} 组重复，其中**跨划分** {len(cross_exact)} 组")
    for g in cross_exact[: args.examples]:
        print("      " + " | ".join(f"{s}:{n[:28]}" for s, n in g))

    # ---- 2. 近似重复 ---------------------------------------------------
    feats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for sp, p in items:
        f = _features(p)
        if f is not None:
            feats[(sp, p.name)] = f
    print(f"\n【2】特征提取完成：{len(feats)} / {len(items)} 张")

    keys = list(feats)
    by_split: dict[str, list[int]] = defaultdict(list)
    for i, (sp, _) in enumerate(keys):
        by_split[sp].append(i)

    train_idx = by_split.get("train", [])
    holdout_idx = [i for sp in ("val", "test") for i in by_split.get(sp, [])]

    if not train_idx or not holdout_idx:
        print("[警告] 需要同时有 train 与 val/test 才能做跨划分检查")
        return 1

    # dHash 打包成整数便于快速汉明距离
    def pack(bits: np.ndarray) -> int:
        v = 0
        for b in bits:
            v = (v << 1) | int(b)
        return v

    packed = {i: pack(feats[keys[i]][0]) for i in range(len(keys))}
    hists = {i: feats[keys[i]][1] for i in range(len(keys))}

    # 对每个 holdout 图，找 train 中最相似的
    nearest: list[tuple[int, int, int, float]] = []   # (holdout_idx, train_idx, ham, cos)
    for hi in holdout_idx:
        best = None
        for ti in train_idx:
            ham = bin(packed[hi] ^ packed[ti]).count("1")
            cos = float(np.dot(hists[hi], hists[ti]))
            score = (ham, -cos)
            if best is None or score < best[0]:
                best = (score, ti, ham, cos)
        if best is not None:
            _, ti, ham, cos = best
            nearest.append((hi, ti, ham, cos))

    hams = np.array([n[2] for n in nearest])
    coss = np.array([n[3] for n in nearest])
    print(f"\n【3】每个 val/test 图在 train 中的最近邻（dHash 汉明距离）")
    print(f"      距离分布：min={hams.min()}  中位={np.median(hams):.0f}  "
          f"均值={hams.mean():.1f}  max={hams.max()}")
    for t in (2, 4, 6, 8, 10):
        n = int((hams <= t).sum())
        print(f"      距离 <= {t:2d} 的 holdout 图：{n:4d} / {len(nearest)}  ({n/len(nearest)*100:.1f}%)")

    # 颜色直方图余弦 >= 0.995 且 dHash <= 阈值 -> 高度疑似同一画面
    suspect = [(h, t, hm, c) for h, t, hm, c in nearest if hm <= args.threshold and c >= 0.99]
    print(f"\n【4】高度疑似泄漏（dHash<={args.threshold} 且颜色余弦>=0.99）："
          f"**{len(suspect)} / {len(nearest)} ({len(suspect)/len(nearest)*100:.1f}%)**")
    for h, t, hm, c in sorted(suspect, key=lambda x: (x[2], -x[3]))[: args.examples]:
        hs, hn = keys[h]
        ts, tn = keys[t]
        print(f"      {hs}:{hn[:30]:32s} <-> train:{tn[:30]:32s} (ham={hm}, cos={c:.4f})")

    # ---- 结论 ----------------------------------------------------------
    rate = len(suspect) / max(1, len(nearest))
    print("\n" + "=" * 78)
    if rate >= 0.3:
        print(f"✗ 结论：**严重泄漏**（{rate*100:.0f}% 的 val/test 图在 train 里有近似副本）")
        print("  该数据集上的 val 指标不可信，不能直接写进报告。")
        print("  处理建议：按「源画面」而不是「文件」重新划分，或改用与论文一致的官方划分。")
    elif rate >= 0.05:
        print(f"⚠️ 结论：**存在泄漏**（{rate*100:.0f}%），指标偏高，需说明或重划。")
    else:
        print(f"✓ 结论：未发现明显泄漏（{rate*100:.1f}%），val 指标基本可信。")
    print("=" * 78)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "data_root": str(args.data_root),
        "threshold": args.threshold,
        "num_images": len(items),
        "exact_duplicate_groups": len(exact_groups),
        "exact_cross_split_groups": len(cross_exact),
        "holdout_total": len(nearest),
        "suspect_leak": len(suspect),
        "leak_rate": rate,
        "hamming": {"min": int(hams.min()), "median": float(np.median(hams)),
                    "mean": float(hams.mean()), "max": int(hams.max())},
        "examples": [
            {"holdout": f"{keys[h][0]}:{keys[h][1]}", "train": f"{keys[t][0]}:{keys[t][1]}",
             "hamming": int(hm), "color_cos": round(float(c), 5)}
            for h, t, hm, c in sorted(suspect, key=lambda x: (x[2], -x[3]))[:50]
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
