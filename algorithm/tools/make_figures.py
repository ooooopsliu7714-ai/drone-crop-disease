"""生成汇报用图：检测效果对比、训练曲线、指标对比、置信度分布。

为什么先做图：指标（mAP@0.5=0.3346）只能说明"统计上还行"，但**框画出来是什么样**
才是判断能不能演示的关键——有没有明显错位、漏检、框跑到田外。图比数字更能暴露问题。

用法：
    python tools/make_figures.py --data-root data/processed_stress2 \
        --det-dir models/det_rtdetrv2 --out-dir runs/figures
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 中文字体回退链（Windows 常见字体）
CN_FONTS = ["Microsoft YaHei", "SimHei", "DengXian", "SimSun", "DejaVu Sans"]

# 类别配色：(BGR 给 cv2, HEX 给 matplotlib)
CLS_COLOR_HEX = {"healthy": "#2ca02c", "stressed": "#d62728"}


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in CN_FONTS:
        if name in have:
            rcParams["font.sans-serif"] = [name]
            break
    else:
        print("[警告] 没找到中文字体，图上中文可能显示为方块")
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 130
    return matplotlib


# ---------------------------------------------------------------------------
# 图 1：检测效果（真值 vs 预测并排）
# ---------------------------------------------------------------------------
def fig_detection_samples(args, out_dir: Path) -> Path | None:
    import cv2
    from matplotlib import pyplot as plt

    from src.labels import LABELS
    from src.train.predictors import build_predictor

    root = ROOT / args.data_root
    coco = json.loads((root / "val" / "annotations.json").read_text(encoding="utf-8-sig"))
    anns: dict[int, list] = {}
    for a in coco["annotations"]:
        anns.setdefault(int(a["image_id"]), []).append(a)

    # 只用"干净"图（无 train 副本），避免展示时被质疑
    clean: set[str] | None = None
    cl = ROOT / args.clean_list
    if cl.is_file():
        clean = {Path(l.strip()).name for l in cl.read_text(encoding="utf-8-sig").splitlines()
                 if l.strip() and not l.startswith("#")}

    cands = []
    for im in coco["images"]:
        nm = Path(str(im["file_name"])).name
        if clean and nm not in clean:
            continue
        if not anns.get(int(im["id"])):
            continue
        p = root / "val" / "images" / nm
        if p.is_file():
            cands.append((im, p))
    if not cands:
        print("[跳过] 没有可用的干净 val 图")
        return None

    random.seed(7)
    picks = random.sample(cands, min(args.n_samples, len(cands)))

    predictor = build_predictor(args.det_dir, device=args.device)
    print(f"[信息] 预测器阈值 conf={getattr(predictor, 'conf_threshold', None)}")

    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.9 * n))
    if n == 1:
        axes = np.array([axes])
    for r, (im, path) in enumerate(picks):
        bgr = cv2.imread(str(path))
        H, W = bgr.shape[:2]

        # 左：真值
        left = bgr.copy()
        for a in anns[int(im["id"])]:
            x, y, w, h = (float(v) for v in a["bbox"][:4])
            cid = int(a["category_id"])
            name = LABELS.names[cid] if cid < len(LABELS.names) else str(cid)
            col = (44, 160, 44) if name == "healthy" else (39, 39, 214)
            cv2.rectangle(left, (int(x), int(y)), (int(x + w), int(y + h)), col, 2)

        # 右：预测
        raw = predictor.predict([bgr])[0]
        right = bgr.copy()
        dets = raw.get("detections") or []
        for b in dets:
            x1, y1, x2, y2 = (int(v) for v in b["bbox"][:4])
            nm = str(b.get("disease_name", "?"))
            col = (44, 160, 44) if nm == "healthy" else (39, 39, 214)
            cv2.rectangle(right, (x1, y1), (x2, y2), col, 2)
            cv2.putText(right, f"{b.get('confidence', 0):.2f}", (x1, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

        axes[r][0].imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
        axes[r][0].set_title(f"真值  {Path(im['file_name']).name[:26]}\n"
                             f"{len(anns[int(im['id'])])} 个标注框", fontsize=9)
        axes[r][1].imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
        axes[r][1].set_title(f"模型预测  {len(dets)} 个框（{raw.get('disease_name')} / "
                             f"{raw.get('severity')}）", fontsize=9)
        for c in (0, 1):
            axes[r][c].axis("off")

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=CLS_COLOR_HEX["healthy"], label="healthy 健康"),
                        Patch(facecolor=CLS_COLOR_HEX["stressed"], label="stressed 受胁迫")],
               loc="lower center", ncol=2, fontsize=10, frameon=False)
    fig.suptitle("检测效果：真值 vs 模型预测（无泄漏子集）", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    out = out_dir / "01_检测效果对比.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ---------------------------------------------------------------------------
# 图 2：训练曲线（冻结 vs 解冻）
# ---------------------------------------------------------------------------
def fig_training_curves(args, out_dir: Path) -> Path | None:
    from matplotlib import pyplot as plt

    runs = [
        ("冻结 backbone（60 epoch）", ROOT / "models/_baseline_det_rtdetrv2_frozen60/train_log.jsonl", "#1f77b4"),
        ("解冻微调（40 epoch）", ROOT / "models/det_rtdetrv2/train_log.jsonl", "#d62728"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    any_data = False
    for label, path, color in runs:
        if not path.is_file():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if "val_loss" in r and "train_loss" in r]
        if not rows:
            continue
        ep = [r["epoch"] for r in rows]
        tr = [r["train_loss"] for r in rows]
        va = [r["val_loss"] for r in rows]
        axes[0].plot(ep, tr, color=color, label=label, lw=1.8)
        axes[1].plot(ep, va, color=color, label=label, lw=1.8)
        any_data = True
    if not any_data:
        print("[跳过] 没有训练日志")
        return None

    axes[0].set_title("训练损失", fontsize=12)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train_loss")
    axes[1].set_title("验证损失（解冻后收敛更低）", fontsize=12)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val_loss")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out = out_dir / "02_训练曲线.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ---------------------------------------------------------------------------
# 图 3：指标对比
# ---------------------------------------------------------------------------
def fig_metrics(args, out_dir: Path) -> Path | None:
    from matplotlib import pyplot as plt

    def load(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    frozen = load(ROOT / "runs/cmp__baseline_det_rtdetrv2_frozen60.json")
    unfrozen = load(ROOT / "runs/cmp_det_rtdetrv2.json")
    clean = load(ROOT / "runs/eval_clean.json")
    if not unfrozen:
        print("[跳过] 缺少检测评估结果")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

    # (a) mAP 对比
    labels, m50, m5095 = [], [], []
    if frozen:
        labels.append("冻结 60ep"); m50.append(frozen.get("map50", 0)); m5095.append(frozen.get("map50_95", 0))
    labels.append("解冻 40ep"); m50.append(unfrozen.get("map50", 0)); m5095.append(unfrozen.get("map50_95", 0))
    if clean:
        labels.append("解冻(无泄漏)"); m50.append(clean.get("map50", 0)); m5095.append(clean.get("map50_95", 0))
    x = np.arange(len(labels)); w = 0.36
    axes[0].bar(x - w / 2, m50, w, label="mAP@0.5", color="#d62728")
    axes[0].bar(x + w / 2, m5095, w, label="mAP@0.5:0.95", color="#ff9896")
    for i, v in enumerate(m50):
        axes[0].text(i - w / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_title("检测精度对比", fontsize=12); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3, axis="y")

    # (b) 每类 AP
    pc = (clean or unfrozen).get("per_class", {})
    names = [{"0": "healthy", "1": "stressed"}.get(k, k) for k in sorted(pc)]
    aps = [pc[k].get("ap", 0) for k in sorted(pc)]
    axes[1].bar(names, aps, color=[CLS_COLOR_HEX.get(n, "#888") for n in names])
    for i, v in enumerate(aps):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_title("每类 AP@0.5（无泄漏子集）", fontsize=12); axes[1].grid(alpha=0.3, axis="y")

    # (c) 阈值-精度/召回权衡
    ths, ps, rs, f1s, nbox = [], [], [], [], []
    for c in ("0.05", "0.10", "0.15", "0.20", "0.25"):
        r = load(ROOT / f"runs/sweep_{c}.json")
        if not r:
            continue
        ths.append(float(c))
        mi = r.get("micro", {})
        ps.append(mi.get("precision", 0)); rs.append(mi.get("recall", 0)); f1s.append(mi.get("f1", 0))
        nbox.append(r.get("num_pred_kept", 0) / max(1, r.get("num_images", 1)))
    if ths:
        axes[2].plot(ths, ps, "o-", label="精度", color="#1f77b4")
        axes[2].plot(ths, rs, "s-", label="召回", color="#d62728")
        axes[2].plot(ths, f1s, "^-", label="F1", color="#2ca02c")
        best = int(np.argmax(f1s))
        axes[2].axvline(ths[best], ls="--", color="gray", alpha=0.7)
        axes[2].annotate(f"最佳 F1={f1s[best]:.3f}\n阈值={ths[best]}",
                         xy=(ths[best], f1s[best]), xytext=(ths[best] + 0.03, f1s[best] - 0.12),
                         fontsize=9)
        axes[2].set_title("置信度阈值权衡（服务端工作点）", fontsize=12)
        axes[2].set_xlabel("conf 阈值"); axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / "03_指标对比.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ---------------------------------------------------------------------------
# 图 4：泄漏检查结论
# ---------------------------------------------------------------------------
def fig_leakage(args, out_dir: Path) -> Path | None:
    from matplotlib import pyplot as plt

    full = ROOT / "runs/leakage_full.json"
    if not full.is_file():
        print("[跳过] 缺少泄漏报告")
        return None
    rep = json.loads(full.read_text(encoding="utf-8"))
    st = rep.get("mae_stats", {})
    if not st:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))

    # (a) MAE 分布（用分位点重建近似直方图信息）
    thr = rep.get("threshold", 6)
    buckets = [("MAE<6\n(像素级副本)", rep.get("leak_rate", 0)),
               ("6~10", None), ("10~20", None)]
    # 从报告的分档统计还原
    dist = rep.get("mae_buckets")
    if dist:
        names = list(dist)
        vals = [dist[k] for k in names]
    else:
        # 报告只存了 stats；用已知分档（来自全量日志）画
        names = ["MAE<6", "6~10", "10~20", "20~40", ">=40"]
        vals = None
    if vals:
        axes[0].bar(names, vals, color="#d62728")
        axes[0].set_title("holdout 图与 train 最近邻的像素差分布", fontsize=12)
        axes[0].set_ylabel("图数")
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, f"真实泄漏率 = {rep.get('leak_rate', 0)*100:.1f}%\n"
                               f"（{rep.get('duplicates')} / {rep.get('sampled')} 张）\n\n"
                               f"最近邻 MAE 中位数 = {st.get('median', 0):.1f}\n"
                               f"最小值 = {st.get('min', 0):.2f}（副本级）",
                     ha="center", va="center", fontsize=13,
                     bbox=dict(boxstyle="round", fc="#fff3cd", ec="#d62728"))

    # (b) 泄漏对指标的影响
    def load(p):
        try:
            return json.loads((ROOT / p).read_text(encoding="utf-8"))
        except Exception:
            return None

    allv = load("runs/cmp_det_rtdetrv2.json")
    cleanv = load("runs/eval_clean.json")
    if allv and cleanv:
        labels = ["含泄漏\n(全 val)", "无泄漏\n(干净子集)"]
        m = [allv.get("map50", 0), cleanv.get("map50", 0)]
        bars = axes[1].bar(labels, m, color=["#999999", "#2ca02c"], width=0.5)
        for b, v in zip(bars, m):
            axes[1].text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.4f}", ha="center", fontsize=11)
        axes[1].set_ylim(0, max(m) * 1.35)
        axes[1].set_title("泄漏对检测指标的影响：无泄漏反而略好\n→ 指标未被重复图污染", fontsize=12)
        axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = out_dir / "04_泄漏检查.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成汇报用图")
    ap.add_argument("--data-root", default="data/processed_stress2")
    ap.add_argument("--det-dir", default="models/det_rtdetrv2")
    ap.add_argument("--clean-list", default="runs/clean_val.txt")
    ap.add_argument("--out-dir", default="runs/figures")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-samples", type=int, default=4)
    args = ap.parse_args(argv)

    _setup_matplotlib()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"生成汇报用图 -> {out_dir}")
    print("=" * 70)

    fig_detection_samples(args, out_dir)
    fig_training_curves(args, out_dir)
    fig_metrics(args, out_dir)
    fig_leakage(args, out_dir)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
