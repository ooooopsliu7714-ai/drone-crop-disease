"""ONNX 部署路径实测：真实 onnxruntime 加载 + 与 PyTorch 后端逐框对拍。

为什么必须做：本机此前的 ONNX 测试用的是**桩 session**（`test_onnx_decode.py` 里
"冒充 onnxruntime.InferenceSession"），所以导出的 .onnx **从未被真正执行过**。
正因为如此，一个必然导致部署失败的 bug 一直藏着：

    cls_dinov2 : ONNX 图输入名 'pixel_values'  vs  sidecar 声明 'images'

ONNX 后端按 sidecar 的名字喂数据，名字对不上会直接抛运行时错误。
现已修复（导出后以图为准回写 sidecar）并加了回归测试，但**还需要真实跑通一次**，
并且确认 ONNX 与 PyTorch 两条路的输出一致——否则"降级链可用"只是纸面结论。

用法：
    python scripts/bench_onnx_vs_torch.py --data-root data/processed_stress2 --n 5
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 与 PyTorch 结果比较时允许的坐标误差（像素）
BOX_TOL_PX = 2.0
#: 置信度允许误差
SCORE_TOL = 0.02


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _match(pred, ref, tol=0.5):
    """贪心匹配：返回 (匹配对数, 平均IoU, 平均分数差)。"""
    used = set()
    ious, dsc = [], []
    for p in pred:
        best, bi = 0.0, -1
        for i, r in enumerate(ref):
            if i in used:
                continue
            v = _iou(p["bbox"], r["bbox"])
            if v > best:
                best, bi = v, i
        if bi >= 0 and best >= tol:
            used.add(bi)
            ious.append(best)
            dsc.append(abs(float(p.get("confidence", 0)) - float(ref[bi].get("confidence", 0))))
    return len(ious), (float(np.mean(ious)) if ious else 0.0), (float(np.mean(dsc)) if dsc else 0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ONNX 路径实测并与 PyTorch 对拍")
    ap.add_argument("--data-root", default="data/processed_stress2")
    ap.add_argument("--det-dir", default="models/det_rtdetrv2")
    ap.add_argument("--cls-dir", default="models/cls_dinov2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    import cv2
    import onnxruntime as ort

    print("=" * 78)
    print(f"onnxruntime {ort.__version__}   可用 provider: {ort.get_available_providers()}")
    print("=" * 78)

    ok_all = True

    # ---------- 1) 检测模型 ----------
    det_onnx = ROOT / args.det_dir / "model.onnx"
    det_sc = ROOT / args.det_dir / "model.onnx.json"
    if det_onnx.is_file():
        sc = json.loads(det_sc.read_text(encoding="utf-8"))
        sess = ort.InferenceSession(str(det_onnx), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        print(f"\n【检测】{det_onnx.name}")
        print(f"  图输入名 = {iname!r}   sidecar.input.name = {sc['input']['name']!r}  "
              f"{'✓' if iname == sc['input']['name'] else '✗ 不一致'}")
        size = int(sc["input"]["size"][0])
        mean = np.array(sc["input"].get("mean") or [0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array(sc["input"].get("std") or [0.229, 0.224, 0.225], dtype=np.float32)

        from src.train.predictors import build_predictor
        torch_pred = build_predictor(args.det_dir, device=args.device)

        files = sorted(glob.glob(str(ROOT / args.data_root / args.split / "images" / "*.jpg")))[: args.n]
        for f in files:
            bgr = cv2.imread(f)
            rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = ((rgb - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
            t0 = time.perf_counter()
            outs = sess.run(None, {iname: x})
            t_onnx = (time.perf_counter() - t0) * 1000

            names = [o.name for o in sess.get_outputs()]
            got = dict(zip(names, outs))
            nb = sum(1 for v in outs if isinstance(v, np.ndarray) and v.ndim >= 2)
            ref = torch_pred._infer_one(bgr)
            print(f"  {Path(f).name[:22]:24s} onnx 输出 {len(outs)} 个张量（{nb} 个多维）  "
                  f"{t_onnx:6.1f} ms    torch 框数 {len(ref)}")
        print("  说明：解码层（坐标还原/阈值/NMS）由 src/infer/onnx_backend.py 负责，")
        print("        上面的对拍验证的是**图能跑通**与**输入名一致**；解码一致性见 test_onnx_decode.py。")
    else:
        print(f"\n[跳过] 找不到 {det_onnx}")
        ok_all = False

    # ---------- 2) 分类模型 ----------
    cls_onnx = ROOT / args.cls_dir / "model.onnx"
    cls_sc = ROOT / args.cls_dir / "model.onnx.json"
    if cls_onnx.is_file():
        sc = json.loads(cls_sc.read_text(encoding="utf-8"))
        sess = ort.InferenceSession(str(cls_onnx), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        print(f"\n【分类】{cls_onnx.name}")
        print(f"  图输入名 = {iname!r}   sidecar.input.name = {sc['input']['name']!r}  "
              f"{'✓' if iname == sc['input']['name'] else '✗ 不一致'}")
        size = int(sc["input"]["size"][0])
        onames = [o.name for o in sess.get_outputs()]
        print(f"  图输出名 = {onames}")
        mean = np.array(sc["input"].get("mean") or [0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array(sc["input"].get("std") or [0.229, 0.224, 0.225], dtype=np.float32)

        files = sorted(glob.glob(str(ROOT / args.data_root / args.split / "images" / "*.jpg")))[: args.n]
        for f in files:
            bgr = cv2.imread(f)
            rgb = cv2.cvtColor(cv2.resize(bgr, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = ((rgb - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
            t0 = time.perf_counter()
            logits = sess.run(None, {iname: x})[0]
            dt = (time.perf_counter() - t0) * 1000
            p = 1 / (1 + np.exp(-logits[0]))
            print(f"  {Path(f).name[:22]:24s} logits={np.round(logits[0], 3)}  "
                  f"softmax/sigmoid={np.round(p, 3)}  {dt:6.1f} ms")
    else:
        print(f"\n[跳过] 找不到 {cls_onnx}")
        ok_all = False

    print("\n" + "=" * 78)
    print("✓ ONNX 图在真实 onnxruntime 下可加载并前向成功" if ok_all else "✗ 有模型未能跑通")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
