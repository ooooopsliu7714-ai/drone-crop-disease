#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""病害分布热力图生成工具（检测框 → 地块病害分布地图 + 网格严重度 CSV）。

【它解决什么】
项目要求「病害分布地图与统计报表」。检测框只能告诉你"哪里有框"，农户真正想看的是
"整块地里哪一片最严重、要不要马上打药"。本工具把 boxes 按「严重度 × 置信度」加权，
在每个框中心叠加一个高斯核（核宽随框尺寸自适应），累加成连续的风险场，再用
colormap 着色输出 PNG；同时按 N×N 网格统计每格的严重度数值导出 CSV，
供 Java 后端 / 小程序画统计报表，不用再跑一遍模型。

【输入格式（三选一，自动识别）】
1) 契约响应 JSON（POST /algo/predict 的返回体，字段与 configs/labels.yaml 一致）：
     {"code":200, "image_id":"...", "boxes":[{"x1":..,"y1":..,"x2":..,"y2":..,
       "severity":"moderate","confidence":0.87,"disease_name":"rice_blast",...}, ...],
      "image_width":4096, "image_height":4096}
   坐标优先用像素 x1/y1/x2/y2；若只有归一化 x/y/w/h，则需给 --width/--height 换回像素。
2) 批量结果：{"results": [ 上述响应, ... ]}，用 --image-id 选其中一张。
3) 纯框列表 JSON：[{...box...}, ...]，或本工具 / tile_infer.py 导出的 predictions.csv。

【用法示例】
  python tools/make_heatmap.py --help

  # 1) 用契约响应 + 原图，输出叠加热力图和 16x16 网格 CSV
  python tools/make_heatmap.py --input runs/predictions.json --image data/demo/field.jpg \\
      --grid 16 --out runs/heatmap --cmap turbo

  # 2) 没有原图也能出图（纯热力图，需指定画布尺寸）
  python tools/make_heatmap.py --input runs/predictions.json --width 4096 --height 4096 \\
      --grid 32 --out runs/heatmap

  # 3) 用 tile_infer.py 导出的 CSV
  python tools/make_heatmap.py --input runs/tile_infer/predictions.csv --width 4096 \\
      --height 4096 --out runs/heatmap --cmap jet --no-blend

【输出】
  <out>/<id>_heatmap.png   热力图（默认与原图 alpha 混合，--no-blend 则只出热力图）
  <out>/<id>_grid.csv      每个网格的严重度数值（行=grid_row，列含像素范围）
  <out>/<id>_heatmap.json  汇总统计（加权总量、峰值位置、各严重度框数、网格分位数）

【权重口径】
  单框权重 = severity_index(severity) × confidence(可选) × 面积归一化系数
  severity_index 来自 configs/labels.yaml（healthy=0, light=1, moderate=2, severe=3）。
  --weight severity 只用严重度（默认）；--weight conf 再乘置信度；--weight area 再乘框面积占比。

【依赖】numpy、Pillow（均已装）。cv2 可选，仅用于更快的读图。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# --------------------------------------------------------------------------
# 标签真源（读 configs/labels.yaml，失败退回内置表）
# --------------------------------------------------------------------------
_FALLBACK_SEVERITY = {"healthy": 0, "light": 1, "moderate": 2, "severe": 3}
_FALLBACK_SEV_CN = {"healthy": "健康", "light": "轻度", "moderate": "中度", "severe": "重度"}


def load_severity_index(yaml_path: Optional[str] = None) -> Tuple[Dict[str, int], Dict[str, str], str]:
    """返回 (severity→index, severity→中文, 来源说明)。"""
    if yaml_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(os.path.dirname(here), "configs", "labels.yaml")
    if os.path.isfile(yaml_path):
        try:
            import yaml

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            levels = data.get("severity_levels") or []
            idx = {str(s["key"]): int(s["severity_index"]) for s in levels}
            cn = {str(s["key"]): str(s.get("key_cn", s["key"])) for s in levels}
            if idx:
                return idx, cn, yaml_path
        except Exception as e:
            print(f"[警告] 读取 {yaml_path} 失败（{type(e).__name__}: {e}），改用内置严重度表。",
                  file=sys.stderr)
    else:
        print(f"[警告] 未找到 {yaml_path}，改用内置严重度表。", file=sys.stderr)
    return dict(_FALLBACK_SEVERITY), dict(_FALLBACK_SEV_CN), "builtin-fallback"


# --------------------------------------------------------------------------
# colormap：纯 numpy 实现，避免 matplotlib 依赖。锚点间线性插值。
# --------------------------------------------------------------------------
_CMAP_ANCHORS: Dict[str, List[Tuple[float, Tuple[int, int, int]]]] = {
    # 蓝 → 青 → 黄 → 红（经典 jet 观感，农业风险图习惯"越红越严重"）
    "jet": [
        (0.000, (0, 0, 131)), (0.125, (0, 60, 170)), (0.250, (5, 255, 255)),
        (0.375, (255, 255, 0)), (0.625, (250, 204, 0)), (0.875, (220, 0, 0)),
        (1.000, (128, 0, 0)),
    ],
    # Google Turbo 近似（颜色过渡比 jet 更均匀）
    "turbo": [
        (0.000, (48, 18, 59)), (0.143, (70, 107, 227)), (0.286, (40, 187, 235)),
        (0.429, (33, 227, 155)), (0.571, (146, 250, 65)), (0.714, (248, 194, 43)),
        (0.857, (238, 88, 27)), (1.000, (122, 4, 3)),
    ],
    # Matplotlib inferno 近似（暗→亮，适合打印/黑白友好）
    "inferno": [
        (0.000, (0, 0, 4)), (0.200, (66, 10, 104)), (0.400, (147, 38, 103)),
        (0.600, (221, 81, 58)), (0.800, (252, 165, 10)), (1.000, (252, 255, 164)),
    ],
    # 热金属：黑→红→黄→白
    "hot": [
        (0.000, (10, 0, 0)), (0.350, (180, 0, 0)), (0.700, (255, 200, 0)),
        (1.000, (255, 255, 255)),
    ],
    # Viridis 近似（色盲友好）
    "viridis": [
        (0.000, (68, 1, 84)), (0.250, (59, 82, 139)), (0.500, (33, 145, 140)),
        (0.750, (94, 201, 98)), (1.000, (253, 231, 37)),
    ],
}


def apply_colormap(heat01: np.ndarray, name: str = "turbo") -> np.ndarray:
    """把 0~1 的浮点热力场映射成 uint8 RGB 图，形状 (H, W, 3)。"""
    name = (name or "turbo").lower()
    anchors = _CMAP_ANCHORS.get(name)
    if anchors is None:
        print(f"[警告] 未知 colormap '{name}'，回退为 turbo。可选：{', '.join(_CMAP_ANCHORS)}",
              file=sys.stderr)
        anchors = _CMAP_ANCHORS["turbo"]
    ts = np.array([a[0] for a in anchors], dtype=np.float64)
    cols = np.array([a[1] for a in anchors], dtype=np.float64)
    t = np.clip(heat01.astype(np.float64), 0.0, 1.0).ravel()
    out = np.empty((t.size, 3), dtype=np.float64)
    for c in range(3):
        out[:, c] = np.interp(t, ts, cols[:, c])
    return out.reshape(heat01.shape + (3,)).astype(np.uint8)


# --------------------------------------------------------------------------
# 高斯核累加
# --------------------------------------------------------------------------
def gaussian_kernel_2d(sigma: float) -> np.ndarray:
    """生成归一化的二维高斯核（峰值 1.0，便于多框叠加后仍可解释）。"""
    sigma = max(1e-3, float(sigma))
    radius = max(1, int(round(3.0 * sigma)))  # 3σ 截断，误差可忽略
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k


def add_gaussian(heat: np.ndarray, cx: float, cy: float, sigma: float, weight: float) -> None:
    """把加权高斯核就地累加到热力场上（原地修改，避免大图反复分配内存）。"""
    if weight <= 0:
        return
    k = gaussian_kernel_2d(sigma) * float(weight)
    r = (k.shape[0] - 1) // 2
    h, w = heat.shape
    x0, x1 = int(round(cx)) - r, int(round(cx)) + r + 1
    y0, y1 = int(round(cy)) - r, int(round(cy)) + r + 1
    # 与画布求交，超出部分裁掉（边界框的核会被自然截断）
    kx0, ky0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return
    kx1 = kx0 + (x1c - x0c)
    ky1 = ky0 + (y1c - y0c)
    heat[y0c:y1c, x0c:x1c] += k[ky0:ky1, kx0:kx1]


# --------------------------------------------------------------------------
# 输入解析
# --------------------------------------------------------------------------
def _to_pixel_box(b: dict, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    """把一条 box 记录统一成像素 xyxy；解析不出来返回 None。"""
    if all(k in b and b[k] is not None for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
    elif all(k in b and b[k] is not None for k in ("x", "y", "w", "h")):
        x1 = float(b["x"]) * width
        y1 = float(b["y"]) * height
        x2 = x1 + float(b["w"]) * width
        y2 = y1 + float(b["h"]) * height
    elif b.get("bbox") and len(b["bbox"]) == 4:
        x1, y1, x2, y2 = (float(v) for v in b["bbox"])
    else:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def load_boxes(path: str, image_id: Optional[str], width_hint: int, height_hint: int) -> Tuple[List[dict], Dict[str, Any]]:
    """读取框记录。返回 (boxes, meta)，meta 含画布尺寸与来源说明。"""
    ext = os.path.splitext(path)[1].lower()
    meta: Dict[str, Any] = {"source": path}
    boxes: List[dict] = []

    if ext in (".csv", ".txt"):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                boxes.append(dict(row))
        meta["image_id"] = image_id or "grid"
        return boxes, meta

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    top: Dict[str, Any] = {}
    if isinstance(data, dict) and "results" in data:
        cands = data["results"]
        if image_id:
            top = next((r for r in cands if str(r.get("image_id")) == str(image_id)), cands[0])
        else:
            top = cands[0]
            if len(cands) > 1:
                print(f"[信息] 输入含 {len(cands)} 张图结果，未指定 --image-id，"
                      f"默认取第一张 '{top.get('image_id')}'。", file=sys.stderr)
        boxes = list(top.get("boxes") or [])
    elif isinstance(data, dict) and "boxes" in data:
        top = data
        boxes = list(data.get("boxes") or [])
    elif isinstance(data, list):
        boxes = list(data)
        top = {}
    elif isinstance(data, dict) and "tiles" in data:
        # 兼容 tile_infer.py --dump-tiles 的清单（裁掉每片的框，坐标已是局部，不直接可用）
        raise ValueError(
            "输入看起来是切片清单（tiles），切片内坐标不能直接画地块热力图。"
            "请改传 tile_infer.py 输出的 predictions.json。"
        )
    else:
        raise ValueError(f"无法识别的输入结构：{path}（期望契约响应 / {{'results':[...]}} / box 列表 / CSV）")

    meta["image_id"] = top.get("image_id") or image_id or "heatmap"
    if width_hint <= 0 and top.get("image_width"):
        width_hint = int(top["image_width"])
    if height_hint <= 0 and top.get("image_height"):
        height_hint = int(top["image_height"])
    meta["declared_width"] = width_hint
    meta["declared_height"] = height_hint
    return boxes, meta


# --------------------------------------------------------------------------
# 网格统计
# --------------------------------------------------------------------------
def grid_statistics(
    heat: np.ndarray,
    grid: int,
    sev_index: Dict[str, int],
    sev_cn: Dict[str, str],
    boxes: List[dict],
    width: int,
    height: int,
) -> List[dict]:
    """把热力场切成 grid×grid，统计每格的严重度数值，返回行列表。"""
    h, w = heat.shape
    rows: List[dict] = []
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    peak = float(heat.max()) if heat.size else 0.0
    for r in range(grid):
        for c in range(grid):
            y0, y1 = ys[r], ys[r + 1]
            x0, x1 = xs[c], xs[c + 1]
            cell = heat[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            s_sum = float(cell.sum())
            # 按格内严重度面积占比给出主严重度等级
            sev_w: Dict[str, float] = {}
            for b in boxes:
                pb = _to_pixel_box(b, width, height)
                if pb is None:
                    continue
                bx1, by1, bx2, by2 = pb
                # 框与格的交叠面积按比例摊到该格，用于判定该格的主导严重度
                ox = max(0.0, min(bx2, x1) - max(bx1, x0))
                oy = max(0.0, min(by2, y1) - max(by1, y0))
                if ox > 0 and oy > 0:
                    key = str(b.get("severity") or "healthy")
                    sev_w[key] = sev_w.get(key, 0.0) + ox * oy
            dominant = max(sev_w.items(), key=lambda kv: kv[1])[0] if sev_w else "healthy"
            rows.append(
                {
                    "grid_row": r,
                    "grid_col": c,
                    "x1": int(x0), "y1": int(y0), "x2": int(x1), "y2": int(y1),
                    "severity_sum": round(s_sum, 6),
                    "severity_mean": round(s_sum / float(cell.size), 6),
                    "heat_max": round(float(cell.max()), 6),
                    "heat_norm": round(float(cell.max()) / peak, 6) if peak > 0 else 0.0,
                    "dominant_severity": dominant,
                    "dominant_severity_cn": sev_cn.get(dominant, dominant),
                    "dominant_level": sev_index.get(dominant, 0),
                }
            )
    return rows


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def build_heatmap(
    boxes: List[dict],
    width: int,
    height: int,
    sev_index: Dict[str, int],
    weight_mode: str,
    sigma_scale: float,
    sigma_min: float,
    sigma_max: float,
    use_conf: bool,
) -> np.ndarray:
    """把所有框加权累加成热力场（未归一化）。"""
    heat = np.zeros((height, width), dtype=np.float64)
    total = float(max(1, width * height))
    for b in boxes:
        pb = _to_pixel_box(b, width, height)
        if pb is None:
            continue
        x1, y1, x2, y2 = pb
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        sev = str(b.get("severity") or "healthy")
        level = sev_index.get(sev, 0)
        if level <= 0:  # healthy 不产生风险
            continue
        w_ = float(level)
        if use_conf:
            w_ *= float(b.get("confidence", b.get("score", 1.0)) or 1.0)
        if weight_mode == "area":
            w_ *= (bw * bh) / total * 100.0  # 面积占比放大 100 倍，量纲更可读
        # 核宽随病斑尺寸自适应：小病斑给窄核（定位锐利），大病斑给宽核（影响范围大）
        sigma = float(np.clip(sigma_scale * np.sqrt(bw * bh), sigma_min, sigma_max))
        add_gaussian(heat, cx, cy, sigma, w_)
    return heat


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="make_heatmap.py",
        description="把检测框按严重度加权生成病害分布热力图（PNG）+ 每网格严重度 CSV。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python tools/make_heatmap.py --input runs/predictions.json --image field.jpg --grid 16\n"
            "  python tools/make_heatmap.py --input runs/predictions.csv --width 4096 --height 4096\n"
        ),
    )
    p.add_argument("--input", required=True, help="输入框数据：契约响应 JSON / {'results':[...]} / box 列表 / CSV")
    p.add_argument("--out", default="runs/heatmap", help="输出目录（默认 runs/heatmap）")
    p.add_argument("--image", default=None, help="可选：底图原图，用于叠加显示；不给则输出纯热力图")
    p.add_argument("--width", type=int, default=0, help="画布宽（像素）；未给且 JSON 无 image_width 时必填")
    p.add_argument("--height", type=int, default=0, help="画布高（像素）；未给且 JSON 无 image_height 时必填")
    p.add_argument("--image-id", default=None, help="输入含多张图结果时，选哪一张（默认第一张）")
    p.add_argument("--grid", type=int, default=16, help="网格划分数 N（N×N），默认 16")
    p.add_argument("--cmap", default="turbo", choices=sorted(_CMAP_ANCHORS.keys()),
                   help="配色方案（默认 turbo）")
    p.add_argument("--sigma-scale", type=float, default=0.6,
                   help="高斯核宽系数：sigma = scale × sqrt(框面积)，默认 0.6")
    p.add_argument("--sigma-min", type=float, default=6.0, help="核宽下限（像素），默认 6")
    p.add_argument("--sigma-max", type=float, default=200.0, help="核宽上限（像素），默认 200")
    p.add_argument("--weight", default="severity", choices=["severity", "conf", "area"],
                   help="权重口径：severity(默认) / conf(再乘置信度) / area(再乘面积占比)")
    p.add_argument("--no-conf", action="store_true", help="不乘置信度（默认会在 conf/area 模式下乘）")
    p.add_argument("--alpha", type=float, default=0.55, help="与原图混合的透明度 0~1，默认 0.55")
    p.add_argument("--no-blend", action="store_true", help="不与原图混合，只输出纯热力图")
    p.add_argument("--percentile", type=float, default=99.0,
                   help="归一化上限分位数，默认 99（避免个别孤立峰值把整图压暗）")
    p.add_argument("--quiet", action="store_true", help="安静模式")
    args = p.parse_args(argv)

    sev_index, sev_cn, sev_src = load_severity_index()
    if not args.quiet:
        print(f"[信息] 严重度真源：{sev_src} → {sev_index}")

    boxes, meta = load_boxes(args.input, args.image_id, args.width, args.height)
    if not boxes:
        print(f"[错误] 输入中没有可用的 boxes：{args.input}", file=sys.stderr)
        return 2

    # ---- 确定画布尺寸 ----
    width, height = args.width, args.height
    if args.image and os.path.isfile(args.image):
        if cv2 is not None:
            data = np.fromfile(args.image, dtype=np.uint8)
            base = cv2.imdecode(data, cv2.IMREAD_COLOR)
            base = None if base is None else base[:, :, ::-1]
        else:
            base = np.array(Image.open(args.image).convert("RGB"))
        ih, iw = base.shape[:2]
        width = width or iw
        height = height or ih
        if (width, height) != (iw, ih):
            print(f"[警告] --width/--height ({width}x{height}) 与底图 ({iw}x{ih}) 不一致，"
                  f"按底图尺寸缩放热力图。", file=sys.stderr)
            width, height = iw, ih
    else:
        base = None
        width = width or meta.get("declared_width") or 0
        height = height or meta.get("declared_height") or 0

    if not width or not height:
        print("[错误] 无法确定画布尺寸：请给 --image，或 --width/--height，"
              "或确保输入 JSON 含 image_width/image_height。", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"[信息] 画布 {width}x{height}，输入 {len(boxes)} 个框，"
              f"权重口径={args.weight}{'' if args.no_conf else '+置信度'}，网格 {args.grid}x{args.grid}")

    # ---- 累加 + 归一化 ----
    heat = build_heatmap(
        boxes, width, height, sev_index,
        weight_mode=("severity" if args.weight == "severity" else args.weight),
        sigma_scale=args.sigma_scale, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        use_conf=(not args.no_conf) and args.weight in ("conf", "area"),
    )
    peak = float(heat.max())
    # 用分位数做归一化上限，避免单个孤立峰值把整片区域压成深蓝
    if peak > 0:
        cap = float(np.percentile(heat, args.percentile))
        cap = cap if cap > 0 else peak
        heat01 = np.clip(heat / cap, 0.0, 1.0)
    else:
        cap = 0.0
        heat01 = np.zeros_like(heat)

    rgb = apply_colormap(heat01, args.cmap)

    # ---- 叠加底图 ----
    if base is not None and not args.no_blend:
        if base.shape[:2] != (height, width):
            base = np.array(Image.fromarray(base).resize((width, height), Image.BILINEAR))
        a = float(np.clip(args.alpha, 0.0, 1.0))
        # 热力越强（红）越不透明，保持原图纹理可见
        wmap = (0.35 + 0.65 * heat01)[..., None]
        blended = (base.astype(np.float64) * (1 - a * wmap) + rgb.astype(np.float64) * (a * wmap))
        out_rgb = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        out_rgb = rgb

    os.makedirs(args.out, exist_ok=True)
    stem = str(meta.get("image_id") or "heatmap").replace(os.sep, "_")
    png_path = os.path.join(args.out, f"{stem}_heatmap.png")
    Image.fromarray(out_rgb).save(png_path)

    # ---- 网格 CSV ----
    rows = grid_statistics(heat, args.grid, sev_index, sev_cn, boxes, width, height)
    csv_path = os.path.join(args.out, f"{stem}_grid.csv")
    cols = ["grid_row", "grid_col", "x1", "y1", "x2", "y2", "severity_sum", "severity_mean",
            "heat_max", "heat_norm", "dominant_severity", "dominant_severity_cn", "dominant_level"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in cols})

    # ---- 汇总 JSON ----
    idx_max = int(np.argmax(heat)) if heat.size else 0
    py, px = divmod(idx_max, width) if width else (0, 0)
    sev_count: Dict[str, int] = {}
    for b in boxes:
        k = str(b.get("severity") or "healthy")
        sev_count[k] = sev_count.get(k, 0) + 1
    summary = {
        "source": os.path.abspath(args.input),
        "image_id": meta.get("image_id"),
        "canvas": {"width": width, "height": height},
        "grid": args.grid,
        "cmap": args.cmap,
        "weight_mode": args.weight,
        "use_confidence": bool((not args.no_conf) and args.weight in ("conf", "area")),
        "grade_note": "分级术语 severity/healthy/light/moderate/severe 与 configs/labels.yaml 一致",
        "box_count": len(boxes),
        "severity_counts": sev_count,
        "severity_counts_cn": {sev_cn.get(k, k): v for k, v in sev_count.items()},
        "heat_total": round(float(heat.sum()), 4),
        "heat_peak": round(peak, 6),
        "heat_cap_p{}".format(args.percentile): round(cap, 6),
        "heat_percentiles": {
            "p50": round(float(np.percentile(heat, 50)), 6),
            "p90": round(float(np.percentile(heat, 90)), 6),
            "p99": round(float(np.percentile(heat, 99)), 6),
        },
        "peak_location": {"x": int(px), "y": int(py)},
        "grid_top5": sorted(rows, key=lambda r: -r["severity_sum"])[:5],
        "outputs": {"png": png_path, "grid_csv": csv_path},
    }
    json_path = os.path.join(args.out, f"{stem}_heatmap.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 68)
    print(f"[完成] 热力图 → {png_path}")
    print(f"       网格 CSV → {csv_path}（{len(rows)} 格）")
    print(f"       汇总 JSON → {json_path}")
    print(f"       峰值位置 = ({px}, {py})，heat_peak={peak:.4f}，"
          f"参与加权框数 = {sum(1 for b in boxes if sev_index.get(str(b.get('severity') or 'healthy'), 0) > 0)}")
    top = sorted(rows, key=lambda r: -r["severity_sum"])[:3]
    for r in top:
        print(f"       最严重网格 行{r['grid_row']}列{r['grid_col']} "
              f"({r['x1']},{r['y1']})-({r['x2']},{r['y2']}) "
              f"severity_sum={r['severity_sum']} 主导={r['dominant_severity_cn']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
