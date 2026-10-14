#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大图滑窗切片推理工具（无人机正射影像 → 病斑检测框）。

【为什么需要它】
无人机正射大图常见 4096x4096 甚至 8000x8000 以上，而检测网络输入只有 640。
若把整张大图直接 resize 到 640，缩放比约 6.4 倍，一个原本 20 像素的小病斑会被压到
3 个像素——网络直接看不见，召回率崩塌。正确做法是「切片推理（tiling inference）」：
按 tile 尺寸带 overlap 滑窗，逐片推理，再把各片的框映射回原图坐标并用
class-wise NMS 合并重叠区域的重复框。

【设计原则：与 src/ 完全解耦】
本脚本 **不 import 任何 src/ 下的模块**，避免算法同学重构 src/ 时把工具带崩。
真实 predictor 通过两种方式接入，任选其一：
  (A) 插件模块：--predictor-module path/to/predictor.py
      该文件只需定义一个函数：
          def predict(image_bgr) -> list[dict]
      入参 image_bgr 是 numpy uint8 的 HWC/BGR 切片图；
      返回每个框的 dict，字段（切片内局部像素坐标 xyxy）：
          {"class_id": int, "confidence": float, "x1": float, "y1": float,
           "x2": float, "y2": float, "severity": str(可选)}
      可选的 disease_name / disease_name_cn 若不给，本脚本按 configs/labels.yaml 自动补全。
      推荐实现：from src.infer.your_detector import Detector; _d = Detector(); ...
  (B) 预计算结果：--predictions results.json（或 .csv）
      先用别的脚本把每片的框算好存盘，本脚本只负责切片→映射→NMS→导出。
      JSON 结构（由本脚本 --dump-tiles 生成的同名模板为准）：
          {"tiles": [{"tile_index": 0, "boxes": [ {..同上..} ]}, ...]}
      CSV 需含列：tile_index,class_id,confidence,x1,y1,x2,y2

【内置假 predictor（自测/演示用）】
  --dry-run 时不加载任何模型，改用「图像亮度阈值 + 连通域」编造假框：
  灰度 > 阈值 的连通块即为一个假病斑。这保证在 src/infer 就位之前，
  整个切片→映射→NMS→可视化→导出链路可以被完整测试。

  ⚠️ --brightness 默认 180，只适用于病斑明显发亮的图。
  **如果跑真实影像一片框都出不来，先看灰度分布再调这个阈值**：
      python -c "import cv2,numpy as np; g=cv2.cvtColor(cv2.imread(r'你的图.jpg'),cv2.COLOR_BGR2GRAY); print(g.min(),g.max(),g.mean(),np.percentile(g,99))"
  把 --brightness 设到 p99 附近或略低即可（例如实测某样例图 max=122、p99=117，
  阈值得降到 110 左右才出框）。出不来框不是脚本坏了——「无框 → 判 healthy」
  本身就是一条需要被验证的正常分支。

【用法示例】
  # 1) 看一眼参数
  python tools/tile_infer.py --help

  # 2) 最简自测：假 predictor + 切片 + NMS + 可视化叠加图
  python tools/tile_infer.py --image data/demo/big_field.jpg \\
      --dry-run --tile 640 --overlap 128 --conf 0.3 --save-vis --out runs/tile_demo

  # 3) 接真实 predictor（算法同学写好的）
  python tools/tile_infer.py --image data/demo/big_field.jpg \\
      --predictor-module src/infer/rtdetr_predictor.py --tile 640 --overlap 128 \\
      --conf 0.35 --save-vis --out runs/rtdetr_tiles

  # 4) 只做切片并导出空白模板，交给外部批量推理
  python tools/tile_infer.py --image data/demo/big_field.jpg --dump-tiles \\
      --tile 640 --overlap 128 --out runs/tiles

  # 5) 整目录批处理
  python tools/tile_infer.py --image data/demo/imgs --predictions runs/pred.json \\
      --out runs/batch --save-vis

【输出】
  <out>/predictions.json  契约形状的响应（字段与 configs/labels.yaml 完全一致）
  <out>/predictions.csv   便于 Excel/统计
  <out>/vis/<name>_vis.jpg 可视化叠加图（--save-vis）
  <out>/tiles.json        切片清单（--dump-tiles）

【依赖】numpy、Pillow、cv2（均已装）。pyyaml 可选，缺失时用内置标签兜底表。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - 依环境而定
    import cv2
except Exception as _e:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_ERR = _e
else:
    _CV2_ERR = None

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# 控制台中文输出：Windows 默认 GBK，重设为 UTF-8 避免中文乱码/报错
# --------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# --------------------------------------------------------------------------
# 标签体系：优先读 configs/labels.yaml（全项目唯一真源），失败则用内置兜底表
# --------------------------------------------------------------------------
_FALLBACK_SEVERITY = [
    {"key": "healthy", "key_cn": "健康", "severity_index": 0},
    {"key": "light", "key_cn": "轻度", "severity_index": 1},
    {"key": "moderate", "key_cn": "中度", "severity_index": 2},
    {"key": "severe", "key_cn": "重度", "severity_index": 3},
]

_FALLBACK_CLASSES = [
    {"class_id": 0, "disease_name": "healthy", "disease_name_cn": "健康", "is_disease": False},
    {"class_id": 1, "disease_name": "rice_blast", "disease_name_cn": "稻瘟病", "is_disease": True},
    {"class_id": 2, "disease_name": "rice_bacterial_blight", "disease_name_cn": "水稻白叶枯病", "is_disease": True},
    {"class_id": 3, "disease_name": "rice_sheath_blight", "disease_name_cn": "水稻纹枯病", "is_disease": True},
    {"class_id": 4, "disease_name": "wheat_powdery_mildew", "disease_name_cn": "小麦白粉病", "is_disease": True},
    {"class_id": 5, "disease_name": "wheat_stripe_rust", "disease_name_cn": "小麦条锈病", "is_disease": True},
    {"class_id": 6, "disease_name": "wheat_fusarium_head_blight", "disease_name_cn": "小麦赤霉病", "is_disease": True},
    {"class_id": 7, "disease_name": "corn_northern_leaf_blight", "disease_name_cn": "玉米大斑病", "is_disease": True},
    {"class_id": 8, "disease_name": "corn_southern_leaf_blight", "disease_name_cn": "玉米小斑病", "is_disease": True},
    {"class_id": 9, "disease_name": "corn_rust", "disease_name_cn": "玉米锈病", "is_disease": True},
    {"class_id": 10, "disease_name": "potato_late_blight", "disease_name_cn": "马铃薯晚疫病", "is_disease": True},
    {"class_id": 11, "disease_name": "tomato_early_blight", "disease_name_cn": "番茄早疫病", "is_disease": True},
    {"class_id": 12, "disease_name": "tomato_late_blight", "disease_name_cn": "番茄晚疫病", "is_disease": True},
    {"class_id": 13, "disease_name": "grape_black_rot", "disease_name_cn": "葡萄黑腐病", "is_disease": True},
    {"class_id": 14, "disease_name": "citrus_huanglongbing", "disease_name_cn": "柑橘黄龙病", "is_disease": True},
]


class Labels:
    """标签表容器：class_id ↔ disease_name ↔ 中文名 的互查。"""

    def __init__(self, classes: List[dict], severity_levels: List[dict], source: str) -> None:
        self.classes = classes
        self.severity_levels = severity_levels
        self.source = source
        self.by_id = {int(c["class_id"]): c for c in classes}
        self.by_name = {str(c["disease_name"]): c for c in classes}
        self.sev_index = {str(s["key"]): int(s["severity_index"]) for s in severity_levels}
        self.max_severity = max(self.sev_index.values()) if self.sev_index else 3

    def name_of(self, class_id: int) -> str:
        return str(self.by_id.get(int(class_id), {}).get("disease_name", f"class_{class_id}"))

    def name_cn_of(self, class_id: int) -> str:
        return str(self.by_id.get(int(class_id), {}).get("disease_name_cn", f"类别{class_id}"))


def load_labels(yaml_path: Optional[str] = None) -> Labels:
    """读取 configs/labels.yaml；pyyaml 缺失或文件不存在时退回内置标签表。"""
    if yaml_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(os.path.dirname(here), "configs", "labels.yaml")
    if os.path.isfile(yaml_path):
        try:
            import yaml  # 可选依赖

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            classes = data.get("classes") or _FALLBACK_CLASSES
            severity = data.get("severity_levels") or _FALLBACK_SEVERITY
            return Labels(classes, severity, source=yaml_path)
        except Exception as e:  # pyyaml 没装 / 文件格式异常
            print(f"[警告] 读取 {yaml_path} 失败（{type(e).__name__}: {e}），改用内置标签兜底表。"
                  f"请确认 pyyaml 已装：pip install pyyaml", file=sys.stderr)
    else:
        print(f"[警告] 未找到标签文件 {yaml_path}，改用内置标签兜底表。", file=sys.stderr)
    return Labels(_FALLBACK_CLASSES, _FALLBACK_SEVERITY, source="builtin-fallback")


# --------------------------------------------------------------------------
# 几何工具
# --------------------------------------------------------------------------
def iou_matrix_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """纯 numpy 计算两组 xyxy 框的 IoU 矩阵，形状 (len(a), len(b))。

    a、b 均为 (N,4) 的 float 数组，列顺序 x1,y1,x2,y2。
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = a.astype(np.float64).reshape(-1, 4)
    b = b.astype(np.float64).reshape(-1, 4)
    # 广播出 (N,M,2)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)


def nms_classwise(
    boxes: Sequence[dict],
    iou_thr: float = 0.5,
    score_key: str = "confidence",
    class_key: str = "class_id",
) -> List[dict]:
    """按类别分组做贪心 NMS，返回保留的框（dict 列表）。

    这里是「自己用 numpy 实现的 class-wise NMS」：不同类别的框即使高度重叠也互不抑制，
    因为同一位置同时存在两种病害在农业上是可能的，且类别混淆不应由 NMS 掩盖。
    另加一条跨类去重：同位置若一个是 healthy、另一个是病害，优先保留病害（病虫害防治以病害为准）。
    """
    if not boxes:
        return []
    out: List[dict] = []
    groups: Dict[int, List[int]] = {}
    for i, b in enumerate(boxes):
        groups.setdefault(int(b.get(class_key, 0)), []).append(i)

    for cid, idxs in groups.items():
        arr = np.array(
            [[float(boxes[i]["x1"]), float(boxes[i]["y1"]), float(boxes[i]["x2"]), float(boxes[i]["y2"])] for i in idxs],
            dtype=np.float64,
        )
        scores = np.array([float(boxes[i].get(score_key, 0.0)) for i in idxs], dtype=np.float64)
        order = np.argsort(-scores)
        while order.size > 0:
            cur = order[0]
            out.append(boxes[idxs[cur]])
            if order.size == 1:
                break
            ious = iou_matrix_np(arr[cur : cur + 1], arr[order[1:]])[0]
            order = order[1:][ious <= iou_thr]
    out.sort(key=lambda b: -float(b.get(score_key, 0.0)))
    return out


def iter_tiles(width: int, height: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int, int]]:
    """生成滑窗切片坐标，返回 [(tile_index, x0, y0, x1, y1), ...]。

    边缘处理：若最后一片越界，则把起点回退到 width-tile，保证切片尺寸恒为 tile×tile
    （避免边缘小片尺寸不一致导致的 batch 推理麻烦），因此最后一片与倒数第二片重叠更多。
    """
    if tile <= 0:
        raise ValueError("tile 必须为正整数")
    if overlap < 0 or overlap >= tile:
        raise ValueError("overlap 必须满足 0 <= overlap < tile")
    step = tile - overlap

    def _starts(total: int) -> List[int]:
        """一维起点序列：步长 step 前进，最后一片回退贴边，保证每片尺寸恒为 tile。"""
        if total <= tile:
            return [0]
        pos: List[int] = []
        x = 0
        while x + tile < total:
            pos.append(x)
            x += step
        pos.append(total - tile)  # 右/下边界收尾片，与上一片重叠更多
        return sorted(set(pos))

    xs = _starts(width)
    ys = _starts(height)
    tiles: List[Tuple[int, int, int, int, int]] = []
    k = 0
    for y0 in ys:
        for x0 in xs:
            tiles.append((k, x0, y0, min(x0 + tile, width), min(y0 + tile, height)))
            k += 1
    return tiles


# --------------------------------------------------------------------------
# 假 predictor（--dry-run）：亮度阈值 + 连通域，纯 cv2/numpy
# --------------------------------------------------------------------------
def _fallback_cc_boxes(mask: np.ndarray, min_area: int) -> List[Tuple[int, int, int, int]]:
    """cv2 不可用时的极简连通域：按行扫描的粗糙做法，仅用于自测。"""
    out: List[Tuple[int, int, int, int]] = []
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            if mask[y, x] and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                ys = [y]
                xs = [x]
                cnt = 0
                while stack and cnt < 20000:
                    cy, cx = stack.pop()
                    cnt += 1
                    ys.append(cy)
                    xs.append(cx)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(ys) >= min_area:
                    out.append((min(xs), min(ys), max(xs), max(ys)))
    return out


def make_dry_run_predictor(
    brightness_thr: int = 180,
    min_area: int = 40,
    max_boxes: int = 6,
    seed: int = 0,
) -> Callable[[np.ndarray], List[dict]]:
    """构造一个「假 predictor」：把亮于阈值的连通块当作伪造病斑框。

    用途：在 src/infer 的真实模型就位之前，端到端验证切片→映射→NMS→导出链路。
    返回的函数签名与真实 predictor 一致：predict(image_bgr) -> list[dict]。
    """
    rng = np.random.RandomState(seed)

    def predict(image_bgr: np.ndarray) -> List[dict]:
        if cv2 is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr[..., ::-1].mean(axis=2).astype(np.uint8)
        mask = (gray > brightness_thr).astype(np.uint8)
        boxes: List[Tuple[int, int, int, int]] = []
        if cv2 is not None:
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area >= min_area and w >= 4 and h >= 4:
                    boxes.append((int(x), int(y), int(x + w), int(y + h)))
        else:
            boxes = _fallback_cc_boxes(mask.astype(bool), min_area)
        boxes = boxes[:max_boxes]
        res: List[dict] = []
        for (x1, y1, x2, y2) in boxes:
            area = max(1.0, (x2 - x1) * (y2 - y1))
            # 编造一个稳定的假置信度：面积越大置信度略高，加一点随机扰动
            conf = float(np.clip(0.55 + 0.0004 * area + rng.uniform(-0.05, 0.05), 0.05, 0.99))
            cid = 1 + int(rng.randint(0, 3))  # 1..3 三类水稻病害，便于多类别 NMS 演示
            res.append(
                {
                    "class_id": cid,
                    "confidence": round(conf, 4),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )
        return res

    return predict


def load_predictor_module(path: str) -> Callable[[np.ndarray], List[dict]]:
    """从任意 .py 文件加载真实 predictor（插件式接入，不 import src/ 包）。

    约定该文件暴露 predict(image_bgr) -> list[dict]。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"predictor 模块不存在：{path}")
    spec = importlib.util.spec_from_file_location("_tile_infer_ext_predictor", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 predictor 模块：{path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "predict", None)
    if not callable(fn):
        raise AttributeError(
            f"{path} 中未找到可调用的 predict(image_bgr) 函数。"
            "真实 predictor 必须暴露该函数，详见本文件顶部 docstring。"
        )
    return fn


# --------------------------------------------------------------------------
# 预计算结果（--predictions）读取：JSON / CSV
# --------------------------------------------------------------------------
def load_predictions(path: str) -> Dict[int, List[dict]]:
    """读取每片的预推理结果，返回 {tile_index: [box, ...]}。"""
    ext = os.path.splitext(path)[1].lower()
    per_tile: Dict[int, List[dict]] = {}
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tiles = data.get("tiles") if isinstance(data, dict) else data
        if tiles is None:
            raise ValueError("predictions JSON 需为 {'tiles': [...]} 或直接是列表")
        for t in tiles:
            per_tile[int(t.get("tile_index", 0))] = list(t.get("boxes") or [])
    elif ext in (".csv", ".txt"):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                ti = int(float(row["tile_index"]))
                per_tile.setdefault(ti, []).append(
                    {
                        "class_id": int(float(row.get("class_id", 1))),
                        "confidence": float(row.get("confidence", row.get("score", 0.5))),
                        "x1": float(row["x1"]),
                        "y1": float(row["y1"]),
                        "x2": float(row["x2"]),
                        "y2": float(row["y2"]),
                        "severity": (row.get("severity") or "").strip() or None,
                    }
                )
    else:
        raise ValueError(f"不支持的预测文件格式：{ext}（仅支持 .json / .csv）")
    return per_tile


# --------------------------------------------------------------------------
# 严重度推断（占位启发式，真实系统应由分割面积/SAM 决定）
# --------------------------------------------------------------------------
def guess_severity(area_ratio: float, labels: Labels) -> str:
    """按「框面积占全图比例」粗略推断严重度，仅作占位。

    真实系统里 severity 应由第三级像素级分割得到的真实病斑面积占比决定
    （见 docs/algorithm-design.md 的三级流水线）。此处阈值故意保守，便于流水线先跑通。
    """
    if area_ratio <= 0.0:
        return "healthy"
    if area_ratio < 0.002:
        return "light"
    if area_ratio < 0.02:
        return "moderate"
    return "severe"


# --------------------------------------------------------------------------
# 可视化：PIL 画框 + 中文标签（字体缺失降级英文）
# --------------------------------------------------------------------------
_CJK_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    r"C:\Windows\Fonts\Deng.ttf",      # 等线
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def resolve_cjk_font(size: int = 18, user_font: Optional[str] = None) -> Tuple[ImageFont.ImageFont, bool]:
    """找一个能显示中文的字体。返回 (font, is_cjk_capable)。"""
    cands = ([user_font] if user_font else []) + _CJK_FONT_CANDIDATES
    for p in cands:
        if p and os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size), True
            except Exception:
                continue
    return ImageFont.load_default(), False


def draw_vis(
    image_bgr: np.ndarray,
    boxes: Sequence[dict],
    out_path: str,
    labels: Labels,
    font_path: Optional[str] = None,
) -> bool:
    """把检测框画到原图上并存盘。返回是否使用了中文字体。"""
    img = Image.fromarray(image_bgr[:, :, ::-1]).convert("RGB")
    draw = ImageDraw.Draw(img)
    font, cjk_ok = resolve_cjk_font(18, font_path)
    if not cjk_ok:
        print(
            "[警告] 未找到可显示中文的字体（尝试过 msyh.ttc/simhei.ttf 等），"
            "可视化标签降级为英文 disease_name。可用 --font 指定 .ttf/.ttc 字体文件。",
            file=sys.stderr,
        )

    # 每个类别一个固定颜色，便于肉眼区分类别
    palette = [
        (255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 176, 32),
        (192, 64, 255), (32, 220, 220), (255, 96, 176), (160, 160, 32),
        (96, 96, 255), (0, 160, 96), (255, 128, 0), (128, 0, 200),
        (0, 96, 192), (200, 0, 96), (96, 128, 0),
    ]
    for b in boxes:
        cid = int(b.get("class_id", 0))
        color = palette[cid % len(palette)]
        x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if cjk_ok:
            txt = f"{b.get('disease_name_cn', labels.name_cn_of(cid))} {float(b.get('confidence', 0)):.2f}"
        else:
            txt = f"{b.get('disease_name', labels.name_of(cid))} {float(b.get('confidence', 0)):.2f}"
        # 文本背景条，避免压在深色叶片上看不清
        try:
            tb = draw.textbbox((x1, y1), txt, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = 8 * len(txt), 18
        ty = max(0.0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 3), txt, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path, quality=92)
    return cjk_ok


# --------------------------------------------------------------------------
# 契约组装：与 configs/labels.yaml 末尾字段说明逐字对齐
# --------------------------------------------------------------------------
def build_contract_box(
    raw: dict,
    labels: Labels,
    image_w: int,
    image_h: int,
) -> dict:
    """把内部一个 xyxy 框整理成契约要求的 boxes[] 元素（含归一化与兼容字段）。"""
    x1 = max(0.0, min(float(raw["x1"]), float(image_w)))
    x2 = max(0.0, min(float(raw["x2"]), float(image_w)))
    y1 = max(0.0, min(float(raw["y1"]), float(image_h)))
    y2 = max(0.0, min(float(raw["y2"]), float(image_h)))
    cid = int(raw.get("class_id", 0))
    name = str(raw.get("disease_name") or labels.name_of(cid))
    name_cn = str(raw.get("disease_name_cn") or labels.name_cn_of(cid))
    conf = round(float(raw.get("confidence", 0.0)), 4)
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, float(image_w) * float(image_h))
    severity = str(raw.get("severity") or guess_severity(area_ratio, labels))
    if severity not in labels.sev_index:
        severity = "healthy"
    iw, ih = max(1.0, float(image_w)), max(1.0, float(image_h))
    return {
        # —— 契约主字段 ——
        "disease_name": name,
        "disease_name_cn": name_cn,
        "confidence": conf,
        "severity": severity,
        "class_id": cid,
        "x": round(x1 / iw, 6),
        "y": round(y1 / ih, 6),
        "w": round((x2 - x1) / iw, 6),
        "h": round((y2 - y1) / ih, 6),
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        # —— 兼容字段（部分前端用 label/score/bbox）——
        "label": name,
        "score": conf,
        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
    }


def compute_statistics(boxes: List[dict], labels: Labels, image_w: int, image_h: int) -> dict:
    """发病率 / 病情指数 / 严重度占比 / 类别占比（与 src/api/schema.py 算法一致）。

    病情指数 DI = Σ(病级 × 该病级面积) / (最高病级 × 总面积) × 100
    """
    total = max(1.0, float(image_w) * float(image_h))
    diseased = 0.0
    weighted = 0.0
    sev_area: Dict[str, float] = {}
    cls_area: Dict[str, float] = {}
    for b in boxes:
        area = max(0.0, (float(b["x2"]) - float(b["x1"])) * (float(b["y2"]) - float(b["y1"])))
        level = labels.sev_index.get(str(b.get("severity", "healthy")), 0)
        if level > 0:
            diseased += area
            weighted += level * area
        sev_area[str(b.get("severity"))] = sev_area.get(str(b.get("severity")), 0.0) + area
        cls_area[str(b.get("disease_name"))] = cls_area.get(str(b.get("disease_name")), 0.0) + area
    return {
        "total_boxes": len(boxes),
        "diseased_ratio": round(min(1.0, diseased / total), 6),
        "disease_index": round(100.0 * weighted / (labels.max_severity * total), 4) if labels.max_severity else 0.0,
        "severity_ratio": {k: round(v / total, 6) for k, v in sorted(sev_area.items())},
        "class_ratio": {k: round(v / total, 6) for k, v in sorted(cls_area.items())},
    }


# --------------------------------------------------------------------------
# 单图主流程
# --------------------------------------------------------------------------
def read_image_bgr(path: str) -> Tuple[np.ndarray, str]:
    """读图，返回 (BGR uint8 数组, 读取方式说明)。优先 cv2，退回 PIL。"""
    if cv2 is not None:
        data = np.fromfile(path, dtype=np.uint8)  # 用 fromfile 兼容中文路径
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img, "cv2"
    img_pil = Image.open(path).convert("RGB")
    arr = np.array(img_pil)[:, :, ::-1].copy()
    return arr, "PIL"


def infer_one_image(
    image_path: str,
    predictor: Callable[[np.ndarray], List[dict]],
    labels: Labels,
    tile: int,
    overlap: int,
    conf_thr: float,
    iou_thr: float,
    save_vis: bool,
    out_dir: str,
    font_path: Optional[str],
    max_boxes: int = 0,
    verbose: bool = True,
) -> dict:
    """对一张大图执行 切片→逐片推理→映射回原图→class-wise NMS→契约组装。"""
    t_start = time.perf_counter()
    reset = getattr(predictor, "reset", None)
    if callable(reset):
        reset()  # 预计算表模式：换图时片号归零
    img, how = read_image_bgr(image_path)
    h, w = img.shape[:2]
    tiles = iter_tiles(w, h, tile, overlap)
    if verbose:
        print(f"  [{os.path.basename(image_path)}] {w}x{h} 读图方式={how}，"
              f"切片 {tile}x{tile} overlap={overlap} → 共 {len(tiles)} 片")

    raw_global: List[dict] = []
    for (k, x0, y0, x1, y1) in tiles:
        patch = img[y0:y1, x0:x1]
        if patch.shape[0] != tile or patch.shape[1] != tile:
            # 大图尾部不足一片时补边到 tile×tile（右/下用边缘像素复制，避免黑边干扰）
            patch = np.pad(
                patch,
                ((0, max(0, tile - patch.shape[0])), (0, max(0, tile - patch.shape[1])), (0, 0)),
                mode="edge",
            )
        preds = predictor(patch) or []
        for p in preds:
            c = float(p.get("confidence", 0.0))
            if c < conf_thr:
                continue
            # 切片局部坐标 → 原图全局坐标；并裁剪到原图内（切片补边区域的框会被裁掉）
            gx1 = float(p["x1"]) + x0
            gy1 = float(p["y1"]) + y0
            gx2 = float(p["x2"]) + x0
            gy2 = float(p["y2"]) + y0
            gx1, gx2 = max(0.0, min(gx1, w)), max(0.0, min(gx2, w))
            gy1, gy2 = max(0.0, min(gy1, h)), max(0.0, min(gy2, h))
            if gx2 - gx1 < 2 or gy2 - gy1 < 2:
                continue
            raw_global.append(
                {
                    "class_id": int(p.get("class_id", 0)),
                    "confidence": c,
                    "x1": gx1,
                    "y1": gy1,
                    "x2": gx2,
                    "y2": gy2,
                    "severity": p.get("severity"),
                    "disease_name": p.get("disease_name"),
                    "disease_name_cn": p.get("disease_name_cn"),
                    "tile_index": k,
                }
            )

    merged = nms_classwise(raw_global, iou_thr=iou_thr)
    if max_boxes > 0:
        merged = merged[:max_boxes]
    boxes = [build_contract_box(b, labels, w, h) for b in merged]

    # 主病害：有框取置信度最高者；无框判健康（与 src/api/schema.py 的 top_disease 一致）
    if boxes:
        best = max(boxes, key=lambda b: b["confidence"])
        top_name, top_cn, top_conf, top_sev = (
            best["disease_name"], best["disease_name_cn"], best["confidence"], best["severity"],
        )
    else:
        top_name, top_cn, top_conf, top_sev = "healthy", labels.name_cn_of(0), 0.0, "healthy"

    elapsed = time.perf_counter() - t_start
    resp = {
        # —— 契约顶层字段（与 Apifox 逐字一致，禁止改名）——
        "code": 200,
        "message": "success",
        "image_id": os.path.splitext(os.path.basename(image_path))[0],
        "disease_name": top_name,
        "disease_name_cn": top_cn,
        "confidence": round(float(top_conf), 4),
        "severity": top_sev,
        "boxes": boxes,
        "processing_time": round(elapsed, 4),
        # —— 扩展字段（增量，不影响队友解析）——
        "statistics": compute_statistics(boxes, labels, w, h),
        "model_version": "tile_infer-tool",
        "backend": f"tiling({tile}/ov{overlap})",
        "image_width": w,
        "image_height": h,
        "tile_count": len(tiles),
        "raw_box_count": len(raw_global),
    }

    if save_vis:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(out_dir, "vis", f"{stem}_vis.jpg")
        try:
            draw_vis(img, boxes, vis_path, labels, font_path)
            resp["vis_path"] = vis_path
            if verbose:
                print(f"  可视化叠加图 → {vis_path}")
        except Exception as e:
            print(f"  [警告] 可视化失败（{type(e).__name__}: {e}），已跳过。", file=sys.stderr)

    if verbose:
        print(f"  原始框 {len(raw_global)} → NMS 后 {len(boxes)}，"
              f"主病害 {top_name}/{top_cn} conf={top_conf:.4f}，耗时 {elapsed:.3f}s")
    return resp


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tile_infer.py",
        description="无人机大图滑窗切片推理：切片→逐片推理→映射回原图→class-wise NMS→契约 JSON/CSV + 可视化。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python tools/tile_infer.py --image big.jpg --dry-run --save-vis --out runs/demo\n"
            "  python tools/tile_infer.py --image big.jpg --predictor-module src/infer/rtdetr_predictor.py\n"
            "  python tools/tile_infer.py --image big.jpg --dump-tiles --out runs/tiles\n"
        ),
    )
    p.add_argument("--image", required=True, help="输入大图路径，或包含多张图的目录")
    p.add_argument("--out", default="runs/tile_infer", help="输出目录（默认 runs/tile_infer）")
    p.add_argument("--tile", type=int, default=640, help="切片尺寸（默认 640，与检测网络输入一致）")
    p.add_argument("--overlap", type=int, default=128, help="相邻切片重叠像素（默认 128）")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值，低于此值的框丢弃（默认 0.25）")
    p.add_argument("--iou", type=float, default=0.5, help="class-wise NMS 的 IoU 阈值（默认 0.5）")
    p.add_argument("--max-boxes", type=int, default=0, help="单图最多保留框数，0=不限制（默认 0）")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true",
                   help="使用内置假 predictor（亮度阈值+连通域）自测，不加载任何模型")
    g.add_argument("--predictor-module", metavar="PATH",
                   help="真实 predictor 插件 .py 路径，需暴露 predict(image_bgr)->list[dict]")
    g.add_argument("--predictions", metavar="PATH",
                   help="预计算结果 .json/.csv，跳过推理只做映射与 NMS")
    g.add_argument("--dump-tiles", action="store_true",
                   help="只导出切片清单 JSON（不推理），交给外部批量推理")

    p.add_argument("--brightness", type=int, default=180, help="--dry-run 的亮度阈值（默认 180）")
    p.add_argument("--min-area", type=int, default=40, help="--dry-run 的最小连通域面积（默认 40）")
    p.add_argument("--save-vis", action="store_true", help="输出可视化叠加图（画框+类别标签）")
    p.add_argument("--font", default=None, help="中文字体 .ttf/.ttc 路径；不给我会自动搜索系统字体")
    p.add_argument("--format", default="both", choices=["json", "csv", "both"],
                   help="结果落盘格式（默认 both）")
    p.add_argument("--limit", type=int, default=0, help="最多处理几张图，0=全部（默认 0）")
    p.add_argument("--quiet", action="store_true", help="安静模式，只输出最终汇总")
    return p


def collect_images(path: str, limit: int = 0) -> List[str]:
    """把 --image 归一化成图片路径列表。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f)[1].lower() in exts
        )
        return files[:limit] if limit > 0 else files
    raise FileNotFoundError(f"输入路径不存在：{path}")


def write_csv(path: str, responses: List[dict]) -> None:
    """把每张图的框摊平成一张表，便于 Excel 统计。"""
    cols = ["image_id", "box_index", "disease_name", "disease_name_cn", "confidence",
            "severity", "class_id", "x", "y", "w", "h", "x1", "y1", "x2", "y2"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in responses:
            for i, b in enumerate(r.get("boxes", [])):
                row = {"image_id": r["image_id"], "box_index": i}
                row.update({k: b.get(k) for k in cols if k in b})
                wr.writerow({k: row.get(k, "") for k in cols})


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if cv2 is None and not args.dry_run:
        print(f"[警告] cv2 不可用（{_CV2_ERR}），将使用 PIL 读图，速度较慢。", file=sys.stderr)

    labels = load_labels()
    images = collect_images(args.image, args.limit)
    if not images:
        print(f"[错误] 在 {args.image} 下没有找到任何图片。", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    try:  # 提前校验切片参数合法性，给出中文友好提示而不是抛栈
        iter_tiles(64, 64, args.tile, args.overlap)
    except ValueError as e:
        print(f"[错误] 切片参数非法：{e}（例：--tile 640 --overlap 128 是合法组合）", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"[信息] 标签真源：{labels.source}，共 {len(labels.classes)} 类")
        print(f"[信息] 切片参数：tile={args.tile} overlap={args.overlap} "
              f"step={args.tile - args.overlap}，待处理 {len(images)} 张图")

    # ---- 模式 A：只导出切片清单 ----
    if args.dump_tiles:
        manifest = []
        for img_path in images:
            img, _ = read_image_bgr(img_path)
            h, w = img.shape[:2]
            tiles = iter_tiles(w, h, args.tile, args.overlap)
            manifest.append(
                {
                    "image_path": os.path.abspath(img_path),
                    "image_id": os.path.splitext(os.path.basename(img_path))[0],
                    "width": w, "height": h, "tile": args.tile, "overlap": args.overlap,
                    "tiles": [
                        {"tile_index": k, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "boxes": []}
                        for (k, x0, y0, x1, y1) in tiles
                    ],
                }
            )
            if not args.quiet:
                print(f"  [{os.path.basename(img_path)}] {w}x{h} → {len(tiles)} 片")
        out_json = os.path.join(args.out, "tiles.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"tiles_manifest": manifest}, f, ensure_ascii=False, indent=2)
        print(f"[完成] 切片清单 → {out_json}")
        print("       请让外部推理脚本按 tiles 填写 boxes 字段，再用 --predictions 回灌本工具。")
        return 0

    # ---- 组装 predictor ----
    if args.dry_run:
        predictor = make_dry_run_predictor(brightness_thr=args.brightness, min_area=args.min_area)
        if not args.quiet:
            print("[信息] 使用内置假 predictor（亮度阈值+连通域），仅用于链路自测。")
    elif args.predictor_module:
        predictor = load_predictor_module(args.predictor_module)
        if not args.quiet:
            print(f"[信息] 已加载真实 predictor：{args.predictor_module}")
    else:
        per_tile = load_predictions(args.predictions)
        if not args.quiet:
            print(f"[信息] 已加载预计算结果：{args.predictions}（{len(per_tile)} 片有结果）")
        # 预计算结果模式下，predictor 退化为「按当前片号查表」；每张图开始前必须归零
        class _TablePredictor:
            """按 tile_index 顺序返回预计算框的假 predictor（仅内部使用）。"""

            def __init__(self, table: Dict[int, List[dict]]) -> None:
                self.table = table
                self.k = -1

            def reset(self) -> None:
                self.k = -1

            def __call__(self, _patch: np.ndarray) -> List[dict]:
                self.k += 1
                return self.table.get(self.k, [])

        predictor = _TablePredictor(per_tile)

    # ---- 模式 B/C：逐图推理 ----
    responses: List[dict] = []
    for img_path in images:
        responses.append(
            infer_one_image(
                image_path=img_path,
                predictor=predictor,
                labels=labels,
                tile=args.tile,
                overlap=args.overlap,
                conf_thr=args.conf,
                iou_thr=args.iou,
                save_vis=args.save_vis,
                out_dir=args.out,
                font_path=args.font,
                max_boxes=args.max_boxes,
                verbose=not args.quiet,
            )
        )

    # ---- 落盘 ----
    written: List[str] = []
    if args.format in ("json", "both"):
        out_json = os.path.join(args.out, "predictions.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(responses[0] if len(responses) == 1 else {"results": responses},
                      f, ensure_ascii=False, indent=2)
        written.append(out_json)
    if args.format in ("csv", "both"):
        out_csv = os.path.join(args.out, "predictions.csv")
        write_csv(out_csv, responses)
        written.append(out_csv)

    total_boxes = sum(len(r["boxes"]) for r in responses)
    avg_t = float(np.mean([r["processing_time"] for r in responses])) if responses else 0.0
    print("=" * 68)
    print(f"[完成] 处理 {len(responses)} 张图，累计输出 {total_boxes} 个框，"
          f"单图平均耗时 {avg_t:.3f}s（本机 CPU 切片推理含 I/O，不含模型时延）")
    for w_ in written:
        print(f"       已写出：{w_}")
    if responses:
        r0 = responses[0]
        print(f"       首图主病害：{r0['disease_name']} / {r0['disease_name_cn']} "
              f"conf={r0['confidence']} severity={r0['severity']}")
        print(f"       首图统计：发病率={r0['statistics']['diseased_ratio']} "
              f"病情指数={r0['statistics']['disease_index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
