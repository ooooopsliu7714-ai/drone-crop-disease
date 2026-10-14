#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""标注质量检查工具（COCO json → 可视化 + 分布统计 + 明显错误检测）。

【它解决什么】
无人机航拍数据集的标注质量直接决定模型上限。本项目最常见的三个坑：
  1) 框太小：原图 4096 缩到 640 后只剩几个像素，网络根本学不到——必须在标注阶段就发现；
  2) 越界框 / 宽高为 0 的框：训练时会被 resize 或除零搞崩，或者悄悄变成噪声标签；
  3) 重复框、单图类别冲突：同一位置标了两次或标成互斥类别，会让 AP 虚高、推理出双框。
本工具把这三类问题一次性扫出来，并画出每张图的框+类别，供人工复核。

【用法示例】
  python tools/vis_labels.py --help

  # 1) 全量检查：可视化 + 统计 + 错误清单
  python tools/vis_labels.py --coco data/annotations/instances_train.json \\
      --images data/images/train --out runs/label_check

  # 2) 只跑统计与错误检测（不画图，快；大数据集推荐）
  python tools/vis_labels.py --coco annotations.json --no-vis --out runs/label_check

  # 3) 只抽查前 50 张，并指定小目标阈值（按 640 缩放后的等效边长）
  python tools/vis_labels.py --coco annotations.json --images imgs --limit 50 \\
      --small-size 16 --out runs/label_check

【输出】
  <out>/vis/<file_name>_gt.jpg   每张图的标注可视化（框 + 类别中文名）
  <out>/report.json              机器可读报告（分布、错误清单、每图统计）
  <out>/report.md                中文简报，可直接贴到群里让标注同学返工
  <out>/issues.csv               错误明细，便于逐条勾销

【错误检测规则】
  out_of_bounds    框超出图像边界（宽高为 0 的框单独归类）
  zero_area        宽或高 <= 0，或面积 < 1 像素
  duplicate        同一图内与另一框 IoU > 阈值（默认 0.85）且类别相同
  class_conflict   同一图内与另一框 IoU > 阈值但类别不同（尤其 healthy vs 病害）
  tiny_box         等效边长小于 --small-size（按最短边缩放到 640 输入后的像素数）
  missing_image    标注里的 file_name 在 --images 下找不到（仅当给了 --images）
  empty_image      图里有标注却 0 个框（或反过来，图无框但被标了）

【依赖】numpy、Pillow（均已装）；cv2 可选（更快读图/缩放）。pyyaml 可选（读类别中文名）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
# 标签真源
# --------------------------------------------------------------------------
def load_cn_names(yaml_path: Optional[str] = None) -> Tuple[Dict[str, str], Dict[int, str], str]:
    """读标签体系 yaml，返回 (disease_name→中文, class_id→中文, 来源)。

    标签文件解析顺序（与 src/core/labels.py 保持一致，避免工具和训练脚本
    报出两套类别名）：
        1. 显式传入的 yaml_path
        2. 环境变量 ALGO_LABELS_FILE（换用别的数据集时用它切换类别体系）
        3. 默认 configs/labels.yaml
    """
    if yaml_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        env = os.environ.get("ALGO_LABELS_FILE", "").strip()
        if env:
            yaml_path = env if os.path.isabs(env) else os.path.join(root, env)
        else:
            yaml_path = os.path.join(root, "configs", "labels.yaml")
    if os.path.isfile(yaml_path):
        try:
            import yaml

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            by_name, by_id = {}, {}
            for c in data.get("classes") or []:
                by_name[str(c["disease_name"])] = str(c.get("disease_name_cn", c["disease_name"]))
                by_id[int(c["class_id"])] = str(c.get("disease_name_cn", c["disease_name"]))
            if by_name:
                return by_name, by_id, yaml_path
        except Exception as e:
            print(f"[警告] 读取 {yaml_path} 失败（{type(e).__name__}: {e}），类别中文名将用英文代替。",
                  file=sys.stderr)
    else:
        print(f"[警告] 未找到 {yaml_path}，类别中文名将用英文代替。", file=sys.stderr)
    return {}, {}, "builtin-none"


# --------------------------------------------------------------------------
# 几何
# --------------------------------------------------------------------------
def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """COCO 的 [x, y, w, h] 形式算 IoU。"""
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix = max(0.0, min(ax2, bx2) - max(ax, bx))
    iy = max(0.0, min(ay2, by2) - max(ay, by))
    inter = ix * iy
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 1e-9 else 0.0


def find_font(size: int = 16, user_font: Optional[str] = None) -> Tuple[ImageFont.ImageFont, bool]:
    """找中文字体，找不到就退回默认字体（英文标签）。"""
    cands = ([user_font] if user_font else []) + [
        r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\Deng.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for p in cands:
        if p and os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size), True
            except Exception:
                continue
    return ImageFont.load_default(), False


# --------------------------------------------------------------------------
# COCO 读取与自检
# --------------------------------------------------------------------------
def load_coco(path: str) -> dict:
    """读 COCO json，缺字段时给出中文提示而不是 KeyError。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k in ("images", "annotations", "categories"):
        if k not in data:
            raise KeyError(f"COCO 文件缺少必需字段 '{k}'：{path}。"
                           f"标准 COCO 需含 images / annotations / categories 三个字段。")
    return data


def check_coco(
    coco: dict,
    images_dir: Optional[str],
    iou_dup: float,
    small_size: float,
    input_size: int,
    by_id: Dict[int, str],
    by_name: Dict[str, str],
) -> Dict[str, Any]:
    """执行全部检查，返回报告字典。"""
    images = {int(im["id"]): im for im in coco["images"]}
    cats = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    anns_by_img: Dict[int, List[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[int(a["image_id"])].append(a)

    issues: List[dict] = []
    class_counter: Counter = Counter()
    # 框面积分布：用相对面积（占全图比例）更有可比性
    rel_areas: List[float] = []
    short_sides_640: List[float] = []
    per_image: List[dict] = []

    for img_id, im in images.items():
        W = float(im.get("width") or 0)
        H = float(im.get("height") or 0)
        fname = str(im.get("file_name", ""))
        img_anns = anns_by_img.get(img_id, [])

        # 文件存在性
        if images_dir and fname and not os.path.isfile(os.path.join(images_dir, fname)):
            issues.append({"type": "missing_image", "image_id": img_id, "file_name": fname,
                           "detail": f"在 {images_dir} 下找不到该图"})
        if not img_anns:
            issues.append({"type": "empty_image", "image_id": img_id, "file_name": fname,
                           "detail": "该图没有任何标注框（若是无病负样本可忽略）"})

        n_tiny = 0
        for a in img_anns:
            aid = a.get("id")
            cat = cats.get(int(a.get("category_id", -1)), f"unknown_{a.get('category_id')}")
            class_counter[cat] += 1
            bb = a.get("bbox") or [0, 0, 0, 0]
            x, y, w, h = (float(v) for v in bb[:4])

            # —— 规则：宽高为 0 / 面积过小 ——
            if w <= 0 or h <= 0 or w * h < 1.0:
                issues.append({"type": "zero_area", "image_id": img_id, "ann_id": aid,
                               "file_name": fname, "category": cat,
                               "detail": f"bbox={bb} 宽或高为 0（或面积<1px）"})
                continue

            # —— 规则：越界 ——
            if W > 0 and H > 0:
                if x < 0 or y < 0 or x + w > W + 0.5 or y + h > H + 0.5:
                    issues.append({"type": "out_of_bounds", "image_id": img_id, "ann_id": aid,
                                   "file_name": fname, "category": cat,
                                   "detail": f"bbox={[round(v,1) for v in (x,y,w,h)]} "
                                             f"超出图像 {int(W)}x{int(H)}"})

            # —— 分布：相对面积、缩放到 640 后的短边 ——
            if W > 0 and H > 0:
                rel = (w * h) / (W * H)
                rel_areas.append(rel)
                scale = float(input_size) / max(W, H)
                short_640 = min(w, h) * scale
                short_sides_640.append(short_640)
                # —— 规则：小目标 ——
                if short_640 < small_size:
                    n_tiny += 1
                    issues.append({"type": "tiny_box", "image_id": img_id, "ann_id": aid,
                                   "file_name": fname, "category": cat,
                                   "detail": f"缩放至 {input_size} 输入后短边仅 {short_640:.1f}px"
                                             f"（阈值 {small_size}px），网络可能学不到"})

        # —— 规则：重复框 / 类别冲突（同图内两两比较，n 小可接受）——
        for i in range(len(img_anns)):
            for j in range(i + 1, len(img_anns)):
                a, b = img_anns[i], img_anns[j]
                o = iou_xywh(a.get("bbox") or [0, 0, 0, 0], b.get("bbox") or [0, 0, 0, 0])
                if o <= iou_dup:
                    continue
                ca = cats.get(int(a.get("category_id", -1)), "?")
                cb = cats.get(int(b.get("category_id", -1)), "?")
                if ca == cb:
                    issues.append({"type": "duplicate", "image_id": img_id,
                                   "ann_id": a.get("id"), "ann_id_b": b.get("id"),
                                   "file_name": fname, "category": ca,
                                   "detail": f"两框 IoU={o:.3f}，疑似重复标注"})
                else:
                    issues.append({"type": "class_conflict", "image_id": img_id,
                                   "ann_id": a.get("id"), "ann_id_b": b.get("id"),
                                   "file_name": fname, "category": f"{ca} vs {cb}",
                                   "detail": f"两框 IoU={o:.3f} 但类别不同，互斥类别冲突"})

        per_image.append({
            "image_id": img_id, "file_name": fname,
            "width": int(W), "height": int(H),
            "num_boxes": len(img_anns), "num_tiny": n_tiny,
            "num_out_of_bounds": sum(1 for e in issues
                                     if e.get("image_id") == img_id and e["type"] == "out_of_bounds"),
        })

    issue_counts = Counter(e["type"] for e in issues)
    rel = np.array(rel_areas) if rel_areas else np.zeros(0)
    short = np.array(short_sides_640) if short_sides_640 else np.zeros(0)

    def _pct(arr: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
        if arr.size == 0:
            return {f"p{int(q)}": 0.0 for q in qs}
        return {f"p{int(q)}": round(float(np.percentile(arr, q)), 6) for q in qs}

    report = {
        "coco_path": os.path.abspath(coco.get("_path", "")) if coco.get("_path") else None,
        "summary": {
            "num_images": len(images),
            "num_annotations": len(coco["annotations"]),
            "num_categories": len(cats),
            "num_classes_used": len(class_counter),
            "empty_images": sum(1 for p in per_image if p["num_boxes"] == 0),
            "issue_total": len(issues),
            "issue_counts": dict(issue_counts),
            "issue_counts_cn": {_ISSUE_CN.get(k, k): v for k, v in issue_counts.items()},
        },
        "class_distribution": {
            cats.get(int(c["id"]), str(c["id"])): class_counter.get(cats.get(int(c["id"]), ""), 0)
            for c in coco["categories"]
        },
        "class_distribution_cn": {
            by_id.get(int(c["id"]), cats.get(int(c["id"]), str(c["id"]))):
                class_counter.get(cats.get(int(c["id"]), ""), 0)
            for c in coco["categories"]
        },
        "class_name_cn": {
            cats.get(int(c["id"]), str(c["id"])): by_id.get(int(c["id"]), cats.get(int(c["id"]), str(c["id"])))
            for c in coco["categories"]
        },
        "box_area_relative": {
            "note": "框面积 / 图像面积；无人机小病斑通常在 1e-5 ~ 1e-3 量级",
            "percentiles": _pct(rel, (1, 5, 25, 50, 75, 95, 99)),
            "min": round(float(rel.min()), 8) if rel.size else 0.0,
            "max": round(float(rel.max()), 8) if rel.size else 0.0,
        },
        "short_side_after_resize": {
            "note": f"短边缩放到 {input_size} 输入后的等效像素；< {small_size}px 视为过小",
            "input_size": input_size,
            "small_threshold_px": small_size,
            "percentiles": _pct(short, (1, 5, 25, 50, 75, 95, 99)),
            "num_too_small": int((short < small_size).sum()) if short.size else 0,
            "ratio_too_small": round(float((short < small_size).mean()), 4) if short.size else 0.0,
        },
        "issues": issues,
        "per_image": per_image,
    }
    return report


_ISSUE_CN = {
    "out_of_bounds": "越界框",
    "zero_area": "宽高为0/面积极小",
    "duplicate": "重复框",
    "class_conflict": "单图类别冲突",
    "tiny_box": "小目标（缩放后过小）",
    "missing_image": "标注对应的图片缺失",
    "empty_image": "图无任何标注框",
}


# --------------------------------------------------------------------------
# 可视化
# --------------------------------------------------------------------------
def draw_gt(image_path: str, img_info: dict, anns: List[dict], cats: Dict[int, str],
            out_path: str, by_id: Dict[int, str], font: ImageFont.ImageFont, cjk: bool) -> None:
    """把 GT 框画到图上。"""
    if cv2 is not None:
        data = np.fromfile(image_path, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img = Image.fromarray(bgr[:, :, ::-1]).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    palette = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 176, 32), (192, 64, 255),
               (32, 220, 220), (255, 96, 176), (160, 160, 32), (96, 96, 255), (0, 160, 96),
               (255, 128, 0), (128, 0, 200), (0, 96, 192), (200, 0, 96), (96, 128, 0)]
    for a in anns:
        cid = int(a.get("category_id", 0))
        color = palette[cid % len(palette)]
        x, y, w, h = (float(v) for v in (a.get("bbox") or [0, 0, 0, 0])[:4])
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        name = cats.get(cid, str(cid))
        txt = by_id.get(cid, name) if cjk else name
        try:
            tb = draw.textbbox((x, y), txt, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except Exception:
            tw, th = 8 * len(txt), 16
        ty = max(0.0, y - th - 6)
        draw.rectangle([x, ty, x + tw + 8, ty + th + 6], fill=color)
        draw.text((x + 4, ty + 3), txt, fill=(255, 255, 255), font=font)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path, quality=90)


# --------------------------------------------------------------------------
# Markdown 简报
# --------------------------------------------------------------------------
def write_markdown(report: dict, out_path: str, top_n: int = 20) -> None:
    """生成可以直接发群里的中文简报。"""
    s = report["summary"]
    lines: List[str] = []
    lines.append("# 标注质量检查简报")
    lines.append("")
    lines.append(f"- 标注文件：`{report['coco_path']}`")
    lines.append(f"- 图片数：**{s['num_images']}**，标注框数：**{s['num_annotations']}**，"
                 f"类别数：{s['num_categories']}（实际出现 {s['num_classes_used']} 类）")
    lines.append(f"- 无框图片：**{s['empty_images']}** 张")
    lines.append(f"- 检出问题总数：**{s['issue_total']}**")
    lines.append("")
    lines.append("## 一、问题分布")
    lines.append("")
    lines.append("| 问题类型 | 数量 | 说明 |")
    lines.append("| --- | ---: | --- |")
    for k, v in sorted(s["issue_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {_ISSUE_CN.get(k, k)} (`{k}`) | {v} | {_ISSUE_CN.get(k, '')} |")
    if not s["issue_counts"]:
        lines.append("| 无 | 0 | 未检出问题 |")
    lines.append("")
    lines.append("## 二、类别分布")
    lines.append("")
    lines.append("| 类别（中文） | 英文标识 | 框数 | 占比 |")
    lines.append("| --- | --- | ---: | ---: |")
    total = max(1, s["num_annotations"])
    cn_map = report.get("class_name_cn", {})
    for en, cnt in sorted(report["class_distribution"].items(), key=lambda kv: -kv[1]):
        cn = cn_map.get(en, en)
        lines.append(f"| {cn} | `{en}` | {cnt} | {cnt / total:.1%} |")
    lines.append("")
    lines.append("## 三、框尺寸分布（关键：小病斑能不能被学到）")
    lines.append("")
    ar = report["box_area_relative"]
    sh = report["short_side_after_resize"]
    lines.append(f"- 相对面积分位：{ar['percentiles']}（占全图比例）")
    lines.append(f"- 缩放到 {sh['input_size']} 输入后短边分位：{sh['percentiles']}（像素）")
    lines.append(f"- **短边 < {sh['small_threshold_px']}px 的框：{sh['num_too_small']} 个"
                 f"（{sh['ratio_too_small']:.1%}）**")
    if sh["ratio_too_small"] > 0.3:
        lines.append("")
        lines.append("> ⚠️ 超过 30% 的框在 640 输入下过小。建议：① 提高输入分辨率（如 1024/1280）；"
                     "② 采用切片训练/推理（见 `tools/tile_infer.py`）；③ 复核这些小框是否真的有判别信息。")
    lines.append("")
    lines.append(f"## 四、问题明细（前 {top_n} 条）")
    lines.append("")
    lines.append("| # | 类型 | 图片 | 类别 | 详情 |")
    lines.append("| ---: | --- | --- | --- | --- |")
    for i, e in enumerate(report["issues"][:top_n], 1):
        lines.append(f"| {i} | {_ISSUE_CN.get(e['type'], e['type'])} | `{e.get('file_name','')}` | "
                     f"{e.get('category','')} | {e.get('detail','')} |")
    if not report["issues"]:
        lines.append("| - | - | - | - | 无 |")
    lines.append("")
    lines.append("---")
    lines.append(f"完整明细见 `issues.csv`，机器可读报告见 `report.json`。")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="vis_labels.py",
        description="标注质量检查：COCO json → 可视化 + 类别/面积分布 + 明显错误检测。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python tools/vis_labels.py --coco annotations.json --images imgs --out runs/label_check\n"
               "  python tools/vis_labels.py --coco annotations.json --no-vis --limit 100\n",
    )
    p.add_argument("--coco", required=True, help="COCO 格式标注 json 路径")
    p.add_argument("--images", default=None, help="图片目录（用于可视化与存在性检查）；不给则跳过画图")
    p.add_argument("--out", default="runs/label_check", help="输出目录（默认 runs/label_check）")
    p.add_argument("--iou-dup", type=float, default=0.85, help="判定重复框/冲突的 IoU 阈值（默认 0.85）")
    p.add_argument("--small-size", type=float, default=16.0,
                   help="缩放后短边小于该值视为小目标（默认 16px）")
    p.add_argument("--input-size", type=int, default=640,
                   help="换算小目标时假定的网络输入边长（默认 640）")
    p.add_argument("--no-vis", action="store_true", help="不画可视化图，只出统计与错误清单（大数据集推荐）")
    p.add_argument("--limit", type=int, default=0, help="最多画多少张图的可视化，0=全部（默认 0）")
    p.add_argument("--font", default=None, help="中文字体路径，不给则自动搜索")
    p.add_argument("--md-top", type=int, default=20, help="Markdown 简报里列出的问题条数（默认 20）")
    p.add_argument("--quiet", action="store_true", help="安静模式")
    args = p.parse_args(argv)

    by_name, by_id, src = load_cn_names()
    coco = load_coco(args.coco)
    coco["_path"] = args.coco
    if not args.quiet:
        print(f"[信息] 标注文件：{args.coco}")
        print(f"[信息] 标签中文名真源：{src}（{len(by_id)} 类）")

    report = check_coco(
        coco,
        images_dir=args.images,
        iou_dup=args.iou_dup,
        small_size=args.small_size,
        input_size=args.input_size,
        by_id=by_id,
        by_name=by_name,
    )
    os.makedirs(args.out, exist_ok=True)

    # ---- 可视化 ----
    drawn = 0
    if not args.no_vis and args.images:
        font, cjk = find_font(16, args.font)
        if not cjk:
            print("[警告] 未找到中文字体，可视化标签降级为英文 disease_name。"
                  "可用 --font 指定 .ttf/.ttc。", file=sys.stderr)
        cats = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
        anns_by_img: Dict[int, List[dict]] = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[int(a["image_id"])].append(a)
        for im in coco["images"]:
            if args.limit and drawn >= args.limit:
                break
            fname = str(im.get("file_name", ""))
            src_path = os.path.join(args.images, fname)
            if not os.path.isfile(src_path):
                continue
            stem = os.path.splitext(os.path.basename(fname))[0]
            out_path = os.path.join(args.out, "vis", f"{stem}_gt.jpg")
            try:
                draw_gt(src_path, im, anns_by_img.get(int(im["id"]), []), cats,
                        out_path, by_id, font, cjk)
                drawn += 1
            except Exception as e:
                print(f"[警告] 画图失败 {fname}：{type(e).__name__}: {e}", file=sys.stderr)
        if not args.quiet:
            print(f"[信息] 已输出 {drawn} 张标注可视化图 → {os.path.join(args.out, 'vis')}")
    elif not args.no_vis and not args.images:
        print("[提示] 未指定 --images，跳过可视化（统计与错误检测不受影响）。", file=sys.stderr)

    # ---- 报告落盘 ----
    rep_json = os.path.join(args.out, "report.json")
    with open(rep_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    rep_md = os.path.join(args.out, "report.md")
    write_markdown(report, rep_md, top_n=args.md_top)
    iss_csv = os.path.join(args.out, "issues.csv")
    cols = ["type", "image_id", "ann_id", "ann_id_b", "file_name", "category", "detail"]
    with open(iss_csv, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for e in report["issues"]:
            wr.writerow({k: e.get(k, "") for k in cols})

    s = report["summary"]
    sh = report["short_side_after_resize"]
    print("=" * 68)
    print(f"[完成] 图片 {s['num_images']} 张 / 标注 {s['num_annotations']} 框 / 检出问题 {s['issue_total']} 个")
    for k, v in sorted(s["issue_counts"].items(), key=lambda kv: -kv[1]):
        print(f"       {_ISSUE_CN.get(k, k):<16} {v}")
    print(f"       小目标占比 {sh['ratio_too_small']:.1%}"
          f"（{sh['input_size']} 输入下短边 < {sh['small_threshold_px']}px）")
    print(f"       报告 → {rep_json}")
    print(f"       简报 → {rep_md}")
    print(f"       明细 → {iss_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
