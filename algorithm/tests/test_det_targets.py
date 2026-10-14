"""检测训练目标编码的正确性测试 —— 拦住"目标和真值对不上"这类灾难性错误。

背景（本机真实事故）：为了让 transformers 5.x 接受标注，`collate_detection` 把
**归一化 cxcywh** 直接当成 **COCO 像素 xywh** 传给了 processor。processor 于是
把 0.0867 当像素、再按图宽归一化一次，真值 107×132 像素的框变成了 **0.2×0.2 像素**。
模型忠实地学会了"贴在图左上角、几乎零面积"的框，60 个 epoch、2.5 小时训完 mAP=0。
当时的测试只检查"训练能跑起来"，没有检查"喂进去的目标对不对"，所以没拦住。

本测试的做法：把编码后的目标交给真实的 processor，把 processor 解码出的框
**还原成像素**，与原始 COCO 真值逐框比对。偏差超过 2 像素就失败。

运行：python tests/test_det_targets.py
依赖：transformers（已装）。没有真实数据集时自动跳过。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 允许的像素级偏差（processor 的缩放/取整会引入亚像素误差）
TOL_PX = 2.0


def _find_val_coco() -> Path | None:
    """找一个带图片的 val 集（本项目两个数据集都可能存在）。"""
    for cand in (
        ROOT / "data" / "processed_stress2" / "val",
        ROOT / "data" / "processed" / "val",
    ):
        p = cand / "annotations.json"
        if p.is_file() and (cand / "images").is_dir():
            return p
    return None


def test_processor_targets_match_ground_truth_pixels() -> None:
    """核心测试：processor 解码出的目标还原成像素后必须等于真值。"""
    coco_path = _find_val_coco()
    if coco_path is None:
        print("[skip] 没有带图片的 val 集（先跑 src.data.prepare_data --copy-images）")
        return
    try:
        from PIL import Image
        from transformers import RTDetrImageProcessor
    except Exception as exc:  # pragma: no cover
        print(f"[skip] 缺少依赖：{exc}")
        return

    from src.train.detectors import collate_detection, encode_coco_to_rtdetr

    coco = json.loads(coco_path.read_text(encoding="utf-8-sig"))
    # 挑一张有标注的图
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)
    rec = next((im for im in coco["images"] if anns_by_img.get(int(im["id"]))), None)
    assert rec is not None, "val 集里没有带标注的图片"
    img_path = coco_path.parent / rec["file_name"]
    assert img_path.is_file(), f"图片不存在：{img_path}"

    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    anns = [dict(a) for a in anns_by_img[int(rec["id"])]]
    for a in anns:
        a["width"], a["height"] = W, H

    target = encode_coco_to_rtdetr(anns)
    out = collate_detection([{"image": pil, "target": target}], RTDetrImageProcessor())
    labels = out["labels"][0]
    boxes = labels["boxes"]
    assert len(boxes) == len(target["class_labels"]), (
        f"框数不一致：processor 解出 {len(boxes)}，编码时 {len(target['class_labels'])}"
    )

    # processor 输出是归一化 cxcywh（相对原图），还原成像素 xywh
    restored = []
    for cx, cy, w, h in boxes.tolist():
        restored.append([(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H])

    worst = 0.0
    worst_i = -1
    for i, (r, a) in enumerate(zip(restored, anns)):
        gx, gy, gw, gh = (float(v) for v in a["bbox"][:4])
        d = max(abs(r[0] - gx), abs(r[1] - gy), abs(r[2] - gw), abs(r[3] - gh))
        if d > worst:
            worst, worst_i = d, i
    assert worst <= TOL_PX, (
        "训练目标与真值对不上（这是会导致 mAP 恒为 0 的灾难性错误）：\n"
        f"  最大偏差 {worst:.2f} 像素，超出容差 {TOL_PX}\n"
        f"  第 {worst_i} 个框：解码后 {[round(v, 1) for v in restored[worst_i]]} "
        f"vs 真值 {[round(float(v), 1) for v in anns[worst_i]['bbox'][:4]]}\n"
        "  常见原因：把归一化 cxcywh 当成 COCO 像素 xywh 传给了 processor，"
        "或忘了从 boxes 换算出左上角基准。"
    )
    print(f"[PASS] 目标编码正确：{len(boxes)} 个框，最大偏差 {worst:.2f} 像素")


def test_degenerate_targets_are_filtered() -> None:
    """退化目标（<1 像素）应被剔除，而不是原样喂给模型。"""
    from src.train.detectors import encode_coco_to_rtdetr

    anns = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 0.5, 0.5], "width": 640, "height": 640},
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 100], "width": 640, "height": 640},
    ]
    t = encode_coco_to_rtdetr(anns)
    assert len(t["class_labels"]) == 1, f"<1 像素的框应被过滤，实际保留了 {len(t['class_labels'])} 个"
    assert t["class_labels"][0] == 1
    print("[PASS] 退化目标被正确过滤")


def test_clip_keeps_small_but_valid_boxes() -> None:
    """裁剪函数只能丢零面积框——不能把"小框"静默删掉。

    事故背景：早期版本用 `< 1.0` 像素过滤，模型输出退化框时会把**所有框删光**，
    评估显示"预测 0 个框"，把训练目标编码错误的病根掩盖成"模型没学到东西"。
    """
    from src.train.predictors import _clip_boxes_to_image

    boxes = [
        {"class_id": 0, "confidence": 0.5, "bbox": [-0.3, 52.4, 5.6, 107.1]},   # 负坐标、窄框
        {"class_id": 0, "confidence": 0.5, "bbox": [0.5, 0.5, 0.9, 0.9]},       # 0.4 像素小框
        {"class_id": 0, "confidence": 0.5, "bbox": [100, 100, 100, 200]},       # 零宽 -> 该丢
    ]
    out = _clip_boxes_to_image(boxes, 640, 640)
    assert len(out) == 2, f"应保留 2 个有效框（含那个 0.4 像素的），实际 {len(out)}"
    for b in out:
        x1, y1, x2, y2 = b["bbox"]
        assert x1 >= 0 and y1 >= 0 and x2 <= 640 and y2 <= 640, f"未裁剪到边界内：{b['bbox']}"
    print("[PASS] 裁剪保留小框、只丢零面积框")


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n共 {len(tests)} 项，失败 {failed} 项")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
