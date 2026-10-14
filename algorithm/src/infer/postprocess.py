"""检测后处理：切片合并用的 class-wise NMS、坐标裁剪、伪框过滤。

为什么需要它：无人机正射大图（4000×4000 以上）直接缩到 640 会让小病斑只剩几个像素，
所以采用「滑窗切片 + 逐片推理 + class-wise NMS 合并」。同一个病斑在相邻切片的重叠区
会被检到多次，必须按类别分别做 NMS（不同类别的框不能互相抑制，否则漏检）。

本模块刻意只依赖 numpy，方便在评测脚本、离线批处理、服务端共用。
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU 矩阵（纯 numpy 向量化）。"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])       # (N,M,2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0).astype(np.float32)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    max_det: int = 300,
) -> list[int]:
    """单类别 NMS（贪心），返回保留的下标列表。"""
    if boxes.shape[0] == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return keep


def class_wise_nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    class_ids: Sequence[int],
    iou_threshold: float = 0.45,
    max_det: int = 300,
    score_threshold: float = 0.0,
) -> list[int]:
    """按类别分别 NMS，再按分数全局排序返回下标。"""
    bx = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    sc = np.asarray(scores, dtype=np.float32).reshape(-1)
    cl = np.asarray(class_ids, dtype=np.int32).reshape(-1)
    if bx.shape[0] == 0:
        return []

    keep: list[int] = []
    for c in np.unique(cl):
        idx = np.where(cl == c)[0]
        idx = idx[sc[idx] >= score_threshold]
        if idx.size == 0:
            continue
        local = nms(bx[idx], sc[idx], iou_threshold=iou_threshold, max_det=max_det)
        keep.extend(int(idx[k]) for k in local)
    keep.sort(key=lambda i: -float(sc[i]))
    return keep[:max_det]


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """把 xyxy 框裁剪进图像范围。"""
    out = boxes.astype(np.float32).copy()
    out[:, 0] = np.clip(out[:, 0], 0, width)
    out[:, 2] = np.clip(out[:, 2], 0, width)
    out[:, 1] = np.clip(out[:, 1], 0, height)
    out[:, 3] = np.clip(out[:, 3], 0, height)
    return out


def filter_degenerate(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, min_size: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """丢掉宽或高小于 min_size 的退化框（几像素的框在上报里只会造成误报）。"""
    if boxes.shape[0] == 0:
        return boxes, scores, class_ids
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w >= min_size) & (h >= min_size)
    return boxes[keep], scores[keep], class_ids[keep]


def merge_tile_predictions(
    per_tile: Iterable[dict[str, Any]],
    iou_threshold: float = 0.45,
    max_det: int = 300,
    class_wise: bool = True,
) -> list[dict[str, Any]]:
    """把各切片的结果合并回原图坐标并去重。

    入参 per_tile 每项：{"boxes": Nx4 xyxy(已加偏移、原图坐标),
                        "scores": [N], "class_ids": [N], "severities": [N](可选)}
    出参：按分数降序的字典列表，字段为 boxes/scores/class_ids/severities。
    """
    all_boxes: list[list[float]] = []
    all_scores: list[float] = []
    all_classes: list[int] = []
    all_sev: list[str] = []
    for t in per_tile:
        bx = np.asarray(t.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        sc = np.asarray(t.get("scores", []), dtype=np.float32).reshape(-1)
        cl = np.asarray(t.get("class_ids", []), dtype=np.int32).reshape(-1)
        sev = list(t.get("severities", [""] * bx.shape[0]))
        if bx.shape[0] == 0:
            continue
        n = min(bx.shape[0], sc.shape[0], cl.shape[0])
        all_boxes.extend(bx[:n].tolist())
        all_scores.extend(sc[:n].tolist())
        all_classes.extend(cl[:n].tolist())
        all_sev.extend(sev[:n])

    if not all_boxes:
        return []

    if class_wise:
        keep = class_wise_nms(all_boxes, all_scores, all_classes, iou_threshold, max_det)
    else:
        keep = nms(np.asarray(all_boxes, np.float32), np.asarray(all_scores, np.float32),
                   iou_threshold, max_det)
    return [
        {
            "boxes": all_boxes[i],
            "score": float(all_scores[i]),
            "class_id": int(all_classes[i]),
            "severity": all_sev[i] if i < len(all_sev) else "",
            # 框面积：跨切片比较"这个结果有多重要"时必须用它加权。
            # 只用置信度会让"病斑只占切片一角"的切片压过"病斑占满切片"的切片。
            "area": float(max(0.0, all_boxes[i][2] - all_boxes[i][0]))
            * float(max(0.0, all_boxes[i][3] - all_boxes[i][1])),
        }
        for i in keep
    ]


def tile_origins(
    width: int, height: int, tile: int = 640, overlap: int = 128
) -> list[tuple[int, int, int, int]]:
    """生成滑窗切片坐标 (x0, y0, x1, y1)，保证覆盖到右/下边界。"""
    if tile <= overlap:
        raise ValueError("tile 必须大于 overlap")
    step = tile - overlap
    xs = list(range(0, max(1, width - tile) + 1, step))
    ys = list(range(0, max(1, height - tile) + 1, step))
    if not xs or xs[-1] + tile < width:
        xs.append(max(0, width - tile))
    if not ys or ys[-1] + tile < height:
        ys.append(max(0, height - tile))
    return [(x, y, min(width, x + tile), min(height, y + tile)) for y in ys for x in xs]
