"""统一预测入口：把各后端（ONNX / PyTorch / 规则基线）包成同一个对象。

对外只暴露 `Predictor.predict(bgr) -> dict`，字典字段与 `src/train/predictors.py`
的契约完全一致：
    {"disease_name","disease_name_cn","confidence","severity","class_id","bbox"}
检测任务额外带 "detections": [ ... 每个框一条 ... ]。

大图策略：短边超过 `slice_threshold` 时自动切片推理再合并（class-wise NMS），
避免 4000×4000 正射图缩放到 640 后小病斑彻底消失。
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.infer.postprocess import class_wise_nms, merge_tile_predictions, tile_origins
from src.labels import LABELS


def _info_from_model_table(name: str, table: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    """按"模型自带类别表"解释 disease_name，返回 {cn, is_disease, class_id}。

    为什么需要它：训练侧 predictor 只输出 `disease_name`（英文标识），中文名要接口层补；
    而不同数据集类别表不同（2 类胁迫模型 vs 15 类病害模型）。用全局表查会得到"未知"，
    所以优先用模型自己的类别表（来自 sidecar / checkpoint.json）。
    """
    if table and name in table:
        return dict(table[name])
    if name in LABELS.names:
        info = LABELS.by_name(name)
        return {
            "disease_name_cn": info["disease_name_cn"],
            "is_disease": bool(info.get("is_disease", True)),
            "class_id": int(info["class_id"]),
        }
    return {"disease_name_cn": name, "is_disease": name != "healthy", "class_id": -1}


def _normalize_backend_output(
    out: dict[str, Any],
    width: int,
    height: int,
    class_table: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """把后端输出补齐成契约字典，容忍后端只给最小字段的情况。

    中文名与 is_disease 的**优先来源是后端自己给的字段**；其次查"模型自带类别表"；
    最后才回退全局 labels.yaml。顺序不能反——本机实测：2 类胁迫模型的 "stressed"
    不在 15 类病害表里，用全局表会报"未知"。
    """
    name = str(out.get("disease_name", "healthy"))
    info = _info_from_model_table(name, class_table)
    cn = out.get("disease_name_cn") or info.get("disease_name_cn") or name
    is_disease = bool(info.get("is_disease", name != "healthy"))
    bbox = out.get("bbox") or [0.0, 0.0, float(width), float(height)]
    result = {
        "disease_name": name,
        "disease_name_cn": cn,
        "confidence": float(out.get("confidence", 0.0)),
        "severity": out.get("severity") or (
            "healthy" if not is_disease else LABELS.severity_from_ratio(0.01)
        ),
        "class_id": int(out.get("class_id", info.get("class_id", -1))),
        "bbox": [float(v) for v in bbox],
    }
    if "detections" in out:
        result["detections"] = out["detections"]
    if "topk" in out:
        result["topk"] = out["topk"]
    # 透传"规则基线"用来做整图口径统计的原始量（切片合并时会跨切片累加）
    for key in ("symptom_area", "image_area", "symptom_key", "severity_ratio"):
        if key in out:
            result[key] = out[key]
    return result


def _is_tile_placeholder(det: dict[str, Any], tile_min: int = 256) -> bool:
    """判断一个检测项是否是"无框后端"为整片区域返回的 healthy 占位框。

    这类框只表示"这片区域没查出病"，不是真实病斑：
    `_hot_region_box` 找不到热点时返回整片尺寸的框，合并回原图后就是一个
    几百像素见方的"健康块"，而且置信度高达 0.9。服务层画框前必须清掉它，
    否则前端会看到一堆"健康块"框，主病害也会被它挤掉。
    真实检测器（RT-DETR 等）不输出 healthy 类框，因此不受影响。
    """
    if str(det.get("disease_name")) != "healthy":
        return False
    bbox = det.get("bbox") or []
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (x2 - x1) >= tile_min and (y2 - y1) >= tile_min


class Predictor:
    """把后端 + 切片策略封装在一起。线程安全（后端推理本身是无状态的前向计算）。"""

    def __init__(
        self,
        backend: Any,
        *,
        slice_threshold: int = 1600,
        tile: int = 640,
        overlap: int = 128,
        max_det: int = 300,
        device: str = "cpu",
    ) -> None:
        self.backend = backend
        self.name = getattr(backend, "name", backend.__class__.__name__)
        self.model_version = getattr(backend, "model_version", "unknown")
        self.slice_threshold = int(slice_threshold)
        self.tile = int(tile)
        self.overlap = int(overlap)
        self.max_det = int(max_det)
        self.device = device
        # 分类后端没有框的语义，不走切片
        self.is_detector = getattr(backend, "task", "detect") != "classify"
        # 暴露后端的置信度阈值，便于排查"为什么返回的框这么少"
        self.conf_threshold = getattr(backend, "conf_threshold", None)
        # 模型自带的类别表：{disease_name: {disease_name_cn, is_disease, class_id}}。
        # 由后端提供（ONNX 从 sidecar、torch 从 checkpoint.json），
        # 用于把 disease_name 翻成中文并判断是否病害；缺失时回退全局 labels.yaml。
        self.class_table: dict[str, dict[str, Any]] = self._build_class_table(backend)

    @staticmethod
    def _build_class_table(backend: Any) -> dict[str, dict[str, Any]]:
        """从后端的 classes / label_book 构造 disease_name -> 类别信息 的查表。"""
        table: dict[str, dict[str, Any]] = {}

        def _add(name: str, cn: str | None, is_disease: bool | None, cid: Any) -> None:
            if not name:
                return
            if is_disease is None:
                is_disease = name != "healthy"
            table[name] = {
                "disease_name": name,
                "disease_name_cn": cn or name,
                "is_disease": bool(is_disease),
                "class_id": int(cid) if cid is not None else -1,
            }

        # 训练侧 predictor 提供的是 label_book（LabelBook 对象），优先用它
        book = getattr(backend, "label_book", None)
        if book is not None:
            try:
                for cid in range(int(getattr(book, "num_classes", 0))):
                    _add(book.name_of(cid), book.cn_of(cid), book.is_disease(cid), cid)
            except Exception:
                pass
        if table:
            return table

        # ONNX 后端提供的是解析好的 classes 列表（来自 sidecar）
        classes = getattr(backend, "classes", None)
        if isinstance(classes, list):
            for i, c in enumerate(classes):
                if not isinstance(c, dict):
                    continue
                _add(
                    str(c.get("disease_name", f"class_{i}")),
                    c.get("disease_name_cn"),
                    c.get("is_disease"),
                    c.get("class_id", i),
                )
        return table

    # ------------------------------------------------------------------
    def predict(self, bgr: np.ndarray) -> dict[str, Any]:
        h, w = bgr.shape[:2]
        if self.is_detector and max(h, w) > self.slice_threshold:
            result = self._predict_sliced(bgr)
        else:
            raw = self.backend.predict([bgr])[0]
            result = _normalize_backend_output(raw, w, h, self.class_table)
        return self._drop_healthy_placeholders(result, w, h)

    # ------------------------------------------------------------------
    @staticmethod
    def _drop_healthy_placeholders(result: dict[str, Any], width: int, height: int) -> dict[str, Any]:
        """主病害是病害时，清掉结果里的"健康占位框"。

        必须在这里清理，而不是留给服务层：服务层按置信度选主病害，
        而规则基线给"确认健康"的置信度是 0.9，会盖过真实病斑的低置信度，
        导致整块地被判成 healthy（本机实测踩过：2400×2400 图中 200×200 病斑）。
        真实检测器不会输出 healthy 类别框，所以这个清理对它无副作用。
        """
        dets = result.get("detections")
        if not dets:
            return result
        if not LABELS.by_name(result.get("disease_name", "healthy")).get("is_disease", False):
            return result  # 主病害就是健康：保留原样，服务层按健康处理
        cleaned = [d for d in dets if not _is_tile_placeholder(d)]
        if cleaned:
            result["detections"] = cleaned
        return result

    # ------------------------------------------------------------------
    def _predict_sliced(self, bgr: np.ndarray) -> dict[str, Any]:
        """大图切片推理：每片单独跑后端，再 class-wise NMS 合并到原图坐标。"""
        h, w = bgr.shape[:2]
        origins = tile_origins(w, h, tile=self.tile, overlap=self.overlap)
        boxes: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        severities: list[str] = []
        # 规则基线等"无框后端"返回的症状像素量：跨切片累加后按整图面积算比例，
        # 否则一个切片内的局部占比会被误当成整块地的发病率。
        symptom_area = 0
        image_area = 0
        symptom_keys: list[str] = []

        for x0, y0, x1, y1 in origins:
            patch = bgr[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            out = _normalize_backend_output(
                self.backend.predict([patch])[0], x1 - x0, y1 - y0, self.class_table
            )
            symptom_area += int(out.get("symptom_area", 0) or 0)
            image_area += int(out.get("image_area", 0) or 0)
            if out.get("symptom_key"):
                symptom_keys.append(str(out["symptom_key"]))
            dets = out.get("detections") or [out]
            for d in dets:
                bx = d.get("bbox") or [0, 0, x1 - x0, y1 - y0]
                boxes.append([bx[0] + x0, bx[1] + y0, bx[2] + x0, bx[3] + y0])
                scores.append(float(d.get("confidence", 0.0)))
                class_ids.append(int(d.get("class_id", 0)))
                severities.append(str(d.get("severity", "healthy")))

        if not boxes:
            info = LABELS.by_name("healthy")
            return {
                "disease_name": info["disease_name"],
                "disease_name_cn": info["disease_name_cn"],
                "confidence": 0.0,
                "severity": "healthy",
                "class_id": int(info["class_id"]),
                "bbox": [0.0, 0.0, float(w), float(h)],
                "detections": [],
            }

        merged = merge_tile_predictions(
            [{"boxes": boxes, "scores": scores, "class_ids": class_ids, "severities": severities}],
            iou_threshold=0.45,
            max_det=self.max_det,
        )
        # 严重度必须按**整图面积**重算，不能沿用切片内的局部严重度：
        # 一个病斑在一张 640 切片里可能占 10%（局部"重度"），
        # 但在 4000×4000 的整块地里只占 0.3%（整块地"轻度"）。
        # 早期版本直接继承切片结果，导致同一块地的 severity 随切片尺寸变化（真实的静默错误）。
        total_area = max(1.0, float(w) * float(h))
        detections: list[dict[str, Any]] = []
        for m in merged:
            info = LABELS.by_id(m["class_id"])
            x1, y1, x2, y2 = (float(v) for v in m["boxes"])
            area = max(0.0, (x2 - x1) * (y2 - y1))
            area_ratio = area / total_area
            severity = (
                "healthy" if not info.get("is_disease", False) else LABELS.severity_from_ratio(area_ratio)
            )
            detections.append(
                {
                    "disease_name": info["disease_name"],
                    "disease_name_cn": info["disease_name_cn"],
                    "confidence": round(m["score"], 4),
                    "severity": severity,
                    "class_id": m["class_id"],
                    "bbox": [x1, y1, x2, y2],
                    "area": area,
                }
            )

        # 关键过滤：没有症状的切片会返回"healthy + 该切片的全图占位框"，置信度还很高
        # （规则基线给"确认健康"的置信度是 0.9）。必须在选主病害前清掉，
        # 否则服务层按置信度选主病害时会把真实病斑挤掉，整块地判成 healthy。
        # 真实检测器不输出 healthy 类框，所以这里只对"会输出占位框的后端"生效。
        if getattr(self.backend, "emits_healthy_placeholder", False):
            detections = [d for d in detections if not _is_tile_placeholder(d)]
        disease_dets = [d for d in detections if LABELS.by_name(d["disease_name"]).get("is_disease", False)]
        pool = disease_dets or detections
        if not pool:
            # 整张图（含所有切片）都没查出病：返回健康，boxes 为空列表。
            # 注意：健康判定统一用 disease_name == "healthy"，前端不要用 boxes 是否为空来判断。
            info = LABELS.by_name("healthy")
            return {
                "disease_name": info["disease_name"],
                "disease_name_cn": info["disease_name_cn"],
                "confidence": 0.0,
                "severity": "healthy",
                "class_id": int(info["class_id"]),
                "bbox": [0.0, 0.0, float(w), float(h)],
                "detections": [],
                "severity_ratio": symptom_area / float(image_area) if image_area else 0.0,
                "tiles": len(origins),
            }
        # 主病害选取：按"置信度 × 框面积"加权，而不是只比置信度。
        # 原因：同一个病斑常被多个重叠切片各检到一部分，只比置信度时
        # "只框住病斑一角"的切片可能压过"框住完整病斑"的切片，
        # 于是换个切片尺寸主病害就变了（本机实测：1200×1200 图在 512 切片下
        # 由 chlorosis 翻成 necrosis）。用面积加权后结论稳定。
        best = max(pool, key=lambda d: d["confidence"] * max(d.get("area", 0.0), 1.0))
        result = dict(best)
        result["detections"] = detections
        result["tiles"] = len(origins)
        # 整图口径的症状占比（供服务层统一严重度与发病率，跨切片累加后计算）
        if symptom_area and image_area:
            result["severity_ratio"] = symptom_area / float(image_area)
        if symptom_keys:
            predominant = max(set(symptom_keys), key=symptom_keys.count)
            result["symptom_key"] = predominant
        return result

    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """跑一次小图推理，把 ONNX session / torch 的首次开销提前消化掉。"""
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            self.backend.predict([dummy])
        except Exception:
            # warmup 失败不阻塞启动，真实请求会再暴露问题
            pass


class HeuristicDetector:
    """规则基线的 Predictor 适配层（把 predict_one 的输出转成 detections 列表）。"""

    name = "heuristic-cv"
    model_version = "heuristic-v1"
    task = "detect"
    # 该后端在"没查出病"时会返回整片尺寸的 healthy 占位框，切片合并前必须清理
    emits_healthy_placeholder = True

    def __init__(self, grid: int = 4) -> None:
        from src.infer.heuristics import HeuristicBackend

        self._impl = HeuristicBackend(grid=grid)

    def predict(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        outs = []
        for img in images:
            h, w = img.shape[:2]
            one = self._impl.predict_one(img)
            one["detections"] = [dict(one)]
            outs.append(one)
        return outs


def load_image_bgr(path: str) -> np.ndarray:
    """读图（中文路径安全：cv2.imread 在 Windows 上遇中文路径会失败）。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图片：{path}")
    return img


def decode_image_bytes(data: bytes) -> np.ndarray:
    """把上传的二进制解码成 BGR 图。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("上传的文件不是可识别的图片格式（支持 jpg/png/bmp/webp）")
    return img
