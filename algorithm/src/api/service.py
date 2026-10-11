"""业务编排层：把 Predictor 的原始输出组装成 Apifox 契约响应。

这里不关心模型怎么跑，只关心：
  1. 坐标归一化、字段名、类型都符合契约（src/api/schema.py）；
  2. 主病害选取规则可解释（取置信度最高的框，低于阈值时标记复核）；
  3. 统计报表、防治建议作为增量字段一并返回。
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.api.advice import build_advice
from src.api.schema import (
    BBoxOut,
    PredictResponse,
    bbox_to_out,
    compute_statistics,
    top_disease,
)
from src.infer.predictor import Predictor, decode_image_bytes, load_image_bgr
from src.labels import LABELS

# 单图推理耗时预算（写进答辩文档）：超过它就要考虑降分辨率或换更小模型
LATENCY_BUDGET_SEC = 1.5
# 低于该置信度 -> 进农技员复核队列（对应"AI 初筛 + 人工复核"闭环）
REVIEW_CONFIDENCE_THRESHOLD = 0.6


class AlgorithmService:
    """算法服务：一次 predict = 一次模型前向 + 一次契约组装。"""

    def __init__(self, predictor: Predictor, backend_status: Any = None) -> None:
        self.predictor = predictor
        self.status = backend_status

    # ------------------------------------------------------------------
    def _detections(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        """取出本次推理的所有框；分类后端没有 detections 就把整体当成一个框。"""
        dets = raw.get("detections")
        if dets:
            return list(dets)
        return [raw]

    # ------------------------------------------------------------------
    def _info(self, name: str) -> dict:
        """按"模型自带类别表 -> 全局标签表"的顺序解释类别。

        顺序不能反：2 类胁迫模型的 "stressed" 不在 15 类病害表里，
        用全局表查会得到"未知"（本机实测踩过）。模型类别表由 Predictor 从
        sidecar / checkpoint.json 构造。
        """
        table = getattr(self.predictor, "class_table", None) or {}
        if name in table:
            return dict(table[name])
        info = LABELS.by_name(name)
        return {
            "disease_name": info["disease_name"],
            "disease_name_cn": info["disease_name_cn"],
            "is_disease": bool(info.get("is_disease", name != "healthy")),
            "class_id": int(info["class_id"]),
        }

    # ------------------------------------------------------------------
    def predict_bgr(
        self,
        bgr: np.ndarray,
        image_id: str,
        *,
        crop: str | None = None,
        include_advice: bool = True,
    ) -> PredictResponse:
        t0 = time.perf_counter()
        raw = self.predictor.predict(bgr)
        h, w = bgr.shape[:2]

        # 决策级严重度：由**整图口径的症状面积占比**给出（切片推理时该值跨切片累加），
        # 比"框面积占比"更贴近真实发病程度（框里混着大量健康组织）。
        # 分类后端不提供这个量，此时保留分类头自己的判断。
        ratio = raw.get("severity_ratio")
        decided_severity: str | None = None
        if isinstance(ratio, (int, float)) and ratio >= 0:
            info = self._info(str(raw.get("disease_name", "healthy")))
            decided_severity = (
                "healthy" if not info.get("is_disease", False) or ratio <= 0
                else LABELS.severity_from_ratio(float(ratio))
            )

        boxes: list[BBoxOut] = []
        for d in self._detections(raw):
            info = self._info(str(d.get("disease_name", "healthy")))
            severity = d.get("severity") or (
                "healthy" if not info.get("is_disease") else LABELS.severity_from_ratio(0.01)
            )
            if decided_severity and info.get("is_disease", False):
                severity = decided_severity
            boxes.append(
                bbox_to_out(
                    disease_name=info["disease_name"],
                    disease_name_cn=info.get("disease_name_cn") or info["disease_name"],
                    confidence=float(d.get("confidence", 0.0)),
                    severity=severity,
                    class_id=int(d.get("class_id", info.get("class_id", -1))),
                    xyxy=d.get("bbox") or [0.0, 0.0, float(w), float(h)],
                    image_w=w,
                    image_h=h,
                )
            )

        # 统一按置信度降序，前端画框顺序稳定
        boxes.sort(key=lambda b: -b.confidence)
        disease_name, disease_name_cn, confidence, severity = top_disease(boxes)

        # 整图严重度：取所有框里最严重的那个（保守原则，宁可高估不漏报）
        order = {k: i for i, k in enumerate(LABELS.severity_keys)}
        if boxes:
            severity = max((b.severity for b in boxes), key=lambda s: order.get(s, 0))

        # 发病率口径：规则基线能直接给出"症状像素/整图面积"，优先用它；
        # 检测模型没有这个量，退回"框面积/整图面积"。
        ratio_override = raw.get("severity_ratio")
        stats = compute_statistics(
            boxes,
            w,
            h,
            diseased_ratio_override=float(ratio_override) if isinstance(ratio_override, (int, float)) else None,
        )
        elapsed = time.perf_counter() - t0

        resp = PredictResponse(
            code=200,
            message="success",
            image_id=image_id,
            disease_name=disease_name,
            disease_name_cn=disease_name_cn,
            confidence=round(float(confidence), 4),
            severity=severity,  # type: ignore[arg-type]
            boxes=boxes,
            processing_time=round(elapsed, 4),
            statistics=stats,
            model_version=getattr(self.predictor, "model_version", ""),
            backend=getattr(self.predictor, "name", ""),
        )

        if include_advice:
            advice = build_advice(
                disease_name,
                severity,
                crop=crop,
                disease_ratio=stats.diseased_ratio,
                confidence=float(confidence),
                confidence_threshold=REVIEW_CONFIDENCE_THRESHOLD,
            )
            resp.advice = advice
            resp.within_latency_budget = elapsed <= LATENCY_BUDGET_SEC

        return resp

    # ------------------------------------------------------------------
    def predict_by_id(self, image_id: str, crop: str | None = None) -> PredictResponse:
        """按 image_id 从 data/uploads/ 读图预测（对应 Apifox 请求方式 A）。"""
        from src.runtime.registry import UPLOADS_DIR

        path = _resolve_upload(image_id, UPLOADS_DIR)
        bgr = load_image_bgr(str(path))
        return self.predict_bgr(bgr, image_id, crop=crop)

    def predict_file(self, data: bytes, filename: str = "", crop: str | None = None) -> PredictResponse:
        """按上传文件预测（对应 Apifox 请求方式 B）。"""
        bgr = decode_image_bytes(data)
        image_id = filename or f"upload_{int(time.time() * 1000)}"
        return self.predict_bgr(bgr, image_id, crop=crop)


def _resolve_upload(image_id: str, uploads_dir: Any) -> Any:
    """从上传目录解析 image_id 到真实文件路径，做路径穿越防护。

    解析顺序：直接命中 -> 补扩展名 -> **递归子目录按文件名/主干名匹配**。
    最后一条是必需的：联调时经常把数据集按 test/train 分目录放进来
    （本机实测：154 张测试图放在 uploads/stress2_test/ 下，只查根目录会全 404）。
    """
    from pathlib import Path

    root = Path(uploads_dir)
    safe = Path(image_id).name  # 只取文件名，阻断 ../../ 之类
    direct = root / safe
    if direct.exists():
        return direct
    # 允许不带扩展名 / 扩展名不同
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        cand = root / f"{safe}{ext}"
        if cand.exists():
            return cand
    stem = Path(safe).stem
    for cand in root.glob(f"{stem}.*"):
        if cand.is_file():
            return cand
    # 递归子目录：按完整文件名，再按主干名（容忍 .rf.<hash> 后缀差异）
    for cand in root.rglob(safe):
        if cand.is_file():
            return cand
    for cand in root.rglob(f"{stem}.*"):
        if cand.is_file():
            return cand
    # 最后：主干名去掉 .rf 之后的 hash 再比对一次
    base_stem = stem.split(".rf.")[0]
    if base_stem and base_stem != stem:
        for cand in root.rglob(f"{base_stem}*"):
            if cand.is_file():
                return cand
    raise FileNotFoundError(
        f"找不到 image_id={image_id} 对应的图片，请先上传到 {root} 或改用 multipart 上传接口"
    )
