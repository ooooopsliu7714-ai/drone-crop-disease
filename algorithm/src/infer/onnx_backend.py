"""ONNX Runtime 推理后端：加载训练侧导出的 model.onnx + model.onnx.json。

设计要点：
  - sidecar json 是训练/推理之间的契约（字段见 scripts/export_onnx.py 的 docstring）；
  - 支持三类输出解码：detect_xyxy_score_cls / detect_detr / logits；
  - onnxruntime 缺失或模型文件缺失时，构造函数抛 ModelUnavailable，
    由 service 层决定降级到 heuristics 后端，绝不让接口 500。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.infer.heuristics import RawBox
from src.infer.postprocess import class_wise_nms, clip_boxes, filter_degenerate
from src.labels import LABELS

try:  # onnxruntime 是可选依赖：当前环境未安装
    import onnxruntime as ort  # type: ignore

    ORT_AVAILABLE = True
    ORT_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - 取决于环境
    ort = None  # type: ignore
    ORT_AVAILABLE = False
    ORT_IMPORT_ERROR = str(exc)


class ModelUnavailable(RuntimeError):
    """模型不可用（文件缺失 / 依赖缺失 / 元数据不合法）。"""


def _read_sidecar(onnx_path: Path) -> dict[str, Any]:
    sidecar = onnx_path.with_suffix(onnx_path.suffix + ".json")
    if not sidecar.exists():
        raise ModelUnavailable(f"缺少元数据文件：{sidecar.name}（应由 scripts/export_onnx.py 生成）")
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelUnavailable(f"元数据 {sidecar.name} 不是合法 JSON：{exc}") from exc


def letterbox(
    img: np.ndarray, size: tuple[int, int], color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, float, float, float]:
    """等比缩放到目标尺寸并居中填充，返回 (图, 缩放比, 左填充, 上填充)。"""
    target_w, target_h = size
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, float(pad_x), float(pad_y)


class OnnxDetector:
    """ONNX 检测/分类后端，统一输出 RawBox 列表。"""

    name = "onnxruntime"

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        device: str = "cpu",
        score_threshold: float | None = None,
        iou_threshold: float | None = None,
        max_det: int = 300,
    ) -> None:
        if not ORT_AVAILABLE:
            raise ModelUnavailable(
                "未安装 onnxruntime，无法加载 ONNX 模型。"
                "CPU 推理：pip install onnxruntime；GPU 推理：pip install onnxruntime-gpu"
                f"（原始错误：{ORT_IMPORT_ERROR}）"
            )
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise ModelUnavailable(f"找不到模型文件：{self.onnx_path}")
        self.meta = _read_sidecar(self.onnx_path)
        self.model_version = f"{self.onnx_path.parent.name}:{self.meta.get('meta', {}).get('source', 'onnx')}"

        # 阈值优先级：显式入参 > sidecar 标定值 > 保守默认。
        #
        # 为什么必须读 sidecar（本机实测抓到的部署级不一致）：模型训练后分数整体偏低
        # （每图最高分仅 0.13~0.26），阈值 0.25 会筛掉几乎所有框，因此在 val 集上
        # 标定出 0.10 并写进 sidecar 的 meta.conf_threshold。但本类原先把
        # score_threshold 写死为 0.25 且 registry 构造时不传参 →
        # **torch 后端用 0.10、ONNX 后端用 0.25，同一条降级链上两个后端给出不同结果**，
        # 而 /health 只报后端名，排查时极难发现。统一以 sidecar 为准即可根除。
        meta = self.meta.get("meta") or {}
        meta_conf = meta.get("conf_threshold")
        meta_iou = meta.get("iou_threshold")
        self.score_threshold = float(
            score_threshold if score_threshold is not None
            else (meta_conf if meta_conf is not None else 0.25)
        )
        self.iou_threshold = float(
            iou_threshold if iou_threshold is not None
            else (meta_iou if meta_iou is not None else 0.45)
        )
        self.max_det = int(max_det)

        inp = self.meta.get("input", {})
        hw = inp.get("size", [640, 640])
        self.input_size = (int(hw[1]), int(hw[0]))  # (W, H)
        self.input_name = inp.get("name", "images")
        self.mean = np.asarray(inp.get("mean", [0, 0, 0]), dtype=np.float32)
        self.std = np.asarray(inp.get("std", [255, 255, 255]), dtype=np.float32)
        self.color = str(inp.get("color", "RGB")).upper()
        self.task = str(self.meta.get("task", "detect"))
        self.out_kind = str(self.meta.get("outputs", {}).get("kind", "detect_xyxy_score_cls"))

        providers = ["CPUExecutionProvider"]
        if device not in ("cpu", "") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.output_names = [o.name for o in self.session.get_outputs()]
        # 类别表**以 sidecar 为准**，只在 sidecar 缺失时才回落到 labels.yaml。
        # 为什么必须这样：同一套代码要同时服务 15 类病害模型和 2 类胁迫模型
        # （外部数据集只有 healthy/stressed）。若这里读全局 labels.yaml，
        # 2 类模型输出的 class_id=1 会被解释成"稻瘟病"而不是"stressed"——
        # 这种静默错标比直接报错危险得多。
        self.classes = self.meta.get("classes") or LABELS.to_sidecar_classes()
        self._build_class_map()

    def _build_class_map(self) -> None:
        """由 self.classes 构造 class_id -> 类别信息 的查表。

        单独抽成方法是为了让测试可以用 __new__ 绕过 __init__（避免依赖 onnxruntime），
        但依然能构造出完整的类别表。
        """
        self._class_map: dict[int, dict[str, Any]] = {}
        for i, c in enumerate(self.classes or []):
            cid = int(c.get("class_id", i))
            name = str(c.get("disease_name", f"class_{i}"))
            self._class_map[cid] = {
                "class_id": cid,
                "disease_name": name,
                "disease_name_cn": str(c.get("disease_name_cn") or name),
                "is_disease": bool(c.get("is_disease", name != "healthy")),
            }

    def _info(self, class_id: int) -> dict[str, Any]:
        """按 sidecar 类别表解释 class_id；未知 id 给可读占位而不抛异常。"""
        return self._class_map.get(
            int(class_id),
            {
                "class_id": int(class_id),
                "disease_name": f"class_{class_id}",
                "disease_name_cn": f"未知类别({class_id})",
                "is_disease": True,
            },
        )

    # ------------------------------------------------------------------
    def _preprocess(self, bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        img, scale, pad_x, pad_y = letterbox(bgr, self.input_size)
        if self.color == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.astype(np.float32)
        x = (x - self.mean) / np.maximum(self.std, 1e-6)
        x = np.transpose(x, (2, 0, 1))[None]  # NCHW
        return np.ascontiguousarray(x, dtype=np.float32), scale, pad_x, pad_y

    # ------------------------------------------------------------------
    def _decode(
        self, outputs: list[np.ndarray], scale: float, pad_x: float, pad_y: float, shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """把模型原始输出解码成 (boxes xyxy 原图像素, scores, class_ids)。

        坐标链（很容易出错，务必按顺序）：
            模型输出(归一化, 相对 letterbox 画布)
              × (input_w, input_h)      -> letterbox 画布像素
              - (pad_x, pad_y)          -> 去掉填充
              ÷ scale                   -> 原图像素
        早期版本漏了第一步，导致框被压缩到图像左上角附近且"看起来正常"，
        属于典型的静默错误，因此 tests/test_onnx_decode.py 专门锁住这条链。
        """
        h, w = shape
        named = dict(zip(self.output_names, outputs))
        input_w, input_h = float(self.input_size[0]), float(self.input_size[1])

        if self.out_kind == "logits":
            logits = np.asarray(outputs[0]).reshape(-1)
            exp = np.exp(logits - logits.max())
            probs = exp / max(1e-9, exp.sum())
            cid = int(np.argmax(probs))
            box = np.array([[0.0, 0.0, float(w), float(h)]], dtype=np.float32)
            return box, probs[cid : cid + 1].astype(np.float32), np.array([cid], dtype=np.int32)

        if self.out_kind == "detect_detr":
            # RT-DETR 原生输出：logits (1,Q,C) + pred_boxes (1,Q,4) cxcywh，归一化到输入画布
            logits = np.asarray(named.get("logits", outputs[0]))[0]
            boxes = np.asarray(named.get("pred_boxes", outputs[1]))[0]

            # 必须用 **sigmoid**，不能用 softmax。
            #
            # 真实事故（本机实测）：RT-DETR 的类别分支是用 sigmoid + VFL 训练的
            # （每个类独立二分类，可以"全部都不是"）。若改用 softmax：
            #   · 2 类的 softmax 最大值恒 >= 0.5；
            #   · 背景查询的 logits 通常两个都是负的，sigmoid 下分数约 0.01（会被阈值滤掉），
            #     但 softmax 下会变成"类别 0、分数 0.5~0.9"的**高置信度假框**。
            # 结果：ONNX 后端每图吐出 48~71 个全是"healthy"、置信度 0.9+ 的伪框，
            # 而 torch 后端（用 HF 的 post_process，走 sigmoid）只给 12~13 个框、
            # 主病害为 stressed。**同一模型、同一条降级链，两个后端结论相反。**
            # 改用 sigmoid 后两者一致。
            probs = 1.0 / (1.0 + np.exp(-logits))
            scores = probs.max(axis=-1)
            cids = probs.argmax(axis=-1)
            cx, cy, bw, bh = boxes.T
            xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
            xyxy = xyxy * np.array([input_w, input_h, input_w, input_h], dtype=np.float32)
        else:
            # 常见导出格式：outputs[0] = (1, N, 4+1+C) 或 (N, 6)
            pred = np.asarray(outputs[0])
            pred = pred[0] if pred.ndim == 3 else pred
            if pred.shape[1] == 6:  # xyxy(像素), score, cls
                xyxy, scores, cids = pred[:, :4], pred[:, 4], pred[:, 5].astype(np.int32)
            else:  # xywh + obj + cls probs（通常是归一化）
                cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
                xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
                cls_probs = pred[:, 5:]
                cids = cls_probs.argmax(axis=1).astype(np.int32)
                scores = cls_probs.max(axis=1)
                if xyxy.size and xyxy.max() <= 1.5:
                    xyxy = xyxy * np.array([input_w, input_h, input_w, input_h], dtype=np.float32)
            xyxy = np.asarray(xyxy, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)
            cids = np.asarray(cids, dtype=np.int32)

        # 从 letterbox 画布坐标还原到原图坐标
        xyxy = (xyxy - np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)) / max(scale, 1e-6)
        return xyxy.astype(np.float32), np.asarray(scores, dtype=np.float32), np.asarray(cids, dtype=np.int32)

    # ------------------------------------------------------------------
    def predict(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for bgr in images:
            h, w = bgr.shape[:2]
            x, scale, pad_x, pad_y = self._preprocess(bgr)
            outputs = self.session.run(self.output_names, {self.input_name: x})
            boxes, scores, cids = self._decode(outputs, scale, pad_x, pad_y, (h, w))
            boxes = clip_boxes(boxes, w, h)
            keep = scores >= self.score_threshold
            boxes, scores, cids = boxes[keep], scores[keep], cids[keep]
            boxes, scores, cids = filter_degenerate(boxes, scores, cids, min_size=2.0)
            order = class_wise_nms(boxes, scores, cids, self.iou_threshold, self.max_det)
            boxes, scores, cids = boxes[order], scores[order], cids[order]
            results.append(self._to_boxes(boxes, scores, cids, (h, w)))
        return results

    # ------------------------------------------------------------------
    def _healthy_info(self) -> dict[str, Any]:
        """找"健康"类的信息（key 可能是 healthy / normal，找不到就用 class 0）。"""
        for c in self.classes:
            if str(c.get("disease_name", "")).lower() in ("healthy", "normal", "health"):
                return self._info(int(c.get("class_id", 0)))
        return self._info(0)

    def _to_boxes(
        self, boxes: np.ndarray, scores: np.ndarray, cids: np.ndarray, shape: tuple[int, int]
    ) -> dict[str, Any]:
        """输出与 src.train.predictors.build_predictor 相同的契约字典。"""
        h, w = shape
        raw: list[RawBox] = []
        total_area = max(1.0, float(w) * float(h))
        for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), cids.tolist()):
            info = self._info(int(c))
            ratio = max(0.0, (x2 - x1) * (y2 - y1)) / total_area
            severity = "healthy" if not info.get("is_disease", False) else LABELS.severity_from_ratio(ratio)
            raw.append(
                RawBox(
                    xyxy=(x1, y1, x2, y2),
                    class_id=int(c),
                    confidence=float(s),
                    severity=severity,
                )
            )

        if self.task == "classify":
            # 分类任务：整图一个结果，其余类别可通过 top-5 提供给复核端
            if raw:
                best = max(raw, key=lambda r: r.confidence)
                out = best.as_dict(w, h)
            else:
                info = self._healthy_info()
                out = {
                    "disease_name": info["disease_name"],
                    "disease_name_cn": info["disease_name_cn"],
                    "confidence": 0.0,
                    "severity": "healthy",
                    "class_id": int(info["class_id"]),
                    "bbox": [0.0, 0.0, float(w), float(h)],
                }
            out["topk"] = [
                {
                    "disease_name": self._info(r.class_id)["disease_name"],
                    "confidence": round(r.confidence, 4),
                }
                for r in sorted(raw, key=lambda r: -r.confidence)[:5]
            ]
            return out

        # 检测任务：返回所有框，主病害由 src.api.service 选置信度最高的
        primary = max(raw, key=lambda r: r.confidence) if raw else None
        if primary is None:
            info = self._healthy_info()
            return {
                "disease_name": info["disease_name"],
                "disease_name_cn": info["disease_name_cn"],
                "confidence": 0.0,
                "severity": "healthy",
                "class_id": int(info["class_id"]),
                "bbox": [0.0, 0.0, float(w), float(h)],
                "detections": [],
            }
        result = primary.as_dict(w, h)
        result["detections"] = [r.as_dict(w, h) for r in raw]
        return result
