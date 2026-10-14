"""统一推理入口（torch 后端，不依赖 onnxruntime）—— 供 ``src/infer/`` 直接 import。

**这是与推理侧的核心接口，签名与返回字段不能改。**

怎么用::

    from src.train.predictors import build_predictor

    predictor = build_predictor("models/cls_dinov2", device="auto")   # 或 "cuda" / "cpu"
    results = predictor.predict([bgr_image])          # list[np.ndarray] HxWx3 uint8 (BGR, OpenCV 习惯)
    # 每个结果恰好是：
    # {"disease_name": "rice_blast", "confidence": 0.93, "severity": "moderate",
    #  "class_id": 1, "bbox": [x1, y1, x2, y2]}

支持的后端（由 ``models/<name>/model.onnx.json`` 的 ``meta.source`` 自动选择）：
    - ``rtdetrv2`` / ``rtdetr``：transformers ``RTDetrV2ForObjectDetection``（自训权重）
    - ``ultralytics-rtdetr`` / ``yoloworld``：ultralytics ``YOLO``（惰性 import）
    - ``dinov2-*`` / ``dinov3-*``：DINOv2/v3 backbone + 线性头（分类）
    - ``grounding-dino``：开放词汇零样本检测（文本提示，类别可扩展）

依赖：torch、numpy、Pillow 必需；transformers / ultralytics 按后端惰性 import，
缺失时给出中文提示并抛 RuntimeError（不打印 ImportError 栈）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.labels import LabelBook, load_labels, severity_from_area_ratio
from src.train.exporter import (
    SIDE_CAR_FILENAME,
    WEIGHT_FILENAME,
    load_sidecar,
)

__all__ = ["build_predictor", "BasePredictor"]

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.5


# ---------------------------------------------------------------------------
# 图像工具（BGR numpy <-> RGB PIL，不引入额外依赖）
# ---------------------------------------------------------------------------
def _bgr_to_pil(image: Any):
    """BGR uint8 ndarray -> RGB PIL.Image。"""
    import numpy as np
    from PIL import Image

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    rgb = arr[:, :, ::-1]  # BGR -> RGB
    return Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")


def _pil_to_nchw(pil, mean: Sequence[float], std: Sequence[float], size: int):
    """PIL -> 归一化 float32 NCHW tensor（与训练预处理保持一致：BICUBIC resize）。"""
    import numpy as np
    import torch

    from PIL import Image

    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    m = np.asarray(list(mean) if len(mean) == 3 else [mean[0]] * 3, dtype=np.float32)
    s = np.asarray(list(std) if len(std) == 3 else [std[0]] * 3, dtype=np.float32)
    s = np.where(np.abs(s) < 1e-12, 1.0, s)
    arr = (arr - m) / s
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return torch.from_numpy(np.ascontiguousarray(arr))


def _nms_numpy(boxes, scores, iou_thr: float = DEFAULT_IOU) -> List[int]:
    """纯 numpy NMS，返回保留的下标列表（按分数降序）。"""
    import numpy as np

    if len(boxes) == 0:
        return []
    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    order = s.argsort()[::-1]
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_thr]
    return keep


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------
def _clip_boxes_to_image(boxes: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
    """把框裁剪到图像范围内，并只丢掉**真正为空（零面积）**的框。

    为什么需要裁剪：RT-DETR 的输出坐标不受图像边界约束，实测会出现
    `x1 = -0.29` 这种负坐标。接口契约要求 `0 <= x1 <= x2 <= w`，
    服务层虽然会再裁一次，但预测器自己就该给出合法框。

    ⚠️ 为什么只丢零面积框、不过滤"过小的框"：早期版本用
    `if x2 - x1 < 1.0: continue`，当模型输出退化框时它会把**所有框静默删光**，
    评估里显示"预测框 0 个"，把真正的病根（训练目标编码错误）掩盖成了
    "模型没学到东西"，排查方向被带偏。**静默丢数据比直接暴露更难查**，
    所以这里只剔除零面积框，其余交给下游判断。
    """
    out: List[Dict[str, Any]] = []
    for b in boxes or []:
        bb = list(b.get("bbox") or [])
        if len(bb) != 4:
            out.append(b)
            continue
        x1, y1, x2, y2 = (float(v) for v in bb)
        x1, x2 = max(0.0, min(x1, float(width))), max(0.0, min(x2, float(width)))
        y1, y2 = max(0.0, min(y1, float(height))), max(0.0, min(y2, float(height)))
        if x2 <= x1 or y2 <= y1:
            continue  # 裁完已无面积，确属无效框
        nb = dict(b)
        nb["bbox"] = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        out.append(nb)
    return out


class BasePredictor:
    """所有后端的公共逻辑：标签加载、结果组装、批量推理入口。"""

    def __init__(self, ckpt_dir: str | Path, sidecar: Dict[str, Any], device: str = "cpu") -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.sidecar = sidecar
        self.device = device
        self.label_book: LabelBook = self._build_label_book()
        # 置信度阈值优先级：环境变量 > sidecar 里标定的值 > 全局默认
        #
        # 为什么需要环境变量这一层：模型训完后**分数分布可能整体偏低**
        # （本机实测：最高分只有 0.13~0.26，而默认阈值是 0.25）。
        # 这时服务端会几乎返回不了框——实测同一模型 conf=0.05 时召回 0.77，
        # conf=0.25 时召回 0.007，看起来像"模型没用"，其实是阈值没标定。
        env_conf = os.environ.get("ALGO_CONF_THRESHOLD", "").strip()
        sidecar_conf = sidecar.get("meta", {}).get("conf_threshold")
        if env_conf:
            try:
                self.conf_threshold = float(env_conf)
            except ValueError:
                self.conf_threshold = float(sidecar_conf or DEFAULT_CONF)
        else:
            self.conf_threshold = float(sidecar_conf if sidecar_conf is not None else DEFAULT_CONF)
        self.iou_threshold = float(sidecar.get("meta", {}).get("iou_threshold", DEFAULT_IOU))
        self.imgsz = int(sidecar.get("meta", {}).get("imgsz") or sidecar["input"]["size"][0])
        self._warned_keys = False

    def _build_label_book(self) -> LabelBook:
        """构造标签表：**产物自带的类别优先**，缺失时才回退全局 labels.yaml。

        为什么必须这样：同一个 models/ 目录下可能同时放着 15 类病害模型和
        2 类胁迫模型。若一律用全局 15 类表解释，2 类模型的 class_id=1
        会被读成"稻瘟病"而不是"stressed"——本机实测踩过，属于静默错标。
        产物里的类别表来自训练时写入的 checkpoint.json / sidecar。
        """
        classes = None
        # sidecar 里可能直接带 classes
        if isinstance(self.sidecar.get("classes"), list) and self.sidecar["classes"]:
            classes = self.sidecar["classes"]
        else:
            for fn in ("checkpoint.json", "model.onnx.json"):
                p = self.ckpt_dir / fn
                if not p.is_file():
                    continue
                try:
                    meta = json.loads(p.read_text(encoding="utf-8-sig"))
                except Exception:
                    continue
                if isinstance(meta.get("classes"), list) and meta["classes"]:
                    classes = meta["classes"]
                    break
        if classes:
            try:
                from src.core.labels import _Cls  # 训练侧的类别记录类型

                return LabelBook(
                    [
                        _Cls(
                            class_id=int(c.get("class_id", i)),
                            disease_name=str(c.get("disease_name", f"class_{i}")),
                            disease_name_cn=str(c.get("disease_name_cn") or c.get("disease_name") or f"class_{i}"),
                            crop=str(c.get("crop") or "generic"),
                            is_disease=bool(
                                c.get("is_disease", str(c.get("disease_name", "")) != "healthy")
                            ),
                        )
                        for i, c in enumerate(classes)
                    ]
                )
            except Exception as exc:
                print(f"[警告] 用产物自带类别表构造 LabelBook 失败（{exc}），回退全局 labels.yaml")
        return load_labels()

    # ---- 子类实现 -------------------------------------------------------
    def _infer_one(self, image_bgr: Any) -> List[Dict[str, Any]]:
        """子类实现：返回该图的框列表 ``[{class_id, confidence, bbox}]``（像素 xyxy）。"""
        raise NotImplementedError

    def _enrich(self, boxes: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
        """给裸框补上 disease_name / disease_name_cn / severity。

        为什么必须补：ONNX 后端的框自带这些字段，而 torch 后端只给
        ``{class_id, confidence, bbox}``。接口层（src/api/service.py）按
        ``d.get("disease_name", "healthy")`` 取值，字段缺失时会**把所有框当成 healthy**
        ——本机实测：模型明明检出 18 个"受胁迫"框，接口输出的 20 个框全是"健康"，
        演示时看起来像算法失效。两边形状必须一致。
        """
        total = max(1.0, float(width) * float(height))
        out: List[Dict[str, Any]] = []
        for b in boxes or []:
            cid = int(b.get("class_id", -1))
            x1, y1, x2, y2 = (float(v) for v in (b.get("bbox") or [0, 0, 0, 0])[:4])
            ratio = max(0.0, (x2 - x1) * (y2 - y1)) / total
            try:
                name = self.label_book.name_of(cid)
                name_cn = self.label_book.cn_of(cid)
                is_dis = self.label_book.is_disease(cid)
            except Exception:
                name, name_cn, is_dis = f"class_{cid}", f"class_{cid}", True
            nb = dict(b)
            nb.update(
                {
                    "disease_name": name,
                    "disease_name_cn": name_cn,
                    "severity": "healthy" if not is_dis else severity_from_area_ratio(ratio),
                }
            )
            out.append(nb)
        return out

    # ---- 对外接口 -------------------------------------------------------
    def predict(self, images: List[Any]) -> List[Dict[str, Any]]:
        """批量预测。

        参数:
            images: BGR uint8 ndarray 列表（OpenCV 读入的 HxWxH3）
        返回:
            list[dict]，每个 dict 含 disease_name / confidence / severity / class_id / bbox，
            **并额外带 `detections`（全部框的列表）**。

        为什么必须带 `detections`：接口层（`src/api/service.py`）靠这个字段拿到所有框，
        没有它就只能把整个结果当成"一个框"——本机实测：接口对每张图只返回 1 个框，
        而模型实际输出了几十个，前端看到的框完全不对（还以为是模型不行）。
        ONNX 后端一直有 `detections`，torch 后端缺了，这里是补齐。
        """
        if images is None:
            raise ValueError("predict(images) 收到 None，应为 list[np.ndarray]")
        if not isinstance(images, (list, tuple)):
            images = [images]
        out: List[Dict[str, Any]] = []
        for img in images:
            h, w = int(img.shape[0]), int(img.shape[1])
            boxes = self._enrich(_clip_boxes_to_image(self._infer_one(img), w, h), w, h)
            summary = self.label_book.summarize_boxes(boxes, w, h)
            out.append(
                {
                    "disease_name": summary["disease_name"],
                    "confidence": float(summary["confidence"]),
                    "severity": summary["severity"],
                    "class_id": int(summary["class_id"]),
                    "bbox": self._primary_bbox(boxes, summary, w, h),
                    # 全部框：接口层靠它输出 boxes[]；缺了它服务端只会返回 1 个框
                    "detections": boxes,
                    "num_boxes": len(boxes),
                    "disease_area_ratio": summary.get("disease_area_ratio", 0.0),
                }
            )
        return out

    def predict_debug(self, images: List[Any]) -> List[Dict[str, Any]]:
        """调试用：返回主结果 + 全部框 + 面积占比（推理侧正式接口请用 predict）。"""
        if not isinstance(images, (list, tuple)):
            images = [images]
        out: List[Dict[str, Any]] = []
        for img in images:
            h, w = int(img.shape[0]), int(img.shape[1])
            boxes = _clip_boxes_to_image(self._infer_one(img), w, h)
            summary = self.label_book.summarize_boxes(boxes, w, h)
            rec = {
                "disease_name": summary["disease_name"],
                "confidence": float(summary["confidence"]),
                "severity": summary["severity"],
                "class_id": int(summary["class_id"]),
                "bbox": self._primary_bbox(boxes, summary, w, h),
            }
            rec["boxes"] = boxes
            rec["disease_area_ratio"] = summary["disease_area_ratio"]
            rec["num_boxes"] = summary["num_boxes"]
            out.append(rec)
        return out

    # ---- 工具 -----------------------------------------------------------
    def _primary_bbox(self, boxes, summary, width: int, height: int) -> List[float]:
        """分类任务返回全图框；检测任务返回主病害框（无框时用最大框）。"""
        if summary["num_boxes"] == 0:
            return [0.0, 0.0, float(width), float(height)]
        pos = [b for b in boxes if self.label_book.is_disease(int(b["class_id"]))]
        pool = pos or list(boxes)
        best = max(pool, key=lambda b: float(b.get("confidence", 0.0)))
        return [float(round(v, 2)) for v in best["bbox"][:4]]

    def _boxes_to_records(self, boxes, scores, labels) -> List[Dict[str, Any]]:
        """把 (N,4)xyxy / (N,)score / (N,)label 转成统一记录并做 NMS + 阈值过滤。"""
        import numpy as np

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        out: List[Dict[str, Any]] = []
        if boxes.size == 0:
            return out
        # 按类别分组做 NMS（避免不同病害互相压制）
        for cid in sorted(set(int(v) for v in labels.tolist())):
            m = labels.astype(int) == int(cid)
            b = boxes[m]
            s = scores[m]
            keep = _nms_numpy(b, s, self.iou_threshold)
            for k in keep:
                if float(s[k]) < self.conf_threshold:
                    continue
                out.append(
                    {
                        "class_id": int(cid),
                        "confidence": float(round(float(s[k]), 6)),
                        "bbox": [float(round(float(v), 2)) for v in b[k][:4]],
                    }
                )
        out.sort(key=lambda r: -r["confidence"])
        return out

    def _remind_keys(self) -> None:
        """只提醒一次：接口主字段是 disease_name / confidence / boxes。"""
        if not self._warned_keys:
            self._warned_keys = True
            print(
                "[信息] 推理结果主字段：disease_name / confidence / severity / class_id / bbox；"
                "接口层 (src/api) 需自行补 disease_name_cn（可用 src.core.labels 的 cn_of 映射）。"
            )


# ---------------------------------------------------------------------------
# 后端 1：transformers RT-DETR
# ---------------------------------------------------------------------------
class RTDetrTorchPredictor(BasePredictor):
    """RT-DETRv2（transformers 实现）torch 后端推理。"""

    def __init__(self, ckpt_dir, sidecar, device="cpu", model_name: str = "PekingU/rtdetr_v2_r18vd") -> None:
        super().__init__(ckpt_dir, sidecar, device)
        try:
            import torch
            from transformers import RTDetrV2ForObjectDetection
        except Exception as exc:
            raise RuntimeError(
                "RT-DETR torch 推理需要 transformers 与 torch。\n"
                "  pip install transformers torch\n"
                f"（原始错误：{exc}）"
            ) from exc
        self.torch = torch
        self.processor = None
        ckpt_json = self.ckpt_dir / "checkpoint.json"
        if ckpt_json.is_file():
            with open(ckpt_json, "r", encoding="utf-8-sig") as f:
                info = json.load(f)
            model_name = str(info.get("pretrained") or model_name)
        cfg = self.ckpt_dir / "config.json"
        wpath = self.ckpt_dir / WEIGHT_FILENAME
        has_hf_weights = any(
            (self.ckpt_dir / n).is_file() for n in ("model.safetensors", "pytorch_model.bin")
        )
        if cfg.is_file() and has_hf_weights:
            # Trainer 保存的完整 HF 模型目录
            self.processor = self._load_processor(self.ckpt_dir, model_name)
            self.model = RTDetrV2ForObjectDetection.from_pretrained(str(self.ckpt_dir))
        else:
            # 自写训练循环的产物：checkpoint.json + config.json + best.pt(state_dict)。
            # 注意：**写目录路径给 from_pretrained 会失败**——目录里有 config.json
            # 但没有 model.safetensors/pytorch_model.bin，报
            # "Error no file named model.safetensors ... found in directory"。
            # 所以这里先用预训练名重建网络结构（config.json 存在时用它的结构配置，
            # 保证层数/通道与权重完全一致），再手动载入 best.pt。
            load_src: Any = model_name
            cfg_obj = None
            if cfg.is_file():
                try:
                    from transformers import RTDetrV2Config

                    cfg_obj = RTDetrV2Config.from_pretrained(str(self.ckpt_dir))
                    load_src = cfg_obj
                except Exception as exc:  # 结构配置读不出来时退回预训练名
                    print(f"[警告] 读取 {cfg} 失败（{exc}），改用预训练结构 {model_name}")
                    cfg_obj = None
            self.processor = self._load_processor(None, model_name)
            # transformers 5.x 里 from_pretrained 的**第一个位置参数必须是模型名/repo id**，
            # 传 config 对象会被当成 repo id 并报
            # "Repo id must use alphanumeric chars ...: 'RTDetrV2Config {...}'"。
            # 因此结构配置走 `config=` 关键字；同时 5.x 把 num_labels 移到了 config 上，
            # 直接传 num_labels= 会报 "unexpected keyword argument"。
            if cfg_obj is not None:
                cfg_obj.num_labels = max(1, self.label_book.num_classes)
                self.model = RTDetrV2ForObjectDetection.from_pretrained(
                    model_name, config=cfg_obj, ignore_mismatched_sizes=True
                )
            else:
                try:
                    from transformers import RTDetrV2Config as _Cfg

                    cfg_obj = _Cfg.from_pretrained(model_name)
                    cfg_obj.num_labels = max(1, self.label_book.num_classes)
                    self.model = RTDetrV2ForObjectDetection.from_pretrained(
                        model_name, config=cfg_obj, ignore_mismatched_sizes=True
                    )
                except Exception:
                    # 最老版本仍接受 num_labels，作为兜底
                    self.model = RTDetrV2ForObjectDetection.from_pretrained(
                        model_name,
                        num_labels=max(1, self.label_book.num_classes),
                        ignore_mismatched_sizes=True,
                    )
            if wpath.is_file():
                state = _load_state_dict(wpath)
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                print(f"[信息] 载入自训权重 {wpath.name}：missing={len(missing)} unexpected={len(unexpected)}")
        self.model.to(self.device).eval()
        self._remind_keys()

    def _load_processor(self, local_dir, fallback_name: str):
        """优先用训练时保存的 processor，其次从预训练名加载。"""
        from transformers import RTDetrImageProcessor

        for cand in (local_dir, fallback_name):
            if cand is None:
                continue
            try:
                proc = RTDetrImageProcessor.from_pretrained(str(cand))
                proc.size = {"height": self.imgsz, "width": self.imgsz}
                return proc
            except Exception:
                continue
        raise RuntimeError(
            f"无法加载 RTDetrImageProcessor（尝试了 {local_dir} 与 {fallback_name}）。"
            "离线环境下请把 processor 文件一起放进 models/<name>/。"
        )

    def _infer_one(self, image_bgr) -> List[Dict[str, Any]]:
        pil = _bgr_to_pil(image_bgr)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            outs = self.model(**inputs)
        target_sizes = self.torch.tensor([pil.size[::-1]], device=self.device)  # (h, w)
        res = self.processor.post_process_object_detection(
            outs, target_sizes=target_sizes, threshold=min(self.conf_threshold, 0.05)
        )[0]
        return self._boxes_to_records(
            res["boxes"].detach().cpu().numpy(),
            res["scores"].detach().cpu().numpy(),
            res["labels"].detach().cpu().numpy(),
        )


# ---------------------------------------------------------------------------
# 后端 2：ultralytics（RT-DETR / YOLO-World）
# ---------------------------------------------------------------------------
class UltralyticsPredictor(BasePredictor):
    """ultralytics YOLO 系列（含 RT-DETR、YOLO-World）torch 后端推理。"""

    def __init__(self, ckpt_dir, sidecar, device="cpu") -> None:
        super().__init__(ckpt_dir, sidecar, device)
        try:
            from ultralytics import YOLO  # 惰性 import
        except Exception as exc:
            raise RuntimeError(
                "该权重需要 ultralytics 才能推理（--backend ultralytics-rtdetr / yoloworld 产出的 best.pt 是 ultralytics 格式）。\n"
                "  pip install ultralytics\n"
                f"（原始错误：{exc}）"
            ) from exc
        wpath = self.ckpt_dir / WEIGHT_FILENAME
        if not wpath.is_file():
            raise FileNotFoundError(f"找不到权重：{wpath}")
        self.model = YOLO(str(wpath))
        self.names = self.sidecar.get("meta", {}).get("class_names") or self.label_book.names
        self._remind_keys()

    def _infer_one(self, image_bgr) -> List[Dict[str, Any]]:
        import numpy as np

        results = self.model.predict(
            source=np.asarray(image_bgr),
            imgsz=self.imgsz,
            conf=min(self.conf_threshold, 0.05),
            iou=self.iou_threshold,
            device=self.device if self.device == "cpu" else 0,
            verbose=False,
        )
        out: List[Dict[str, Any]] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            conf = r.boxes.conf.detach().cpu().numpy()
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, conf, cls):
                if float(c) < self.conf_threshold:
                    continue
                out.append(
                    {
                        "class_id": int(k),
                        "confidence": float(round(float(c), 6)),
                        "bbox": [float(round(float(v), 2)) for v in b[:4]],
                    }
                )
        out.sort(key=lambda x: -x["confidence"])
        return out


# ---------------------------------------------------------------------------
# 后端 3：DINOv2 / DINOv3 分类
# ---------------------------------------------------------------------------
class DinovClassifierPredictor(BasePredictor):
    """DINOv2/DINOv3 backbone + 线性分类头的 torch 后端推理。

    直接复用 ``src.train.classifiers.DinoClassifier`` 的同一套结构，
    保证训练保存的 ``best.pt``（state_dict）能 1:1 载入，避免结构漂移。
    """

    def __init__(self, ckpt_dir, sidecar, device="cpu") -> None:
        super().__init__(ckpt_dir, sidecar, device)
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "DINOv2/DINOv3 分类推理需要 transformers 与 torch。\n"
                "  pip install transformers torch\n"
                f"（原始错误：{exc}）"
            ) from exc
        from src.train.classifiers import DinoClassifier

        self.meta: Dict[str, Any] = {}
        ckpt_json = self.ckpt_dir / "checkpoint.json"
        if ckpt_json.is_file():
            with open(ckpt_json, "r", encoding="utf-8-sig") as f:
                self.meta = json.load(f)
        backbone = str(self.meta.get("backbone") or sidecar.get("meta", {}).get("backbone") or "")
        if not backbone:
            raise RuntimeError(f"{self.ckpt_dir}/checkpoint.json 缺少 backbone 字段，无法重建模型")
        num_classes = int(self.meta.get("num_classes") or sidecar.get("meta", {}).get("num_classes") or self.label_book.num_classes)
        self.clf = DinoClassifier(
            backbone,
            num_classes=num_classes,
            pretrained=False,  # 结构从 config 建，权重全部来自 best.pt
            pooling=str(self.meta.get("pooling") or sidecar.get("meta", {}).get("pooling") or "cls"),
            dropout=0.0,
        )
        wpath = self.ckpt_dir / WEIGHT_FILENAME
        if not wpath.is_file():
            raise FileNotFoundError(f"找不到权重：{wpath}")
        state = _load_state_dict(wpath)
        missing, unexpected = self.clf.load_state_dict(state, strict=False)
        if len(missing) > 3 or len(unexpected) > 3:
            print(f"[警告] 分类权重载入不完全匹配：missing={len(missing)} unexpected={len(unexpected)}")
            print(f"        missing 示例：{list(missing)[:5]}")
        else:
            print(f"[信息] 载入分类权重 {wpath.name}（missing={len(missing)} unexpected={len(unexpected)}）")
        self.clf.to(self.device)
        self.clf.net.eval()
        self.mean, self.std = self._norm_from_sidecar()
        self._remind_keys()

    def _norm_from_sidecar(self) -> Tuple[Sequence[float], Sequence[float]]:
        """从 sidecar 里取归一化参数（训练侧写入的就是真实预处理参数）。"""
        inp = self.sidecar.get("input", {})
        return inp.get("mean", (0.485, 0.456, 0.406)), inp.get("std", (0.229, 0.224, 0.225))

    def _infer_one(self, image_bgr) -> List[Dict[str, Any]]:
        import torch

        pil = _bgr_to_pil(image_bgr)
        x = _pil_to_nchw(pil, self.mean, self.std, self.imgsz).to(self.device)
        with torch.no_grad():
            logits = self.clf.net(pixel_values=x)
            probs = torch.softmax(logits.float(), dim=-1)[0]
        # 只取 labels.yaml 范围内的高分候选（num_classes 可能大于当前标签体系）
        n = min(int(probs.shape[-1]), self.label_book.num_classes)
        top = int(torch.argmax(probs[:n]).item())
        return [{
            "class_id": top,
            "confidence": float(round(float(probs[top].item()), 6)),
            "bbox": [0.0, 0.0, float(pil.width), float(pil.height)],
        }]


# ---------------------------------------------------------------------------
# 后端 4：Grounding DINO 开放词汇（零样本，不给标注也能出新类别框）
# ---------------------------------------------------------------------------
class GroundingDinoPredictor(BasePredictor):
    """Grounding DINO 零样本检测：用 labels.yaml 的中文/英文病名作文本提示。"""

    MODEL_NAME = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, ckpt_dir, sidecar, device="cpu", prompts: Optional[Sequence[str]] = None) -> None:
        super().__init__(ckpt_dir, sidecar, device)
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Grounding DINO 需要 transformers>=4.38 与 torch。\n"
                "  pip install transformers torch\n"
                f"（原始错误：{exc}）"
            ) from exc
        self.torch = torch
        name = str(sidecar.get("meta", {}).get("backbone") or sidecar.get("meta", {}).get("source_model") or self.MODEL_NAME)
        lang = str(sidecar.get("meta", {}).get("prompt_lang") or "en")
        self.text_labels = [p.lower() for p in (prompts or self.label_book.openvocab_prompts(lang))]
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(name).to(self.device).eval()
        self.box_threshold = float(sidecar.get("meta", {}).get("box_threshold", self.conf_threshold))
        self.text_threshold = float(sidecar.get("meta", {}).get("text_threshold", 0.25))
        self._remind_keys()

    def _infer_one(self, image_bgr) -> List[Dict[str, Any]]:
        pil = _bgr_to_pil(image_bgr)
        text = ". ".join(self.text_labels) + "."
        inputs = self.processor(images=pil, text=text, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with self.torch.no_grad():
            outs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outs,
            inputs.get("input_ids"),
            box_threshold=min(self.box_threshold, 0.1),
            text_threshold=self.text_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]
        labels = res.get("labels", [])
        name2id = self.label_book.name_to_id_map()
        out: List[Dict[str, Any]] = []
        for box, score, label in zip(
            res["boxes"].detach().cpu().numpy(),
            res["scores"].detach().cpu().numpy(),
            labels,
        ):
            if float(score) < self.box_threshold:
                continue
            key = str(label).strip().lower().replace(" ", "_")
            cid = name2id.get(key)
            if cid is None:
                key2 = str(label).strip().lower()
                cid = next(
                    (i for n, i in name2id.items() if n.replace("_", " ") == key2),
                    self.label_book.healthy_id(),
                )
            out.append(
                {
                    "class_id": int(cid),
                    "confidence": float(round(float(score), 6)),
                    "bbox": [float(round(float(v), 2)) for v in box[:4]],
                }
            )
        out.sort(key=lambda x: -x["confidence"])
        return out


# ---------------------------------------------------------------------------
# 状态字典加载工具
# ---------------------------------------------------------------------------
def _load_state_dict(path: Path) -> Dict[str, Any]:
    """读 .pt：兼容 {'model'|'state_dict'|'model_state_dict': ...} 与裸 state_dict。"""
    import torch

    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "model_state_dict", "weights"):
            inner = obj.get(key)
            if isinstance(inner, dict) and inner and all(isinstance(k, str) for k in inner.keys()):
                return {str(k).replace("module.", "", 1): v for k, v in inner.items()}
        if obj and all(isinstance(k, str) for k in obj.keys()):
            return {str(k).replace("module.", "", 1): v for k, v in obj.items()}
    raise RuntimeError(f"{path} 不是可识别的 state_dict（顶层类型 {type(obj)}）")


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------
def build_predictor(ckpt_dir: str, device: str = "auto") -> BasePredictor:
    """构造推理器（推理侧统一入口）。

    参数:
        ckpt_dir: ``models/<name>`` 目录，需含 ``model.onnx.json``（sidecar）与 ``best.pt``
        device  : 'auto' | 'cpu' | 'cuda' | 'cuda:0'
    返回:
        具有 ``predict(images: list[np.ndarray]) -> list[dict]`` 的对象
    异常:
        FileNotFoundError: 目录或 sidecar / 权重缺失
        RuntimeError     : 对应后端依赖缺失（中文提示）或权重无法载入
    """
    from src.train.common import resolve_device  # 局部 import 避免循环

    cdir = Path(ckpt_dir)
    if not cdir.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{cdir}")
    sidecar_path = cdir / SIDE_CAR_FILENAME
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"缺少 sidecar 元数据：{sidecar_path}\n"
            "请先运行训练脚本或 `python -m scripts.export_onnx --ckpt-dir <dir> --sidecar-only` 生成。"
        )
    sidecar = load_sidecar(sidecar_path)
    dev = resolve_device(device)
    task = sidecar.get("task", "detect")
    source = str(sidecar.get("meta", {}).get("source", "")).lower()

    if task == "classify":
        return DinovClassifierPredictor(cdir, sidecar, dev)
    if "grounding" in source:
        return GroundingDinoPredictor(cdir, sidecar, dev)
    if "ultralytics" in source or "yolo" in source:
        return UltralyticsPredictor(cdir, sidecar, dev)
    return RTDetrTorchPredictor(cdir, sidecar, dev)
