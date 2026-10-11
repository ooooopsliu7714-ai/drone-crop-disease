"""免模型的后端：基于植被指数 + 颜色统计的「绿度/黄化/褐变」规则判别。

它的作用不是替代深度学习，而是：
  1. 让 `POST /algo/predict` 现在就能跑通、让 Java 与小程序同学不被算法进度阻塞；
  2. 作为答辩时的 baseline（对照实验里"传统方法 vs 你的 RT-DETR 流水线"）；
  3. 训练好的模型不可用（显存/权重缺失）时的自动降级后端。

方法（全部是真实可解释的经典做法，不是瞎猜）：
  - 过绿指数 ExG = 2G - R - B 分割植被，Otsu 自适应阈值（无人机航拍的主要是植被）；
  - 在植被掩膜内做 HSV 统计，按黄化(黄)/褐变(褐)/坏死(暗) 的颜色占比判病；
  - 用 Lab 的 a*/b* 均值区分"偏黄（缺氮/黄化型病害）"与"偏褐（坏死型病害）"；
  - 用滑动网格定位出「病害热点区域」并转成框，从而输出 boxes 满足接口契约。

局限（写进答辩的"局限性"里，别当成检测模型卖）：
  - 无法区分具体病种，只能区分"黄化型 / 褐变型"两大表型族；
  - 受光照、土壤背景、生长阶段影响大，必须做白平衡/光照归一化；
  - 单株级病斑定位能力弱，这也是必须上 RT-DETR + SAM 2.1 的原因。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.labels import LABELS


@dataclass
class RawBox:
    """后端统一输出：像素 xyxy 框 + 类别 + 置信度 + 严重度。"""

    xyxy: tuple[float, float, float, float]
    class_id: int
    confidence: float
    severity: str

    def as_dict(self, image_w: int, image_h: int) -> dict[str, Any]:
        info = LABELS.by_id(self.class_id)
        x1, y1, x2, y2 = self.xyxy
        return {
            "disease_name": info["disease_name"],
            "disease_name_cn": info["disease_name_cn"],
            "confidence": float(self.confidence),
            "severity": self.severity,
            "class_id": int(self.class_id),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
        }


class HeuristicBackend:
    """规则基线后端。输入 BGR uint8 图像，输出 RawBox 列表。"""

    name = "heuristic-cv"
    model_version = "heuristic-v1"

    # 症状族 -> 优先选用的代表类别（黄化型 / 褐变型各选一个最典型病）
    # 说明：这里刻意只做"症状族"判别；换成真实病种必须靠 train_cls 的 DINOv3 分类头。
    _PREFERRED = {
        "chlorosis": ("wheat_stripe_rust", "rice_blast"),   # 黄化型：条锈病（黄锈色）
        "necrosis": ("rice_blast", "potato_late_blight"),   # 褐变坏死型：稻瘟病
    }

    def __init__(self, grid: int = 4, min_hot_ratio: float = 0.025) -> None:
        """grid: 把图切成 grid×grid 个格子找热点；min_hot_ratio: 格子内病斑占比阈值。"""
        self.grid = int(grid)
        self.min_hot_ratio = float(min_hot_ratio)

    @staticmethod
    def resolve_symptom_class(symptom: str) -> str:
        """把症状族映射到**当前生效标签体系里真实存在**的类别名。

        为什么不能写死：同一个仓库要服务 15 类病害体系和 2 类胁迫体系。
        早期写死 "wheat_stripe_rust"/"rice_blast"，切到 2 类后这两个名字不存在，
        接口就会返回 "unknown"（本机实测踩过）。这里的策略：
            1. 优先用预设代表类，但必须存在于当前类别表；
            2. 否则退到当前体系里第一个 is_disease=True 的类别（2 类体系 -> stressed）；
            3. 再不行退到第一个类别。
        """
        preferred = HeuristicBackend._PREFERRED.get(symptom, ())
        names = set(LABELS.names)
        for cand in preferred:
            if cand in names:
                return cand
        for c in LABELS.classes:
            if c.get("is_disease"):
                return str(c["disease_name"])
        return LABELS.names[0] if LABELS.names else "healthy"

    # ------------------------------------------------------------------
    @staticmethod
    def vegetation_mask(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (植被或病株掩膜 0/255, ExG 浮点图)。

        ExG = 2G - R - B 是经典的过绿指数。这里先用 Otsu 拿一个初始掩膜，
        再并入"偏黄/偏褐但仍是植株"的像素：发病植株会掉出 ExG 阈值，
        如果把它们算作背景，发病率会被系统性低估（本模块早期版本的真实缺陷）。
        土壤/水面因为 G 不是主导通道、饱和度低，仍然被排除。
        """
        b, g, r = cv2.split(bgr.astype(np.float32))
        exg = 2.0 * g - r - b
        exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        b8, g8, r8 = cv2.split(bgr)
        # G 为主导通道（含黄绿、暗绿），且不是极暗/极灰的像素 -> 植株
        plantish = (
            (g8.astype(np.int16) >= b8.astype(np.int16) - 5)
            & (s >= 50)
            & (v >= 40)
        )
        mask = np.where(plantish, 255, mask).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if mask.sum() < 0.02 * mask.size:  # 几乎没有植被（水田/裸土/夜景）→ 退回全图
            mask = np.full_like(mask, 255)
        return mask, exg

    # ------------------------------------------------------------------
    @staticmethod
    def symptom_mask(bgr: np.ndarray, symptom: str) -> np.ndarray:
        """返回某类症状的像素掩膜（0/255），**不**与植被掩膜求交。

        为什么不相交：病斑一旦黄化/褐变，ExG 会掉到阈值以下，被"植被掩膜"排除——
        这正是早期版本把大面积黄化斑判成健康的原因。症状阈值本身已经足够严格，
        因此症状统计直接在整图上做，植被掩膜只用来做"分母"。

        为什么还要加通道关系判据（R>=G、不被 B 主导）：
        实测健康绿色像素 (B,G,R)=(70,150,60) 的 HSV 是 H=63,S=153,V=150——
        它的 H 不落在任何症状区间，但如果只放宽亮度/饱和度阈值，绿色很容易被
        误当成"褐变"。加上 R 与 G 的相对关系后，健康绿植被稳定排除，
        黄化（H 高、R 与 G 接近）和褐变（R 明显高于 G）各自归位。
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        b, g, r = cv2.split(bgr.astype(np.int16))
        # 绿色植被：G 明显高于 R，且 G 不低于 B 太多 -> 排除出褐变/暗斑
        green_like = (g >= r - 10) & (g >= b - 10)

        if symptom == "chlorosis":  # 黄化：黄相 + 有饱和度 + 不太暗
            m = (h >= 15) & (h <= 40) & (s >= 60) & (v >= 90) & (~green_like | (r >= g - 25))
        elif symptom == "necrosis":  # 褐变/坏死：红褐相 + 适中亮度，且红不低于绿
            m = ((h <= 15) | (h >= 160)) & (s >= 50) & (v >= 50) & (v <= 210) & (r >= g - 10)
        else:  # dark：暗斑（黑斑、霉层），要求红光不弱于绿光以排除阴影与健康绿植
            m = (v < 70) & (s >= 40) & (r >= g - 10)
        mask = (m.astype(np.uint8)) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    # ------------------------------------------------------------------
    def symptom_ratios(self, bgr: np.ndarray, veg_mask: np.ndarray | None = None) -> dict[str, float]:
        """症状像素占**全图面积**的比例（发病率口径）。

        为什么用全图做分母：检测模型给的 diseased_ratio 是"框面积/整图面积"，
        规则基线必须用同一口径，两侧数字才能直接比较（对照实验、报表都依赖这一点）。
        如果要看"植株上有多大比例发病"，用 symptom_density 字段（占植株面积）。
        """
        total = max(1, bgr.shape[0] * bgr.shape[1])
        return {
            symptom: float((self.symptom_mask(bgr, symptom) > 0).sum()) / total
            for symptom in ("chlorosis", "necrosis", "dark")
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _confidence_from_ratio(ratio: float, base: float = 0.35, span: float = 0.45) -> float:
        """把症状面积占比映射成置信度（0.35~0.80），刻意不给高分：规则法不配高置信度。

        注意：规则基线**不应该**产出高置信度——它的作用是给出"有/无 + 症状族"的初判，
        真正的病种判断必须来自 DINOv3 分类头 / RT-DETR 检测头（见 docs/algorithm-design.md）。
        """
        r = max(0.0, min(1.0, ratio / 0.15))  # 症状占全图 15% 即视为饱和
        return round(base + span * r, 4)

    # ------------------------------------------------------------------
    def predict(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        """对每张 BGR 图返回一个"检测结果"字典（与 src.infer 的 predictor 契约一致）。"""
        return [self.predict_one(img) for img in images]

    def predict_one(self, bgr: np.ndarray) -> dict[str, Any]:
        h, w = bgr.shape[:2]
        mask, _ = self.vegetation_mask(bgr)
        ratios = self.symptom_ratios(bgr, mask)

        dominant = max(ratios, key=lambda k: ratios[k])
        dominant_ratio = ratios[dominant]  # 占全图面积，与检测框面积口径一致

        # 植株范围内的症状密度：给"这块地发病重不重"的另一种直观口径
        veg_area = max(1, int((mask > 0).sum()))
        dominant_density = float(
            (self.symptom_mask(bgr, dominant) > 0).sum()
        ) / veg_area

        # 症状占比过低 -> 健康
        if dominant_ratio < 0.004:
            info = LABELS.by_name("healthy")
            return {
                "disease_name": info["disease_name"],
                "disease_name_cn": info["disease_name_cn"],
                "confidence": round(0.5 + 0.4 * (1 - min(1.0, dominant_ratio / 0.004)), 4),
                "severity": "healthy",
                "class_id": int(info["class_id"]),
                "bbox": [0.0, 0.0, float(w), float(h)],
                "symptom_density": round(dominant_density, 6),
                # 绝对像素量：切片推理时上层要跨切片求和，才能得到整图口径的发病率
                "symptom_area": int((self.symptom_mask(bgr, dominant) > 0).sum()),
                "image_area": int(h) * int(w),
                "symptom_key": dominant,
            }

        disease_name = self.resolve_symptom_class(dominant)
        info = LABELS.by_name(disease_name)
        severity = LABELS.severity_from_ratio(dominant_ratio)
        confidence = self._confidence_from_ratio(dominant_ratio)

        # 网格滑窗找症状最密集的区域，给出一个"最像病灶"的框
        box = self._hot_region_box(bgr, mask, dominant)
        return {
            "disease_name": info["disease_name"],
            "disease_name_cn": info["disease_name_cn"],
            "confidence": confidence,
            "severity": severity,
            "class_id": int(info["class_id"]),
            "bbox": box,
            "symptom_density": round(dominant_density, 6),
            "symptom_area": int((self.symptom_mask(bgr, dominant) > 0).sum()),
            "image_area": int(h) * int(w),
            "symptom_key": dominant,
        }

    # ------------------------------------------------------------------
    def _hot_region_box(
        self, bgr: np.ndarray, mask: np.ndarray, symptom: str
    ) -> list[float]:
        """网格滑窗找症状最密集的格子，返回其像素 xyxy；全图都低则返回整图。"""
        h, w = bgr.shape[:2]
        hot = self.symptom_mask(bgr, symptom) > 0

        gy, gx = self.grid, self.grid
        cell_h, cell_w = max(1, h // gy), max(1, w // gx)
        best = (0.0, 0, 0)
        for i in range(gy):
            for j in range(gx):
                y0, y1 = i * cell_h, min(h, (i + 1) * cell_h)
                x0, x1 = j * cell_w, min(w, (j + 1) * cell_w)
                cell = hot[y0:y1, x0:x1]
                ratio = float(cell.sum()) / max(1, cell.size)
                if ratio > best[0]:
                    best = (ratio, i, j)

        if best[0] < self.min_hot_ratio:
            return [0.0, 0.0, float(w), float(h)]
        _, i, j = best
        return [
            float(j * cell_w),
            float(i * cell_h),
            float(min(w, (j + 1) * cell_w)),
            float(min(h, (i + 1) * cell_h)),
        ]
