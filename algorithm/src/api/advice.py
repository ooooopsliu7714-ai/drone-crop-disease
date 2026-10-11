"""防治建议生成（对应项目核心功能 5「防治建议与处方图」）。

**重要合规声明**：这里生成的是"农事建议框架"，不含具体农药品种与剂量。
真实用药必须由当地植保站规范/推荐目录决定（项目要求原文就是"不要乱推荐"）。
接口只输出：
  - 防治优先级（立即防治 / 3 天内 / 观察 / 无需处理）
  - 农事操作建议（排水、通风、控氮、清除病株、分区作业……）
  - 需要人工确认的条目（进专家复核队列）
具体药剂字段留了 `--drug-catalog` 的挂载点，等植保站给的 JSON 目录接进来再填。
"""
from __future__ import annotations

from typing import Any

from src.labels import LABELS

# 防治优先级：按严重度 + 病害类型给出
PRIORITY_BY_SEVERITY: dict[str, dict[str, str]] = {
    "healthy": {"key": "none", "text": "无需药剂处理，保持常规巡田"},
    "light": {"key": "observe", "text": "暂不施药，加密巡田频次至每 3 天一次"},
    "moderate": {"key": "within_3_days", "text": "建议 3 天内防治，优先处理发病中心"},
    "severe": {"key": "immediate", "text": "建议立即防治，并在 5~7 天后复查"},
}

# 按作物给通用农事措施（不含药剂）：这些是通行栽培管理建议，可直接用
CROP_PRACTICES: dict[str, list[str]] = {
    "rice": [
        "适度晒田、避免长期深水，降低田间湿度",
        "控制氮肥用量，增施磷钾肥提高抗性",
        "发病中心区域优先处理，避免串灌串排",
    ],
    "wheat": [
        "及时清沟排水，降低田间湿度",
        "避免过密种植，改善通风透光",
        "收获后清除病残体，减少越冬菌源",
    ],
    "corn": [
        "清除下部病叶，改善田间通风",
        "平衡施肥，避免偏施氮肥",
        "秸秆还田前充分腐熟，压低菌源基数",
    ],
    "potato": [
        "起垄栽培、及时培土，防止薯块外露",
        "雨后及时排水，缩短叶面结露时间",
        "发现中心病株立即清除并带出田外",
    ],
    "tomato": [
        "控制棚内湿度，加强通风",
        "及时摘除病叶病果并带出园区",
        "避免大水漫灌，改用滴灌",
    ],
    "grape": ["及时绑蔓摘心，改善架面通风", "清除病果病叶，减少再侵染源"],
    "citrus": [
        "重点防控媒介木虱，发现病株及时标记上报",
        "加强树势管理，避免偏施氮肥",
        "工具专用并消毒，防止人为传播",
    ],
    "generic": ["保持田间通风透光", "清除病残体", "记录发病区域，下次巡田重点复查"],
}

# 无人机变量施药提示：处方图对接
VARIABLE_RATE_HINT = (
    "可按本结果生成处方图（网格给药量 = f(严重度)），对接植保无人机变量施药；"
    "建议仅在 moderate/severe 网格施药，healthy 网格关闭喷头以省药。"
)


def build_advice(
    disease_name: str,
    severity: str,
    crop: str | None = None,
    disease_ratio: float = 0.0,
    *,
    confidence: float = 0.0,
    confidence_threshold: float = 0.6,
) -> dict[str, Any]:
    """生成防治建议结构。confidence 低于阈值时强制转人工复核。"""
    info = LABELS.by_name(disease_name)
    crop_key = (crop or info.get("crop") or "generic").lower()
    priority = PRIORITY_BY_SEVERITY.get(severity) or PRIORITY_BY_SEVERITY["observe"]

    practices = list(CROP_PRACTICES.get(crop_key, CROP_PRACTICES["generic"]))
    if severity in ("moderate", "severe") and disease_ratio > 0:
        practices.insert(0, f"发病面积约占 {disease_ratio * 100:.1f}%，建议先处理发病中心再全田评估")

    need_review = confidence < confidence_threshold
    return {
        "disease_name": disease_name,
        "disease_name_cn": info["disease_name_cn"],
        "severity": severity,
        "priority": priority["key"],
        "priority_text": priority["text"],
        "practices": practices,
        "variable_rate_hint": VARIABLE_RATE_HINT if severity in ("moderate", "severe") else "",
        # 合规：药剂推荐必须由植保站目录提供，默认留空并标记需人工确认
        "chemical_plan": [],
        "chemical_plan_source": "待接入当地植保站推荐目录（configs/drug_catalog.json）",
        "need_expert_review": need_review,
        "review_reason": (
            f"模型置信度 {confidence:.4f} 低于复核阈值 {confidence_threshold}，转农技员复核"
            if need_review
            else ""
        ),
    }
