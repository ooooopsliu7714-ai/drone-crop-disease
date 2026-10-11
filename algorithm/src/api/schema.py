"""接口契约（与队友 Apifox 文档逐字对齐）。

这里是整个算法服务的"宪法"：
  - 队友（Java 后端 / 小程序 / Apifox mock）只认这里的字段名；
  - 任何改名都会导致联调失败，因此字段名用 const 固定，禁止随手重命名。

契约要点（详见 configs/labels.yaml 末尾注释）：
  POST /algo/predict
    请求 A: {"image_id": "abc123"}
    请求 B: multipart/form-data  file=<图片>
  响应顶层: code / message / image_id / disease_name / confidence /
            disease_name_cn / severity / boxes / processing_time
  boxes[]:  disease_name / disease_name_cn / confidence / severity /
            class_id / x / y / w / h （归一化）+ x1 / y1 / x2 / y2 （像素）
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["healthy", "light", "moderate", "severe"]

# ---- 契约字段名（唯一真源，改这里等于改契约，需同步 Apifox）----
FIELD_DISEASE_NAME = "disease_name"
FIELD_CONFIDENCE = "confidence"
FIELD_BOXES = "boxes"
FIELD_SEVERITY = "severity"


class BBoxOut(BaseModel):
    """单个识别框。坐标同时给归一化和像素两套，前端画框、后端统计各取所需。"""

    disease_name: str = Field(..., description="病害英文标识，如 rice_blast")
    disease_name_cn: str = Field(..., description="病害中文名，如 稻瘟病")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 0~1")
    severity: Severity = Field(..., description="该框的严重度")
    class_id: int = Field(..., description="模型类别索引")

    # 归一化坐标（相对原图，0~1），小程序按百分比定位
    x: float = Field(..., description="左上角 x（归一化）")
    y: float = Field(..., description="左上角 y（归一化）")
    w: float = Field(..., description="宽（归一化）")
    h: float = Field(..., description="高（归一化）")

    # 像素坐标，后端画热力图 / 二次分割用
    x1: float = Field(..., description="左上角 x（像素）")
    y1: float = Field(..., description="左上角 y（像素）")
    x2: float = Field(..., description="右下角 x（像素）")
    y2: float = Field(..., description="右下角 y（像素）")

    # 兼容别名：部分前端同学早期用了 label / score / bbox
    label: str | None = Field(None, description="兼容字段，等同 disease_name")
    score: float | None = Field(None, description="兼容字段，等同 confidence")
    bbox: list[float] | None = Field(None, description="兼容字段，等同 [x1,y1,x2,y2]")


class Statistics(BaseModel):
    """地块/单图统计，对应项目「病害分布地图与统计报表」。"""

    total_boxes: int = 0
    diseased_ratio: float = Field(0.0, description="发病率：患病面积占比 0~1")
    disease_index: float = Field(0.0, description="病情指数 0~100（国标算法）")
    severity_ratio: dict[str, float] = Field(
        default_factory=dict, description="各严重度占比，如 {'moderate': 0.4}"
    )
    class_ratio: dict[str, float] = Field(
        default_factory=dict, description="各病害类别占比（按 disease_name）"
    )


class PredictResponse(BaseModel):
    """POST /algo/predict 的响应体。字段顺序即前端阅读顺序，不要重排。"""

    code: int = Field(200, description="200 表示成功；非 200 时 message 给出原因")
    message: str = Field("success", description="提示信息")

    image_id: str = Field(..., description="图片标识；上传文件时由服务端生成或回显")
    disease_name: str = Field(..., description="主病害英文标识（取置信度最高的框/分类结果）")
    disease_name_cn: str = Field(..., description="主病害中文名")
    confidence: float = Field(..., ge=0.0, le=1.0, description="主病害置信度")

    severity: Severity = Field("healthy", description="整图/整地块严重度")
    boxes: list[BBoxOut] = Field(default_factory=list, description="识别框列表")
    processing_time: float = Field(0.0, description="服务端处理耗时（秒）")

    # 扩展字段（不影响队友既有解析，纯增量）
    statistics: Statistics | None = Field(None, description="统计报表数据")
    model_version: str = Field("", description="产生该结果的模型标识，便于复现与 A/B")
    backend: str = Field("", description="实际使用的推理后端名")
    advice: dict[str, Any] | None = Field(None, description="防治建议（对应核心功能 5）")
    within_latency_budget: bool = Field(
        True, description="本次推理是否在 1.5s 预算内（答辩性能证据）"
    )


class PredictRequest(BaseModel):
    """POST /algo/predict 的 JSON 请求体（与文件上传二选一）。"""

    image_id: str = Field(..., description="已上传图片的标识")
    # 可选：地块/气象上下文，留给防治建议与报表
    field_id: str | None = Field(None, description="地块编号")
    crop: str | None = Field(None, description="作物名，可提高类别先验准确率")


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_loaded: bool
    version: str
    device: str


def bbox_to_out(
    *,
    disease_name: str,
    disease_name_cn: str,
    confidence: float,
    severity: str,
    class_id: int,
    xyxy: list[float],
    image_w: int,
    image_h: int,
) -> BBoxOut:
    """把内部像素 xyxy 框转成契约要求的 BBoxOut（含归一化与兼容字段）。"""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    # 裁剪到图像范围，避免越界框把前端画飞出画布
    x1 = max(0.0, min(x1, float(image_w)))
    x2 = max(0.0, min(x2, float(image_w)))
    y1 = max(0.0, min(y1, float(image_h)))
    y2 = max(0.0, min(y2, float(image_h)))
    iw = max(1.0, float(image_w))
    ih = max(1.0, float(image_h))
    return BBoxOut(
        disease_name=disease_name,
        disease_name_cn=disease_name_cn,
        confidence=round(float(confidence), 4),
        severity=severity,  # type: ignore[arg-type]
        class_id=int(class_id),
        x=round(x1 / iw, 6),
        y=round(y1 / ih, 6),
        w=round((x2 - x1) / iw, 6),
        h=round((y2 - y1) / ih, 6),
        x1=round(x1, 2),
        y1=round(y1, 2),
        x2=round(x2, 2),
        y2=round(y2, 2),
        label=disease_name,
        score=round(float(confidence), 4),
        bbox=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
    )


def compute_statistics(
    boxes: list[BBoxOut],
    image_w: int,
    image_h: int,
    diseased_ratio_override: float | None = None,
) -> Statistics:
    """由识别框算出发病率 / 病情指数 / 严重度占比。

    病情指数（国标通用形式）：
        DI = Σ(病级 × 该病级代表面积) / (最高病级 × 总面积) × 100
    这里病级用 severity_index（healthy=0, light=1, moderate=2, severe=3）。

    `diseased_ratio_override`：当后端能直接给出"症状像素/整图面积"（规则基线）时用它，
    比"框面积/整图面积"更准——框里混着大量健康组织，会把发病率抬高 3~9 倍。
    传入时病情指数也按同一比例计算，保证同一张图在不同后端/不同切片配置下口径一致。
    """
    from src.labels import LABELS  # 局部 import，避免循环依赖

    total_area = max(1.0, float(image_w) * float(image_h))
    sev_index = {s["key"]: int(s["severity_index"]) for s in LABELS.severity_levels}
    max_level = max(sev_index.values()) if sev_index else 3

    diseased_area = 0.0
    weighted = 0.0
    sev_area: dict[str, float] = {}
    cls_area: dict[str, float] = {}
    for b in boxes:
        area = max(0.0, (b.x2 - b.x1) * (b.y2 - b.y1))
        level = sev_index.get(b.severity, 0)
        if level > 0:
            diseased_area += area
            weighted += level * area
        sev_area[b.severity] = sev_area.get(b.severity, 0.0) + area
        cls_area[b.disease_name] = cls_area.get(b.disease_name, 0.0) + area

    if diseased_ratio_override is not None:
        ratio = max(0.0, min(1.0, float(diseased_ratio_override)))
        # 发病率用"症状像素占比"这个更准的口径；病情指数里的**病级分布**仍来自各框，
        # 于是 DI = 发病率 × 平均病级比例，既准又保留了"病斑有多重"的信息。
        mean_level = (weighted / diseased_area) if diseased_area > 0 else 0.0
        weighted = ratio * total_area * mean_level
        diseased_area = ratio * total_area
    else:
        ratio = min(1.0, diseased_area / total_area)

    return Statistics(
        total_boxes=len(boxes),
        diseased_ratio=round(ratio, 6),
        disease_index=round(100.0 * weighted / (max_level * total_area), 4) if max_level else 0.0,
        severity_ratio={k: round(v / total_area, 6) for k, v in sorted(sev_area.items())},
        class_ratio={k: round(v / total_area, 6) for k, v in sorted(cls_area.items())},
    )


def top_disease(boxes: list[BBoxOut]) -> tuple[str, str, float, str]:
    """选主病害：**优先取"真病害"类里置信度最高的框**；没有病害框时才判健康。

    返回 (disease_name, disease_name_cn, confidence, severity)。

    为什么不能简单地取"所有框里置信度最高的"：一张田里同时有健康和病株时，
    健康框往往分数更高，直接取最高分会把整块地报成"健康"——
    本机实测：模型明确检出了 20 个框（含受胁迫植株），接口却报 healthy，
    演示时看起来像"算法失效"。训练侧的 `LabelBook.summarize_boxes` 一直是
    "先挑病害框"的逻辑，这里与之对齐。
    """
    from src.labels import LABELS

    if not boxes:
        healthy = LABELS.by_name("healthy")
        return (
            healthy.get("disease_name", "healthy"),
            healthy.get("disease_name_cn", "健康"),
            0.0,
            "healthy",
        )

    def is_disease(b: BBoxOut) -> bool:
        info = LABELS.by_name(b.disease_name)
        # 非 healthy 类一律视为病害；healthy 类看 is_disease 标记
        return bool(info.get("is_disease", b.disease_name != "healthy"))

    disease_boxes = [b for b in boxes if is_disease(b)]
    pool = disease_boxes or boxes
    best = max(pool, key=lambda b: b.confidence)
    return best.disease_name, best.disease_name_cn, best.confidence, best.severity


def as_dict(model: BaseModel) -> dict[str, Any]:
    """pydantic v2 的 model_dump 包装，避免各处写法不一致。"""
    return model.model_dump(exclude_none=False)  # type: ignore[attr-defined]
