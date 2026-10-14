"""标签体系加载与校验（唯一真源：``configs/labels.yaml``）。

干什么：
    - 读取 ``configs/labels.yaml``，把 class_id / disease_name / disease_name_cn 三者绑定成
      ``LabelBook``，全项目（数据准备、训练、导出 sidecar、评估报告）共用同一份映射；
    - 提供 severity 白名单（healthy|light|moderate|severe）与按「框面积占比」估算严重度的规则，
      保证训练侧导出的产物与 ``src/infer/`` 的接口契约一致。

怎么用::

    from src.core.labels import load_labels
    lb = load_labels()                      # 默认读 configs/labels.yaml
    lb.num_classes                          # 15
    lb.name_of(1)                           # 'rice_blast'
    lb.cn_of(1)                             # '稻瘟病'
    lb.sidecar_classes()                    # [{'class_id':0,'disease_name':...}, ...]
    lb.severity_from_area_ratio(0.12)       # 'light'

换用别的标签体系（例如外部数据集只有 2 类）**不用改代码**，设环境变量即可::

    $env:ALGO_LABELS_FILE = "configs/labels_stress2.yaml"     # PowerShell
    export ALGO_LABELS_FILE=configs/labels_stress2.yaml       # bash
    python -m scripts.train_det --data-root data/processed ...

这样训练、导出 sidecar、评估、现场推理会全部切到同一套类别，避免
"训练按 2 类、推理按 15 类"这种会把 class_id 解释错的静默错误。

依赖：PyYAML（缺失时给出中文提示并抛 RuntimeError，不打印 ImportError 栈）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .paths import LABELS_YAML

#: 环境变量名：指定本次进程使用的标签文件（相对路径按仓库根解析）
LABELS_FILE_ENV = "ALGO_LABELS_FILE"

__all__ = [
    "SEVERITIES",
    "SEVERITY_CN",
    "SEVERITY_INDEX",
    "LabelBook",
    "load_labels",
    "severity_from_area_ratio",
]

#: 接口字段 severity 的合法取值（顺序即严重程度递增）
SEVERITIES: tuple[str, ...] = ("healthy", "light", "moderate", "severe")
SEVERITY_CN: Dict[str, str] = {
    "healthy": "健康",
    "light": "轻度",
    "moderate": "中度",
    "severe": "重度",
}
SEVERITY_INDEX: Dict[str, int] = {
    "healthy": 0,
    "light": 1,
    "moderate": 2,
    "severe": 3,
}

#: 患病面积占全图比例 -> 严重度（检测模型无病级标注时的兜底估算规则）
#: 取值 (上限比例, severity)，按顺序命中第一个 ratio < 上限 的档位。
#: 阈值与 ``src/labels.py::SEVERITY_RATIO_THRESHOLDS``（推理侧使用的分档）刻意保持一致：
#: 2% / 10% 两档，避免训练侧与推理侧对同一张图给出不同的 severity。
DEFAULT_AREA_RATIO_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.02, "light"),
    (0.10, "moderate"),
    (1.001, "severe"),
)


def severity_from_area_ratio(
    ratio: float,
    thresholds: Sequence[tuple[float, str]] = DEFAULT_AREA_RATIO_THRESHOLDS,
) -> str:
    """按「患病框面积 / 整图面积」估算 severity。

    规则（与推理侧 ``src/labels.py`` 的分档一致，避免两侧结论打架）：
        ratio <= 0        -> healthy
        0 < ratio < 0.02  -> light
        0.02 <= ratio < 0.10 -> moderate
        ratio >= 0.10     -> severe

    阈值来源见 ``src/labels.py`` 的注释：无人机航拍是地块级视角，不能直接套国标病叶率，
    10% 作为重度下界参考了农田病害遥感分级实践；**上线前应用自有标注数据校准**。
    """
    if ratio is None or float(ratio) <= 0:
        return "healthy"
    r = float(ratio)
    for thr, key in thresholds:
        if r < float(thr):
            return key
    return str(thresholds[-1][1]) if thresholds else "severe"


@dataclass(frozen=True)
class _Cls:
    """单个类别的只读记录。"""

    class_id: int
    disease_name: str
    disease_name_cn: str
    crop: str = "generic"
    is_disease: bool = True


class LabelBook:
    """标签字典：class_id -> (disease_name, disease_name_cn, crop, is_disease)。"""

    def __init__(self, classes: List[_Cls], version: int = 1, source: Path | None = None):
        if not classes:
            raise ValueError("labels.yaml 中 classes 为空，无法构建标签体系")
        self._classes = sorted(classes, key=lambda c: c.class_id)
        self.version = version
        self.source = source
        expect = list(range(len(self._classes)))
        got = [c.class_id for c in self._classes]
        if got != expect:
            raise ValueError(
                "labels.yaml 的 class_id 必须是 0..N-1 的连续整数（推理侧按索引取类别），"
                f"当前为 {got}"
            )
        names = [c.disease_name for c in self._classes]
        if len(set(names)) != len(names):
            raise ValueError(f"labels.yaml 存在重复 disease_name：{names}")

    # ---- 基本信息 -------------------------------------------------------
    @property
    def num_classes(self) -> int:
        """类别总数（不含背景，检测模型的 num_labels 用它）。"""
        return len(self._classes)

    @property
    def classes(self) -> List[_Cls]:
        """全部类别记录（按 class_id 升序）。"""
        return list(self._classes)

    @property
    def names(self) -> List[str]:
        """按 class_id 排序的英文 disease_name 列表（YOLO 的 names 字段用它）。"""
        return [c.disease_name for c in self._classes]

    @property
    def names_cn(self) -> List[str]:
        """按 class_id 排序的中文名列表。"""
        return [c.disease_name_cn for c in self._classes]

    @property
    def disease_class_ids(self) -> List[int]:
        """只包含「真病害」的 class_id（用于发病率/病情指数统计，排除 healthy）。"""
        return [c.class_id for c in self._classes if c.is_disease]

    # ---- 查询 -----------------------------------------------------------
    def get(self, class_id: int) -> _Cls:
        """按 class_id 取类别记录，越界抛 IndexError。"""
        if not 0 <= int(class_id) < self.num_classes:
            raise IndexError(f"class_id={class_id} 超出范围 [0,{self.num_classes - 1}]")
        return self._classes[int(class_id)]

    def name_of(self, class_id: int) -> str:
        """class_id -> 英文 disease_name。"""
        return self.get(class_id).disease_name

    def cn_of(self, class_id: int) -> str:
        """class_id -> 中文 disease_name_cn。"""
        return self.get(class_id).disease_name_cn

    def crop_of(self, class_id: int) -> str:
        """class_id -> 作物（rice/wheat/corn/...）。"""
        return self.get(class_id).crop

    def is_disease(self, class_id: int) -> bool:
        """该类别是否为真病害（healthy 为 False）。"""
        return bool(self.get(class_id).is_disease)

    def id_of_name(self, disease_name: str) -> int:
        """英文名 -> class_id；找不到抛 KeyError。"""
        for c in self._classes:
            if c.disease_name == disease_name:
                return c.class_id
        raise KeyError(f"labels.yaml 中不存在 disease_name={disease_name!r}")

    def has_healthy_class(self) -> bool:
        """标签体系里是否**真的**定义了「健康」类。

        ⚠️ 必须与 `healthy_id()` 区分开：后者在找不到健康类时会**返回 0 兜底**，
        而 0 在 PlantSeg 那套 10 类病种体系里是 `apple_scab` ——
        直接拿它当"无检出"的默认值，就会把"没检出病斑"报成"苹果黑星病"。
        本机实测（训练末尾的自检抓到）：
            num_boxes=0 却输出 disease_name="apple_scab", confidence=0.0
        """
        return any((not c.is_disease) or c.disease_name == "healthy" for c in self._classes)

    def healthy_id(self) -> int:
        """返回「健康」类的 class_id（找不到时返回 0 并符合接口约定）。

        ⚠️ 调用前建议先用 `has_healthy_class()` 判断该体系是否真有健康类；
        否则 0 可能是一个**病种类别**（见该方法的说明）。
        """
        for c in self._classes:
            if not c.is_disease or c.disease_name == "healthy":
                return c.class_id
        return 0

    def name_to_id_map(self) -> Dict[str, int]:
        """英文名 -> class_id 的字典（评估时把预测类别名映射回 id）。"""
        return {c.disease_name: c.class_id for c in self._classes}

    # ---- 导出给 sidecar / 训练配置 --------------------------------------
    def sidecar_classes(self) -> List[Dict[str, Any]]:
        """生成 sidecar json 的 ``classes`` 字段（严格按接口字段名）。"""
        return [
            {
                "class_id": c.class_id,
                "disease_name": c.disease_name,
                "disease_name_cn": c.disease_name_cn,
            }
            for c in self._classes
        ]

    def openvocab_prompts(self, lang: str = "en") -> List[str]:
        """开放词汇模型（Grounding DINO / YOLO-World）用的文本提示列表。

        lang='en' 用英文 disease_name + 空格替换下划线；lang='cn' 用中文名。
        """
        if lang == "cn":
            return [c.disease_name_cn for c in self._classes]
        if lang == "en":
            return [c.disease_name.replace("_", " ") for c in self._classes]
        raise ValueError("lang 只能是 'en' 或 'cn'")

    def summary_lines(self) -> List[str]:
        """人类可读的类别清单（训练脚本启动时打印，便于对齐标注口径）。"""
        lines = [f"标签体系 version={self.version} 共 {self.num_classes} 类："]
        for c in self._classes:
            flag = "病害" if c.is_disease else "健康"
            lines.append(
                f"  class_id={c.class_id:<3d} {c.disease_name:<32s} {c.disease_name_cn:<12s} "
                f"[{c.crop}/{flag}]"
            )
        return lines

    # ---- 由检测框汇总出接口字段 -----------------------------------------
    def summarize_boxes(
        self,
        boxes: Sequence[Dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> Dict[str, Any]:
        """把一组检测框汇总成接口需要的主病害 + 置信度 + 严重度。

        参数:
            boxes: 每个元素形如
                {"class_id": int, "confidence": float, "bbox": [x1,y1,x2,y2]}
            image_width / image_height: 原图像素尺寸（用于算面积占比）
        返回:
            {"disease_name","disease_name_cn","confidence","severity","class_id",
             "area_ratio","disease_area_ratio","num_boxes"}

        规则（与训练侧导出、评估侧一致）：
            - 主病害 = 置信度最高的「真病害」框；若只有 healthy 框或没有框，则取 healthy 类；
            - severity: 无病害框 -> healthy；否则按病害框并集面积占比用
              ``severity_from_area_ratio`` 分档，并用框面积之和做上界截断（近似并集）。
        """
        total = max(1.0, float(image_width) * float(image_height))
        pos = [b for b in boxes if self.is_disease(int(b["class_id"]))]
        healthy_id = self.healthy_id()

        if not pos:
            any_box = max(boxes, key=lambda b: float(b.get("confidence", 0.0))) if boxes else None
            if self.has_healthy_class():
                cid = int(any_box["class_id"]) if any_box is not None else self.healthy_id()
                name, cn = self.name_of(cid), self.cn_of(cid)
            else:
                # 该体系**只标注病斑、没有健康类**（如 PlantSeg 10 类病种体系）。
                # 此时"没有病害框"就等于"没检出病斑"，必须报"健康"，
                # **不能**沿用 healthy_id() 的 0 兜底 —— 那会把结果报成 class 0 的病名
                # （本机实测：num_boxes=0 却输出 apple_scab，训练自检因此判定字段不匹配）。
                name, cn, cid = "healthy", "健康", -1
            conf = float(any_box.get("confidence", 0.0)) if any_box is not None else 0.0
            return {
                "disease_name": name,
                "disease_name_cn": cn,
                "confidence": round(conf, 4),
                "severity": "healthy",
                "class_id": cid,
                "area_ratio": 0.0,
                "disease_area_ratio": 0.0,
                "num_boxes": len(boxes),
            }

        top = max(pos, key=lambda b: float(b.get("confidence", 0.0)))
        area = 0.0
        for b in pos:
            x1, y1, x2, y2 = (float(v) for v in b["bbox"][:4])
            area += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ratio = min(1.0, area / total)
        return {
            "disease_name": self.name_of(int(top["class_id"])),
            "disease_name_cn": self.cn_of(int(top["class_id"])),
            "confidence": round(float(top.get("confidence", 0.0)), 4),
            "severity": severity_from_area_ratio(ratio),
            "class_id": int(top["class_id"]),
            "area_ratio": round(ratio, 6),
            "disease_area_ratio": round(ratio, 6),
            "num_boxes": len(boxes),
        }


def _read_yaml(path: Path) -> Dict[str, Any]:
    """读 YAML；PyYAML 缺失时给中文提示（不抛 ImportError 栈）。"""
    try:
        import yaml  # 局部 import：保证 PyYAML 缺失时也能被 try/except 捕获
    except Exception as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "缺少 PyYAML，无法解析 configs/labels.yaml。请先安装：pip install pyyaml\n"
            f"（原始错误：{exc}）"
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"标签文件格式异常（顶层不是映射）：{path}")
    return data


def load_labels(path: str | Path | None = None) -> LabelBook:
    """加载标签体系。

    参数:
        path: 标签 yaml 路径。默认 ``<repo>/configs/labels.yaml``；
              若设置了环境变量 ``ALGO_LABELS_FILE`` 则优先用它（相对路径按仓库根解析）。
    返回:
        LabelBook
    异常:
        FileNotFoundError: 文件不存在
        RuntimeError      : 缺 PyYAML 或结构非法
        ValueError        : class_id 不连续 / disease_name 重复
    """
    if path is not None:
        p = Path(path)
    else:
        env = os.environ.get(LABELS_FILE_ENV, "").strip()
        if env:
            p = Path(env)
            if not p.is_absolute():
                # 相对路径按仓库根解析，避免"在别的目录执行脚本就找不到标签文件"
                p = LABELS_YAML.parent.parent / p
        else:
            p = LABELS_YAML
    if not p.exists():
        raise FileNotFoundError(
            f"标签文件不存在：{p}\n"
            f"（当前 ALGO_LABELS_FILE={os.environ.get(LABELS_FILE_ENV, '')!r}，"
            f"默认应为 {LABELS_YAML}）"
        )
    data = _read_yaml(p)
    raw_classes = data.get("classes") or []
    classes: List[_Cls] = []
    for i, item in enumerate(raw_classes):
        if not isinstance(item, dict):
            raise RuntimeError(f"labels.yaml classes[{i}] 不是映射结构")
        missing = [k for k in ("class_id", "disease_name") if k not in item]
        if missing:
            raise RuntimeError(f"labels.yaml classes[{i}] 缺少字段 {missing}")
        classes.append(
            _Cls(
                class_id=int(item["class_id"]),
                disease_name=str(item["disease_name"]),
                disease_name_cn=str(item.get("disease_name_cn") or item["disease_name"]),
                crop=str(item.get("crop") or "generic"),
                is_disease=bool(item.get("is_disease", item["disease_name"] != "healthy")),
            )
        )
    return LabelBook(classes, version=int(data.get("version", 1) or 1), source=p)
