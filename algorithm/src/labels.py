"""标签体系加载器：读取 configs/labels.yaml，作为全项目唯一真源。

用法：
    from src.labels import LABELS
    LABELS.by_id(1)            # -> {"class_id":1,"disease_name":"rice_blast",...}
    LABELS.severity_levels     # -> ["healthy","light","moderate","severe"]
    LABELS.severity_from_ratio(0.32)   # -> "moderate"

注意：disease_name 是接口字段（必须与 Apifox 契约一致），
      disease_name_cn 只用于展示，不要传给后端做逻辑判断。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 pyyaml：pip install pyyaml") from exc

# 项目根目录（本文件在 <root>/src/labels.py）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "configs" / "labels.yaml"


def _load_dotenv() -> list[str]:
    """把项目根 ``.env`` 载入环境变量（**不覆盖**已有值）。

    为什么放在这里：`.env` 里的 `ALGO_LABELS_FILE` 决定用哪套标签体系。
    重启电脑/新开窗口后环境变量会丢，会导致标签体系回退到默认值而产生静默错标
    （本机实测：2 类胁迫数据集被当成 15 类病害表，class 1 被学成"稻瘟病"）。
    训练侧（``src/core/paths.py``）与本文件做同样的加载，保证两侧口径一致。
    """
    p = PROJECT_ROOT / ".env"
    applied: list[str] = []
    if not p.is_file():
        return applied
    try:
        lines = p.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return applied
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not k or os.environ.get(k):
            continue
        os.environ[k] = v
        applied.append(k)
    return applied


#: 导入本模块即自动加载 .env（推理侧所有入口都会经过这里）
DOTENV_APPLIED: list[str] = _load_dotenv()


# 严重度阈值：病斑占**整幅图/整块地**面积的比例 -> 严重度分级
#
# 阈值来源与依据（写清楚便于答辩时解释，也便于按当地标准调整）：
#   - 无人机航拍是"地块级"视角，与植保站"单株病叶率"不是一个口径，不能直接套用国标病级；
#   - 本表按「病斑面积占比」分档：<2% 轻度、2%~10% 中度、>10% 重度。
#     参考：王振等《小麦条锈病遥感监测》及农田病害遥感分级实践中，
#     冠层病斑覆盖率约 5%~10% 是产量损失开始显著的区间，因此把 10% 作为重度下界。
#   - 若评审/植保站要求按国标病级（0~9 级）判定，请在 configs 中改为按"病叶率"标定后替换。
# ⚠️ 这套阈值必须用你自己的标注数据做一次标定（见 scripts/eval.py 输出的分级混淆矩阵），
#    否则分级准确率会成为答辩时被追问的点。
SEVERITY_RATIO_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.02, "light"),       # 病斑面积 < 2%：轻度
    (0.10, "moderate"),    # 2% ~ 10%：中度
    (1.001, "severe"),     # > 10%：重度
)


@dataclass
class Labels:
    """标签体系的内存表示。"""

    version: int = 1
    severity_levels: list[dict[str, Any]] = field(default_factory=list)
    classes: list[dict[str, Any]] = field(default_factory=list)
    _by_id: dict[int, dict[str, Any]] = field(default_factory=dict, repr=False)
    _by_name: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {int(c["class_id"]): c for c in self.classes}
        self._by_name = {str(c["disease_name"]): c for c in self.classes}

    # ---------------- 查询 ----------------
    def by_id(self, class_id: int) -> dict[str, Any]:
        """按内部整数索引取类别；未知 id 返回 unknown 占位，避免接口 500。"""
        return self._by_id.get(int(class_id), self._unknown(class_id))

    def by_name(self, disease_name: str) -> dict[str, Any]:
        """按 disease_name 取类别（接口字段）。"""
        return self._by_name.get(str(disease_name), self._unknown(-1, str(disease_name)))

    @staticmethod
    def _unknown(class_id: int, disease_name: str = "unknown") -> dict[str, Any]:
        return {
            "class_id": int(class_id),
            "disease_name": disease_name,
            "disease_name_cn": "未知",
            "crop": "unknown",
            "is_disease": False,
        }

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def names(self) -> list[str]:
        """按 class_id 排序的 disease_name 列表（ONNX 输出索引对齐用）。"""
        return [c["disease_name"] for c in sorted(self.classes, key=lambda c: c["class_id"])]

    @property
    def severity_keys(self) -> list[str]:
        return [s["key"] for s in self.severity_levels] or [
            "healthy", "light", "moderate", "severe",
        ]

    def is_disease_class(self, class_id: int) -> bool:
        return bool(self.by_id(class_id).get("is_disease", False))

    # ---------------- 严重度推断 ----------------
    def severity_from_ratio(self, diseased_ratio: float) -> str:
        """按「病斑面积 / 总面积」估算严重度。

        用于：检测模型只给框、不给病级时（三级流水线里第二级的输出）。
        第三级用 SAM 2.1 分割得到的真实病斑面积，也是走这个函数换算病级。
        """
        if diseased_ratio is None or diseased_ratio <= 0:
            return "healthy"
        for threshold, key in SEVERITY_RATIO_THRESHOLDS:
            if diseased_ratio < threshold:
                return key
        return "severe"

    def to_sidecar_classes(self) -> list[dict[str, Any]]:
        """导出 ONNX sidecar 里用的精简类别表（与训练侧契约一致）。"""
        return [
            {
                "class_id": c["class_id"],
                "disease_name": c["disease_name"],
                "disease_name_cn": c["disease_name_cn"],
            }
            for c in sorted(self.classes, key=lambda c: c["class_id"])
        ]


def load_labels(path: str | Path | None = None) -> Labels:
    """读取标签体系 yaml。

    解析顺序（与训练侧 ``src/core/labels.py`` 保持一致）：
        1. 显式传入的 path
        2. 环境变量 ``ALGO_LABELS_FILE``（相对路径按项目根解析）
        3. 默认 ``configs/labels.yaml``

    为什么要认环境变量：同一个仓库要同时服务 15 类病害模型和 2 类胁迫模型，
    靠这一个变量切换，不用改代码、不用改接口。
    """
    if path:
        p = Path(path)
    else:
        env = os.environ.get(LABELS_FILE_ENV, "").strip()
        if env:
            p = Path(env)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
        else:
            p = DEFAULT_LABELS_PATH
    if not p.exists():
        raise SystemExit(
            f"找不到标签文件：{p}\n"
            f"（当前 ALGO_LABELS_FILE={os.environ.get(LABELS_FILE_ENV, '')!r}）"
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Labels(
        version=int(raw.get("version", 1)),
        severity_levels=list(raw.get("severity_levels", [])),
        classes=list(raw.get("classes", [])),
    )


#: 环境变量名：指定本次进程使用的标签文件
LABELS_FILE_ENV = "ALGO_LABELS_FILE"

_ACTIVE_CACHE: dict[str, tuple[float, Labels]] = {}


def active_labels() -> Labels:
    """返回**当前**生效的标签体系（跟随 ALGO_LABELS_FILE 变化，按文件 mtime 缓存）。

    为什么不是 import 时定死的单例：本机实测发现，推理侧写死单例、训练侧动态解析，
    两边口径会漂移——2 类模型训练完，推理侧仍按 15 类解释 class_id，
    把 "stressed" 错标成"稻瘟病"。这类静默错标比报错危险得多，所以统一成运行时解析。
    """
    env = os.environ.get(LABELS_FILE_ENV, "").strip()
    p = (PROJECT_ROOT / env) if env and not Path(env).is_absolute() else (Path(env) if env else DEFAULT_LABELS_PATH)
    key = str(p)
    try:
        mtime = p.stat().st_mtime if p.exists() else -1.0
    except OSError:
        mtime = -1.0
    cached = _ACTIVE_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    labels = load_labels(p)
    _ACTIVE_CACHE[key] = (mtime, labels)
    return labels


class _ActiveLabelsProxy:
    """``LABELS`` 的向后兼容代理：每次取属性都解析当前生效的标签体系。

    这样既不用改几十处 ``LABELS.xxx`` 的旧代码，又不会把类别表写死。
    """

    def __getattr__(self, item: str) -> Any:  # noqa: ANN401
        return getattr(active_labels(), item)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<LABELS -> {active_labels()!r}>"


# 全局入口：绝大多数模块直接 import 它（动态跟随 ALGO_LABELS_FILE）
LABELS: Any = _ActiveLabelsProxy()
