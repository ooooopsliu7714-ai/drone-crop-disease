"""路径解析工具。

干什么：把所有脚本里散落的硬编码路径集中到一处，保证「以仓库根目录为锚点」，
        无论从哪个工作目录调用 ``python -m scripts.xxx`` 都能找到 configs/ data/ models/。

怎么用::

    from src.core.paths import PROJECT_ROOT, CONFIGS_DIR, DATA_DIR, MODELS_DIR, model_dir
    d = model_dir("det_rtdetrv2")     # -> <root>/models/det_rtdetrv2 （自动创建）
    ensure_dir(DATA_DIR / "processed")

依赖：仅标准库。
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "CONFIGS_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "SCRIPTS_DIR",
    "SRC_DIR",
    "LABELS_YAML",
    "ensure_dir",
    "model_dir",
    "default_image_roots",
]


def _resolve_project_root() -> Path:
    """确定仓库根目录。

    优先级：
    1. 环境变量 ``DRONE_CROP_ROOT``（CI 或别的机器上迁移时用）；
    2. 本文件向上两级（src/core/paths.py -> src/core -> src -> 仓库根）。
    """
    env = os.environ.get("DRONE_CROP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


#: 仓库根目录（含 configs/ src/ scripts/ data/ models/）
PROJECT_ROOT: Path = _resolve_project_root()
#: 配置文件目录
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
#: 数据目录
DATA_DIR: Path = PROJECT_ROOT / "data"
#: 权重与导出产物目录
MODELS_DIR: Path = PROJECT_ROOT / "models"
#: 脚本目录
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
#: 源码目录
SRC_DIR: Path = PROJECT_ROOT / "src"

#: 标签体系唯一真源
LABELS_YAML: Path = CONFIGS_DIR / "labels.yaml"


def load_dotenv(path: Path | None = None, *, override: bool = False) -> list[str]:
    """把项目根目录 ``.env`` 里的键值写进 ``os.environ``，返回生效的键名列表。

    为什么需要它：`ALGO_LABELS_FILE` / `HF_ENDPOINT` / `HF_HOME` 这类变量只在当前
    终端窗口有效，**重启电脑或新开窗口就丢**。本机实测踩过两次坑：
      1. 重启后 `HF_ENDPOINT` 丢失 -> 训练去连被墙的 huggingface.co，卡在 10060 重试；
      2. 重启后 `ALGO_LABELS_FILE` 丢失 -> 标签体系从 2 类回退成 15 类，
         模型把 class 1（受胁迫）学成"稻瘟病"——静默错标，比报错更难发现。

    默认**不覆盖**已存在的变量，方便命令行临时覆盖。
    """
    p = path or (PROJECT_ROOT / ".env")
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
        if not k:
            continue
        if not override and os.environ.get(k):
            continue
        os.environ[k] = v
        applied.append(k)
    return applied


#: 导入本模块即自动加载 .env（所有训练/数据脚本都会经过这里）
DOTENV_APPLIED: list[str] = load_dotenv()


def ensure_dir(path: str | os.PathLike) -> Path:
    """创建目录（含父目录）并返回 Path，已存在则直接返回。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_dir(name: str, create: bool = True) -> Path:
    """返回 ``models/<name>/`` 目录；``create=True`` 时自动创建。"""
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"非法的模型名 name={name!r}（不得为空、不得包含路径分隔符）")
    p = MODELS_DIR / name
    if create:
        ensure_dir(p)
    return p


def default_image_roots(json_path: str | os.PathLike) -> list:
    """给 COCO json 推断一批「图片可能在哪」的候选根目录。

    典型情况：``prepare_data`` 只写标注（不复制图片），返回的 json 是
    ``data/processed/train/annotations.json``，而图片仍在 ``data/raw`` 或 ``data/samples``，
    这时需要自动探测根目录，否则训练脚本会报「图片全部找不到」。

    返回（去重、仅保留存在的目录）：
        ``[json 所在目录, data/raw, data/samples, data/images, <json目录>/images, ...]``
    """
    p = Path(json_path)
    base = p.parent if p.is_file() else p
    cands = [
        base,
        DATA_DIR / "raw",
        DATA_DIR / "samples",
        DATA_DIR / "images",
        base / "images",
        base.parent / "raw",
        base.parent / "samples",
        base.parent / "images",
    ]
    out: list = []
    seen = set()
    for c in cands:
        try:
            key = str(c.resolve())
        except OSError:
            continue
        if key in seen or not c.is_dir():
            continue
        seen.add(key)
        out.append(c)
    return out
