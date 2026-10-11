"""极简 YAML 配置合并（不引入 hydra / omegaconf）。

干什么：
    - 读取 ``configs/train_*.yaml``，支持点号路径覆盖：``--set train.batch_size=8``；
    - 也支持把命令行已解析的参数（argparse Namespace）覆盖到配置上，忽略 None；
    - 输出仍是普通 dict，训练脚本里 ``cfg["train"]["batch_size"]`` 直接取。

怎么用::

    from src.core.config import load_config, deep_update, apply_overrides
    cfg = load_config("configs/train_cls.yaml",
                      cli_overrides={"train.batch_size": 4, "model.backbone": "facebook/dinov2-base"})
    cfg["train"]["batch_size"]     # 4

依赖：PyYAML（缺失时中文提示）+ 标准库。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

__all__ = ["load_config", "deep_update", "apply_overrides", "dump_config", "ConfigError"]


class ConfigError(RuntimeError):
    """配置相关错误（缺包 / 结构非法 / 覆盖路径不存在）。"""


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - 环境相关
        raise ConfigError(
            "缺少 PyYAML，无法读取 YAML 配置。请先安装：pip install pyyaml\n"
            f"（原始错误：{exc}）"
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射结构：{path}")
    return data


def deep_update(base: MutableMapping[str, Any], over: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """递归合并：``over`` 中的叶子值覆盖 ``base``，两边都是 dict 时继续下钻。"""
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = copy.deepcopy(v)
    return base


def _set_by_path(cfg: MutableMapping[str, Any], dotted: str, value: Any) -> None:
    """按 ``a.b.c`` 设置配置项；中间层不存在时自动创建 dict。"""
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise ConfigError(f"非法的覆盖路径：{dotted!r}")
    node: MutableMapping[str, Any] = cfg
    for p in parts[:-1]:
        nxt = node.get(p)
        if nxt is None:
            nxt = {}
            node[p] = nxt
        if not isinstance(nxt, MutableMapping):
            raise ConfigError(f"覆盖路径 {dotted!r} 的中间项 {p!r} 不是映射结构")
        node = nxt
    node[parts[-1]] = value


def apply_overrides(cfg: MutableMapping[str, Any], overrides: Mapping[str, Any] | None) -> MutableMapping[str, Any]:
    """把 ``{"a.b": 1}`` 形式的覆盖应用到 cfg（原地修改并返回）。"""
    if not overrides:
        return cfg
    for dotted, value in overrides.items():
        if value is None:
            continue
        _set_by_path(cfg, dotted, value)
    return cfg


def load_config(
    config_path: str | Path | None,
    cli_overrides: Mapping[str, Any] | None = None,
    extra_sets: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """读取配置并应用覆盖。

    参数:
        config_path : YAML 路径；为 None 时只用内置默认 + cli_overrides
        cli_overrides: 形如 ``{"train.batch_size": 8}``
        extra_sets  : 形如 ``["train.lr=1e-4", "model.backbone=facebook/dinov2-base"]``
    返回:
        dict 配置
    """
    cfg: Dict[str, Any] = {}
    if config_path:
        p = Path(config_path)
        if not p.is_absolute():
            # 相对路径优先按仓库根解析，其次按当前工作目录
            from .paths import PROJECT_ROOT

            cand = PROJECT_ROOT / p
            p = cand if cand.exists() else p
        if not p.exists():
            raise ConfigError(f"配置文件不存在：{p}")
        cfg = _read_yaml(p)
    apply_overrides(cfg, cli_overrides)

    sets = dict(cli_overrides or {})
    for item in extra_sets or []:
        if "=" not in item:
            raise ConfigError(f"--set 参数必须形如 key=value，收到：{item!r}")
        k, v = item.split("=", 1)
        sets[k.strip()] = _coerce_scalar(v.strip())
    apply_overrides(cfg, sets)
    return cfg


def _coerce_scalar(text: str) -> Any:
    """把 ``--set`` 的字符串值转成 bool/int/float/list/str。"""
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", "~"):
        return None
    if "," in text:
        return [_coerce_scalar(x.strip()) for x in text.split(",") if x.strip() != ""]
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def dump_config(cfg: Mapping[str, Any], title: str = "生效配置") -> str:
    """把配置渲染成缩进文本，供日志打印（避免依赖 yaml）。"""
    lines = [f"===== {title} ====="]

    def _walk(node: Any, prefix: str = "") -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                _walk(v, f"{prefix}{k}.")
        else:
            lines.append(f"  {prefix[:-1]:<44s} = {node!r}")

    _walk(cfg)
    return "\n".join(lines)
