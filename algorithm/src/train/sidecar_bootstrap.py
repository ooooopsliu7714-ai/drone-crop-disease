"""体检脚本的前置准备：确保模型目录里有 sidecar（`model.onnx.json`）。

为什么需要：`build_predictor` 依赖 sidecar 里的元数据（task/classes/input 等），
但 sidecar 是**训练结束时**才写的。想在训练中途做体检（框尺寸、置信度分布）
就会报 "缺少 sidecar 元数据"，只能干等训练跑完——本机实测踩过这个坑。

本模块按 `checkpoint.json` 现场生成一份 sidecar（训练结束时会被正式版本覆盖，
所以中途写入是安全的）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["ensure_sidecar"]


def ensure_sidecar(ckpt_dir: str | Path, task: str | None = None) -> tuple[bool, str]:
    """确保 ``<ckpt_dir>/model.onnx.json`` 存在；缺了就按 checkpoint.json 补写。

    返回 ``(是否可用, 说明信息)``。
    """
    d = Path(ckpt_dir)
    sidecar = d / "model.onnx.json"
    if sidecar.is_file():
        return True, "sidecar 已存在"
    if not (d / "best.pt").is_file():
        return False, f"找不到权重：{d / 'best.pt'}"

    try:
        from src.core.labels import load_labels
        from src.train.exporter import save_sidecar

        info: dict = {}
        ck = d / "checkpoint.json"
        if ck.is_file():
            info = json.loads(ck.read_text(encoding="utf-8-sig"))

        # infer_sidecar_from_ckpt 在 scripts/export_onnx.py 里，用 Namespace 传参
        import sys

        root = d.resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts.export_onnx import infer_sidecar_from_ckpt

        ns = argparse.Namespace(
            task=task or info.get("task") or ("classify" if "cls" in d.name else "detect"),
            source=info.get("source"),
            imgsz=info.get("imgsz") or info.get("image_size"),
            note=None,
            mean=None,
            std=None,
            output_kind=None,
            output_names=None,
            device=None,
            backend_impl=None,
        )
        sc = infer_sidecar_from_ckpt(d, load_labels(), ns)
        save_sidecar(sc, model_dir=d)
        return True, f"已按 checkpoint.json 补写 sidecar：{sidecar}"
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"补写 sidecar 失败（{type(exc).__name__}: {exc}）；"
            f"可手动跑：python -m scripts.export_onnx --ckpt-dir {ckpt_dir} --sidecar-only --task detect"
        )
