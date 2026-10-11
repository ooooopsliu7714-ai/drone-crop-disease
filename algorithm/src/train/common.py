"""训练链路通用工具：设备选择、随机种子、日志、JSONL 记录、ONNX 依赖探测。

干什么：
    - ``resolve_device('auto')``：有 CUDA 用 cuda，否则退 CPU；顺便打印 RTX 5070 的算力提示；
    - ``set_seed`` / ``banner`` / ``log``：统一训练日志风格（中文）；
    - ``JsonlLogger``：把每个 epoch 的指标落成 ``models/<name>/train_log.jsonl``，便于事后画曲线；
    - ``check_optional``：集中给出「缺哪个包、装什么」的中文提示。

怎么用::

    from src.train.common import resolve_device, set_seed, JsonlLogger, banner
    device = resolve_device("auto")
    logger = JsonlLogger("models/det_rtdetrv2/train_log.jsonl")

依赖：标准库；torch 为惰性 import。
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "resolve_device",
    "cuda_diagnostics",
    "set_seed",
    "log",
    "banner",
    "JsonlLogger",
    "Timer",
    "optional_dep_hint",
    "torch_dtype_for",
]

#: 缺包时的中文安装提示（惰性 import 分支统一引用）
DEP_HINTS: Dict[str, str] = {
    "torch": "pip install torch --index-url https://download.pytorch.org/whl/cpu  "
             "（RTX 5070 / Blackwell sm_120 必须用 cu128：--index-url https://download.pytorch.org/whl/cu128）",
    "torchvision": "pip install torchvision --index-url https://download.pytorch.org/whl/cpu",
    "transformers": "pip install transformers>=4.48",
    "timm": "pip install timm（DINOv2/v3 的替代 backbone 或 timm 版 RT-DETR 才需要）",
    "albumentations": "pip install albumentations（本项目已内置最小增广，可不装）",
    "ultralytics": "pip install ultralytics（只影响 --backend ultralytics-rtdetr 与 YOLO-World 分支）",
    "onnx": "pip install onnx（导出 ONNX 需要；不装则只输出 sidecar json）",
    "onnxruntime": "pip install onnxruntime（用于校验导出的 ONNX；训练侧不依赖它）",
    "pycocotools": "本项目评估是纯 numpy 实现，不需要 pycocotools，可不装",
    "peft": "pip install peft（只影响 --peft lora 分支；不装则退化为只训分类头）",
    "cv2": "pip install opencv-python（自动标注/可视化需要）",
}


def optional_dep_hint(pkg: str) -> str:
    """返回某个可选依赖的中文安装提示。"""
    return DEP_HINTS.get(pkg, f"pip install {pkg}")


def resolve_device(requested: str = "auto") -> str:
    """解析设备字符串。

    requested:
        'auto' -> cuda 可用则 'cuda'，否则 'cpu'
        'cuda' / 'cuda:1' / 'cpu' / 'mps' 原样校验后返回
    """
    req = (requested or "auto").strip().lower()
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        return "cpu"
    if req == "auto":
        dev = "cuda" if has_cuda else "cpu"
    elif req.startswith("cuda"):
        if not has_cuda:
            print(
                "[警告] 指定了 CUDA 但当前 torch 检测不到可用 GPU（torch 是 CPU 版或驱动/cu128 未就绪），"
                "已自动回退 CPU。RTX 5070 Laptop 是 Blackwell sm_120，必须安装 cu128 版 torch。"
            )
            return "cpu"
        dev = req
    else:
        dev = req
    if dev == "cpu":
        print("[信息] 当前使用 CPU 训练（速度慢，适合小样例验证流程；正式训练请装 cu128 torch）")
    else:
        print(f"[信息] 当前使用 {dev} 训练：{cuda_diagnostics()}")
    return dev


def cuda_diagnostics() -> str:
    """打印友好的一行 GPU 信息（无 GPU 时返回提示字符串）。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return "无可用 CUDA 设备"
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        return f"{name}  sm_{cap[0]}{cap[1]}  显存 {total:.1f} GB  算力能力={cap}"
    except Exception as exc:  # pragma: no cover
        return f"GPU 信息获取失败：{exc}"


def torch_dtype_for(device: str, amp: bool = False):
    """按设备与是否 AMP 选择 dtype（CPU 上 bf16 支持不全，统一 fp32）。"""
    import torch

    if amp and str(device).startswith("cuda"):
        return torch.float16
    if str(device).startswith("cuda") and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int) -> None:
    """固定随机种子（python / numpy / torch），保证划分与训练可复现。"""
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def log(msg: str, level: str = "信息") -> None:
    """统一日志输出：``[信息] ...``（带时间戳，便于和前端联调时对时间）。"""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}][{level}] {msg}", flush=True)


def banner(title: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """打印分节标题 + 关键环境信息。"""
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"Python {sys.version.split()[0]} | {platform.system()} {platform.release()} | PID {os.getpid()}")
    try:
        import torch

        print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | {cuda_diagnostics()}")
    except Exception:
        print("torch 未安装或不可用")
    try:
        import transformers

        print(f"transformers {transformers.__version__}")
    except Exception:
        print("transformers 未安装")
    for k, v in (extra or {}).items():
        print(f"{k}: {v}")
    print("-" * 78)


class JsonlLogger:
    """每个 epoch/step 追加一行 json 的训练日志。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        rec = dict(record)
        rec.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class Timer:
    """上下文管理器：统计耗时（秒）。"""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0
