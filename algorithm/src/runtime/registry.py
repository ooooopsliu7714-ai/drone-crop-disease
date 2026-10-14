"""模型/后端注册中心：决定这次请求到底用哪个模型跑。

优先级（可用 `ALGO_BACKEND` 环境变量强制指定）：
  1. onnx   —— models/<name>/model.onnx + model.onnx.json（训练侧导出，生产用）
  2. torch  —— 直接调用 src/train/predictors.py::build_predictor（本地调试用，需 torch）
  3. heuristic —— 规则基线，永不失败（保证接口一直可用，答辩现场也不会白屏）

设计原则：**任何模型加载失败都不能让 /algo/predict 返回 5xx**，只降级并记录原因。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.infer.predictor import HeuristicDetector, Predictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"


@dataclass
class BackendStatus:
    """当前后端状态，健康检查接口会返回它。"""

    backend: str = "heuristic-cv"
    model_version: str = "heuristic-v1"
    loaded: bool = False
    device: str = "cpu"
    notes: list[str] = field(default_factory=list)


def _find_latest_onnx() -> Path | None:
    """在 models/ 下选一个 .onnx：**优先检测**，同任务内取最近修改的。

    为什么不能只按 mtime（本机实测抓到的部署级 bug）：原实现是
    `max(candidates, key=mtime)`。当 models/ 下同时有分类和检测两个 ONNX 产物时，
    **谁的导出时间晚就用谁**。实测重导出一次分类模型后，ONNX 后端就悄悄加载了
    `cls_dinov2` 而不是 `det_rtdetrv2` —— 分类模型的输出是"1 个覆盖整图的框 + 类别置信度"，
    于是接口对每张图只返回一个大框，和 torch 后端（正确的多株框）结论相反。
    `/health` 只报 impl=onnxruntime，不报模型名，排查时极难定位。
    现在与 `_find_latest_ckpt_dir` 保持同一策略：task=detect 优先，再比 mtime。
    """
    if not MODELS_DIR.exists():
        return None
    candidates = [
        p for p in MODELS_DIR.rglob("*.onnx")
        if p.is_file() and not p.parent.name.startswith(("_", "."))
    ]
    if not candidates:
        return None

    def rank(p: Path) -> tuple[int, float]:
        task = _ckpt_task(p.parent)
        # 检测优先（0），其余次之（1）；同组内按修改时间倒序
        return (0 if task == "detect" else 1, -p.stat().st_mtime)

    return sorted(candidates, key=rank)[0]


def _ckpt_task(d: Path) -> str:
    """读产物目录里的 task（detect / classify），读不到返回空串。"""
    for fn in ("model.onnx.json", "checkpoint.json"):
        p = d / fn
        if not p.is_file():
            continue
        try:
            meta = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        task = meta.get("task")
        if task:
            return str(task)
    return ""


def _find_latest_ckpt_dir() -> Path | None:
    """在 models/ 下挑一个**可用的**模型目录。

    排序规则（优先级从高到低）：
        0. ``ALGO_MODEL_DIR`` 显式指定的目录 —— 用于 A/B 对比与联调切换
        1. **检测模型**（task=detect）        —— 主线默认产物
        2. 分类模型（task=classify）          —— 一级初筛
        同档内：完整（sidecar+权重）> 只有 sidecar > 只有权重；
        名字以 ``_``/``.`` 开头的临时目录排到最后；同档按修改时间倒序。

    两个都来自实测踩坑：
      - 只按修改时间取最新 -> 训练自测留下的 `_tmp_*` 半成品把真实权重挤掉，
        服务静默退回规则基线；
      - 不区分任务 -> 分类器后训练出来就自动顶掉检测模型，
        接口不再输出病斑框（分类是流水线第一级，检测才是主产物）。
    """
    explicit = os.environ.get("ALGO_MODEL_DIR", "").strip()
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_dir():
            return p

    if not MODELS_DIR.exists():
        return None

    dirs: set[Path] = set()
    for marker in ("best.pt", "checkpoint.json", "model.onnx.json", "*.pt"):
        dirs.update(p.parent for p in MODELS_DIR.rglob(marker) if p.is_file())
    if not dirs:
        return None

    def rank(d: Path) -> tuple[int, int, int, float]:
        task = _ckpt_task(d)
        task_rank = 0 if task == "detect" else (1 if task == "classify" else 2)
        has_sidecar = (d / "model.onnx.json").is_file()
        has_weight = (d / "best.pt").is_file() or any(d.glob("*.pt"))
        if has_sidecar and has_weight:
            tier = 0
        elif has_sidecar:
            tier = 1
        else:
            tier = 2
        temporary = 1 if d.name.startswith(("_", ".")) else 0
        return (temporary, task_rank, tier, -d.stat().st_mtime)

    return sorted(dirs, key=rank)[0]


def _detect_device() -> str:
    """有 CUDA 就用 cuda，否则 cpu（当前环境 torch 是 CPU 版）。"""
    try:
        import torch  # 惰性 import

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_predictor(
    backend: str | None = None,
    onnx_path: str | Path | None = None,
    device: str | None = None,
    slice_threshold: int | None = None,
) -> tuple[Predictor, BackendStatus]:
    """构造 Predictor 并返回状态。永不抛异常（最差回到规则基线）。

    切片策略可用环境变量调整（无人机正射图常达 4000px 以上）：
        ALGO_SLICE_THRESHOLD  长边超过它才切片，默认 1600；设 0 表示关闭切片
        ALGO_TILE / ALGO_OVERLAP  切片尺寸与重叠，默认 640 / 128
    """
    want = (backend or os.environ.get("ALGO_BACKEND") or "auto").lower()
    dev = device or _detect_device()
    status = BackendStatus(device=dev)

    if slice_threshold is None:
        try:
            slice_threshold = int(os.environ.get("ALGO_SLICE_THRESHOLD", "1600"))
        except ValueError:
            slice_threshold = 1600
    tile = int(os.environ.get("ALGO_TILE", "640"))
    overlap = int(os.environ.get("ALGO_OVERLAP", "128"))

    def _make(pred_backend: Any) -> Predictor:
        return Predictor(
            pred_backend,
            slice_threshold=slice_threshold or 10**9,  # 0 -> 基本等于关闭切片
            tile=tile,
            overlap=overlap,
            device=dev,
        )

    def _finish(pred: Predictor, note: str, loaded: bool) -> tuple[Predictor, BackendStatus]:
        status.backend = pred.name
        status.model_version = pred.model_version
        status.loaded = loaded
        if note:
            status.notes.append(note)
        return pred, status

    # ---- 1. ONNX ----
    if want in ("auto", "onnx"):
        path = Path(onnx_path) if onnx_path else _find_latest_onnx()
        if path is not None:
            try:
                from src.infer.onnx_backend import OnnxDetector

                det = OnnxDetector(path, device=dev)
                pred = _make(det)
                return _finish(pred, f"已加载 ONNX：{path}", True)
            except Exception as exc:
                status.notes.append(f"ONNX 后端不可用，尝试降级：{exc}")
        else:
            status.notes.append("models/ 下没有找到 .onnx 模型，跳过 ONNX 后端")

    # ---- 2. PyTorch（本地调试）----
    if want in ("auto", "torch"):
        ckpt_dir = _find_latest_ckpt_dir()
        if ckpt_dir is not None:
            try:
                from src.train.predictors import build_predictor as build_torch_predictor

                torch_pred = build_torch_predictor(str(ckpt_dir), device=dev)
                # 把权重目录名写进 model_version：接口返回里能看出是"哪一份权重"出的结果，
                # A/B 对照与复现都依赖这个字段（早期版本恒为 unknown）。
                if not getattr(torch_pred, "model_version", None) or getattr(torch_pred, "model_version") == "unknown":
                    try:
                        torch_pred.model_version = ckpt_dir.name
                    except Exception:
                        pass
                pred = _make(torch_pred)
                return _finish(pred, f"已加载 PyTorch 权重：{ckpt_dir}", True)
            except Exception as exc:
                status.notes.append(f"PyTorch 后端不可用（{ckpt_dir}）：{exc}")
        else:
            status.notes.append("models/ 下没有 model.onnx.json / best.pt / checkpoint.json，跳过 PyTorch 后端")

    # ---- 3. 规则基线 ----
    det = HeuristicDetector()
    pred = _make(det)
    return _finish(pred, "已降级为规则基线（颜色/植被指数），仅供联调，不作为算法成果", False)


# ---------------------------------------------------------------------------
# 预测结果缓存：队友拿到 image_id 后可以再查一次结果（对应小程序"看结果"）
# ---------------------------------------------------------------------------
_RESULT_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LIMIT = 200


def cache_result(image_id: str, payload: dict[str, Any]) -> None:
    if len(_RESULT_CACHE) >= _CACHE_LIMIT:
        # 简单 FIFO 淘汰，避免长时间运行内存膨胀
        for key in list(_RESULT_CACHE)[: _CACHE_LIMIT // 4]:
            _RESULT_CACHE.pop(key, None)
    _RESULT_CACHE[image_id] = payload


def get_cached_result(image_id: str) -> dict[str, Any] | None:
    return _RESULT_CACHE.get(image_id)
