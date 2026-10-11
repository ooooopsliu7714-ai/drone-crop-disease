"""导出模块：ONNX + sidecar json（推理侧 ``src/infer/`` 直接按固定路径与字段名消费）。

固定产物路径规则（不要改，推理侧按这个约定加载）：
    - 权重：``models/<name>/best.pt``
    - ONNX：``models/<name>/model.onnx``
    - 元数据：``models/<name>/model.onnx.json``

sidecar json 字段（严格）::

    {
      "task": "detect",
      "classes": [{"class_id":0,"disease_name":"healthy","disease_name_cn":"健康"}],
      "input": {"name":"images","size":[640,640],"layout":"NCHW",
                "mean":[0,0,0],"std":[255,255,255],"color":"RGB"},
      "outputs": {"kind":"detect_xyxy_score_cls","names":["boxes","scores","labels"]},
      "meta": {"source":"rtdetrv2","imgsz":640,"note":"..."}
    }

``outputs.kind`` 只允许三种：
    - ``detect_xyxy_score_cls``：boxes 为 Nx4 像素 xyxy，scores N，labels N
    - ``detect_detr``          ：RT-DETR 原生 logits + pred_boxes
    - ``logits``               ：分类 logits

怎么用::

    from src.train.exporter import build_sidecar, save_sidecar, export_onnx

    sc = build_sidecar(task="classify", label_book=lb, source="dinov2-linear",
                       imgsz=224, input_mean=[0.485,0.456,0.406], input_std=[0.229,0.224,0.225],
                       note="DINOv2 线性探测；需先 resize 到 224 并按 ImageNet 归一化")
    save_sidecar(sc, "models/cls_dinov2/model.onnx.json")
    res = export_onnx(model, "models/cls_dinov2", sc)     # 缺 onnx 时优雅降级并返回原因

依赖：torch（必需）；onnx / onnxruntime 可选（缺失只影响 ONNX 文件生成，sidecar 照常写出）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "TASK_DETECT",
    "TASK_CLASSIFY",
    "OUTPUT_KINDS",
    "SIDE_CAR_FILENAME",
    "ONNX_FILENAME",
    "WEIGHT_FILENAME",
    "build_sidecar",
    "validate_sidecar",
    "save_sidecar",
    "load_sidecar",
    "export_onnx",
    "maybe_export_onnx",
]

TASK_DETECT = "detect"
TASK_CLASSIFY = "classify"
OUTPUT_KINDS = ("detect_xyxy_score_cls", "detect_detr", "logits")

#: 固定文件名（推理侧按此加载，不要改）
WEIGHT_FILENAME = "best.pt"
ONNX_FILENAME = "model.onnx"
SIDE_CAR_FILENAME = "model.onnx.json"


# ---------------------------------------------------------------------------
# sidecar 构造 / 校验 / 读写
# ---------------------------------------------------------------------------
def build_sidecar(
    task: str,
    label_book: Any,
    source: str,
    imgsz: int | Sequence[int],
    input_name: str = "images",
    input_mean: Sequence[float] = (0.0, 0.0, 0.0),
    input_std: Sequence[float] = (255.0, 255.0, 255.0),
    output_kind: str = "detect_xyxy_score_cls",
    output_names: Sequence[str] = ("boxes", "scores", "labels"),
    note: str = "",
    extra_meta: Optional[Dict[str, Any]] = None,
    input_color: str = "RGB",
    input_layout: str = "NCHW",
) -> Dict[str, Any]:
    """构造 sidecar dict（严格字段名）。

    参数:
        task        : 'detect' 或 'classify'
        label_book  : ``src.core.labels.LabelBook``（提供 classes）
        source      : 训练来源标识，如 'rtdetrv2' / 'dinov2-linear' / 'ultralytics-rtdetr'
        imgsz       : 输入边长（int 或 [h, w]）
        input_mean/std : 预处理归一化参数。检测默认 mean=0 / std=255 表示
                      「喂入 [0,1] 范围且不做通道标准化」（RT-DETR 官方 processor 口径）。
        output_kind : OUTPUT_KINDS 之一
        output_names: 与 ONNX 输出顺序一致的名称列表
    """
    if task not in (TASK_DETECT, TASK_CLASSIFY):
        raise ValueError(f"task 只能是 {TASK_DETECT!r} / {TASK_CLASSIFY!r}，收到 {task!r}")
    if output_kind not in OUTPUT_KINDS:
        raise ValueError(f"outputs.kind 只能是 {OUTPUT_KINDS}，收到 {output_kind!r}")
    size = [int(imgsz), int(imgsz)] if isinstance(imgsz, int) else [int(v) for v in imgsz]
    if len(size) != 2:
        raise ValueError(f"imgsz 必须是 int 或 [h, w]，收到 {imgsz!r}")
    meta: Dict[str, Any] = {"source": str(source), "imgsz": int(max(size))}
    if note:
        meta["note"] = str(note)
    for k, v in (extra_meta or {}).items():
        meta[k] = v
    return {
        "task": task,
        "classes": label_book.sidecar_classes(),
        "input": {
            "name": str(input_name),
            "size": size,
            "layout": str(input_layout),
            "mean": [float(v) for v in input_mean],
            "std": [float(v) for v in input_std],
            "color": str(input_color),
        },
        "outputs": {"kind": output_kind, "names": [str(n) for n in output_names]},
        "meta": meta,
    }


def validate_sidecar(sc: Dict[str, Any]) -> List[str]:
    """校验 sidecar 字段是否完整合法，返回问题列表（空列表 = 通过）。"""
    problems: List[str] = []
    if sc.get("task") not in (TASK_DETECT, TASK_CLASSIFY):
        problems.append(f"task 非法：{sc.get('task')!r}（应为 detect/classify）")
    classes = sc.get("classes")
    if not isinstance(classes, list) or not classes:
        problems.append("classes 必须是非空列表")
    else:
        for i, c in enumerate(classes):
            for key in ("class_id", "disease_name", "disease_name_cn"):
                if key not in c:
                    problems.append(f"classes[{i}] 缺少字段 {key}")
            if c.get("class_id") != i:
                problems.append(f"classes[{i}].class_id={c.get('class_id')} 与下标不一致（必须连续从 0 开始）")
    inp = sc.get("input")
    if not isinstance(inp, dict):
        problems.append("input 必须是对象")
    else:
        for key in ("name", "size", "layout", "mean", "std", "color"):
            if key not in inp:
                problems.append(f"input 缺少字段 {key}")
        if isinstance(inp.get("size"), list) and len(inp["size"]) != 2:
            problems.append("input.size 必须是 [h, w]")
        for key in ("mean", "std"):
            v = inp.get(key)
            if not isinstance(v, list) or len(v) not in (1, 3):
                problems.append(f"input.{key} 必须是长度 1 或 3 的列表，收到 {v!r}")
    outs = sc.get("outputs")
    if not isinstance(outs, dict):
        problems.append("outputs 必须是对象")
    else:
        if outs.get("kind") not in OUTPUT_KINDS:
            problems.append(f"outputs.kind 非法：{outs.get('kind')!r}（允许 {OUTPUT_KINDS}）")
        if not isinstance(outs.get("names"), list) or not outs["names"]:
            problems.append("outputs.names 必须是非空列表")
    if not isinstance(sc.get("meta"), dict):
        problems.append("meta 必须是对象")
    return problems


def save_sidecar(sc: Dict[str, Any], path: str | Path | None = None, model_dir: str | Path | None = None) -> Path:
    """写出 sidecar json。

    支持两种调用：``save_sidecar(sc, path)`` 或 ``save_sidecar(sc, model_dir=dir)``（自动拼 model.onnx.json）。
    """
    if path is None:
        if model_dir is None:
            raise ValueError("save_sidecar 需要 path 或 model_dir 之一")
        path = Path(model_dir) / SIDE_CAR_FILENAME
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    problems = validate_sidecar(sc)
    if problems:
        print("[警告] sidecar 校验发现问题：" + "；".join(problems))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(sc, f, ensure_ascii=False, indent=2)
    return p


def load_sidecar(path: str | Path) -> Dict[str, Any]:
    """读 sidecar json（路径可以是目录，自动拼 model.onnx.json）。"""
    p = Path(path)
    if p.is_dir():
        p = p / SIDE_CAR_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"sidecar 不存在：{p}")
    # utf-8-sig：容忍 Windows 上手工编辑/另存为时带上的 BOM，否则 json.load 会直接报错
    with open(p, "r", encoding="utf-8-sig") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# ONNX 导出
# ---------------------------------------------------------------------------
def _result(ok: bool, exported: bool, kind: str, **kw: Any) -> Dict[str, Any]:
    """统一返回结构，训练脚本据此决定退出码与提示。"""
    out = {"ok": bool(ok), "exported": bool(exported), "kind": kind}
    out.update(kw)
    return out


def export_onnx(
    model: Any,
    model_dir: str | Path,
    sidecar: Dict[str, Any],
    opset: int = 17,
    device: str = "cpu",
    sample_input: Any = None,
    input_name: str = "images",
    output_names: Optional[Sequence[str]] = None,
    dynamic_batch: bool = True,
    simplify: bool = True,
) -> Dict[str, Any]:
    """把模型导出成 ``models/<name>/model.onnx``（sidecar 已由调用方写好）。

    返回 dict:
        {"ok":bool, "exported":bool, "kind": "exported"|"missing_dep"|"failed"|"skipped",
         "onnx_path":..., "reason":..., "opset":..., "size_mb":..., "validate": {...}}

    缺 onnx / 模型不支持导出时 **不抛异常**，返回 exported=False + 中文 reason，
    让训练脚本能给出可读提示并以既定退出码结束（产物 partial，但 sidecar 有效）。
    """
    mdir = Path(model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    onnx_path = mdir / ONNX_FILENAME
    kind = sidecar.get("outputs", {}).get("kind", "detect_xyxy_score_cls")
    names = list(output_names or sidecar.get("outputs", {}).get("names", []))
    size = sidecar.get("input", {}).get("size", [640, 640])

    try:
        import torch
    except Exception as exc:
        return _result(False, False, "missing_dep", reason=f"缺少 PyTorch：{exc}")

    # 新版 torch 的 onnx 导出器（dynamo 与 TorchScript 两条路）都 **硬依赖 onnx 包**，
    # 缺它时直接给出中文提示，不去试探性地导出（避免抛一大段 ExportError）。
    try:
        import onnx  # noqa: F401
    except Exception as exc:
        return _result(
            False,
            False,
            "missing_dep",
            reason=(
                "缺少 onnx 包，无法导出 ONNX 文件（sidecar json 已正常写出，torch 后端推理不受影响）。\n"
                "  安装：pip install onnx onnxruntime\n"
                "  然后重跑：python -m scripts.export_onnx --ckpt-dir <模型目录>\n"
                f"  （原始错误：{type(exc).__name__}: {exc}）"
            ),
        )
    have_onnx = True

    if sample_input is None:
        try:
            sample_input = dummy_input_for(sidecar, device=device)
        except Exception as exc:
            return _result(False, False, "failed", reason=f"构造 dummy 输入失败：{exc}")

    device = str(device or "cpu")
    model = model.to(device) if hasattr(model, "to") else model
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()

    dyn = {input_name: {0: "batch"}} if dynamic_batch else None

    # 注意：不同算子的最低 opset 不同（如 aten::scaled_dot_product_attention 需要 >=14），
    # 因此这里只试用户给定的 opset，失败时明确提示如何调整，而不是盲目降级。
    attempts: List[str] = []
    kw: Dict[str, Any] = dict(
        f=str(onnx_path),
        input_names=[input_name],
        output_names=names or None,
        opset_version=int(opset),
        dynamic_axes=dyn,
        do_constant_folding=True,
    )
    for use_dynamo in (False, True):
        try:
            with torch.no_grad():
                torch.onnx.export(model, (sample_input,), dynamo=use_dynamo, **kw)
            attempts.append(f"导出成功（dynamo={use_dynamo}, opset={int(opset)}）")
            break
        except Exception as exc:  # noqa: BLE001 - 导出失败原因多样，统一兜底
            attempts.append(f"dynamo={use_dynamo} 失败：{type(exc).__name__}: {str(exc)[:300]}")
    else:
        return _result(
            False,
            False,
            "failed",
            reason=(
                f"torch.onnx.export 失败（opset={int(opset)}）。常见原因与对策：\n"
                "  1) 算子不支持当前 opset（如注意力需要 opset>=14）-> 加 --opset 17 重试；\n"
                "  2) RT-DETR 的多尺度可变形注意力暂不支持导出 -> 改用 --backend ultralytics-rtdetr"
                "（其自带 ONNX 导出），torch 后端推理不受影响；\n"
                "  3) 确认 onnx 包已装：pip install onnx。\n"
                + "\n".join("  - " + a for a in attempts)
            ),
            attempts=attempts,
        )

    info: Dict[str, Any] = {"size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2)}
    if have_onnx and simplify:
        info["validate"] = _validate_onnx(onnx_path, names, input_name)
    elif not have_onnx:
        info["validate"] = {
            "checked": False,
            "reason": "未安装 onnx / onnxruntime，跳过图校验（onnxruntime 只在推理侧才必须）",
        }
    if was_training is not None and hasattr(model, "train"):
        model.train(bool(was_training))
    return _result(True, True, "exported", onnx_path=str(onnx_path), opset=int(opset),
                   attempts=attempts, **info)


def _validate_onnx(onnx_path: Path, names: Sequence[str], input_name: str) -> Dict[str, Any]:
    """用 onnx / onnxruntime（若可用）做一次最小编译校验。"""
    out: Dict[str, Any] = {"checked": True, "onnx_path": str(onnx_path)}
    try:
        import onnx

        m = onnx.load(str(onnx_path))
        onnx.checker.check_model(m)
        out["onnx_checker"] = "ok"
        out["graph_inputs"] = [i.name for i in m.graph.input]
        out["graph_outputs"] = [o.name for o in m.graph.output]
    except Exception as exc:
        out["onnx_checker"] = f"失败：{type(exc).__name__}: {str(exc)[:200]}"
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out["ort_inputs"] = [i.name for i in sess.get_inputs()]
        out["ort_outputs"] = [o.name for o in sess.get_outputs()]
        out["ort_providers"] = sess.get_providers()
    except Exception as exc:
        out["onnxruntime"] = f"未做推理校验（{type(exc).__name__}）：{str(exc)[:160]}"
    return out


def dummy_input_for(sidecar: Dict[str, Any], device: str = "cpu", dtype: Any = None):
    """按 sidecar 的 input 说明构造 dummy 输入张量（NCHW，尺寸 [h, w]）。"""
    import torch

    h, w = (int(sidecar["input"]["size"][0]), int(sidecar["input"]["size"][1]))
    if dtype is None:
        dtype = torch.float32
    return torch.zeros(1, 3, h, w, dtype=dtype, device=device)


def maybe_export_onnx(
    enabled: bool,
    model: Any,
    model_dir: str | Path,
    sidecar: Dict[str, Any],
    device: str = "cpu",
    opset: int = 17,
    **kw: Any,
) -> Dict[str, Any]:
    """``--export-onnx`` 开关的封装：关闭时直接返回 skipped 结果。"""
    if not enabled:
        return _result(True, False, "skipped", reason="未启用 --export-onnx，只写出 sidecar json")
    return export_onnx(model, model_dir, sidecar, device=device, opset=opset, **kw)


def print_export_result(res: Dict[str, Any]) -> int:
    """按导出结果打印中文提示并给出建议退出码（0 成功/未要求，1 导出失败）。"""
    kind = res.get("kind")
    if kind == "exported":
        print(f"[完成] ONNX 已导出：{res.get('onnx_path')}")
        print(f"        校验：{res.get('validate')}")
        return 0
    if kind == "skipped":
        print(f"[信息] {res.get('reason')}")
        return 0
    if kind == "missing_dep":
        print(f"[错误] {res.get('reason')}", file=sys.stderr)
        return 1
    print(f"[警告] ONNX 导出失败（训练权重与 sidecar 仍然有效）：{res.get('reason')}", file=sys.stderr)
    for a in res.get("attempts", []):
        print(f"        - {a}", file=sys.stderr)
    return 1
