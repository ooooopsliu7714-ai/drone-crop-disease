"""导出 ONNX + sidecar json（训练链路的最后一步，也是推理侧唯一需要的产物形态）。

产物路径固定（不要改）::

    models/<name>/best.pt            权重（训练脚本产出）
    models/<name>/model.onnx         ONNX 图（本脚本产出）
    models/<name>/model.onnx.json    sidecar 元数据（本脚本 / 训练脚本产出）

sidecar 字段（严格）::

    {
      "task": "detect",
      "classes": [{"class_id":0,"disease_name":"healthy","disease_name_cn":"健康"}],
      "input": {"name":"images","size":[640,640],"layout":"NCHW",
                "mean":[0,0,0],"std":[255,255,255],"color":"RGB"},
      "outputs": {"kind":"detect_xyxy_score_cls","names":["boxes","scores","labels"]},
      "meta": {"source":"rtdetrv2","imgsz":640,"note":"..."}
    }

``outputs.kind`` 只允许：``detect_xyxy_score_cls`` / ``detect_detr`` / ``logits``。

怎么用::

    # 1) 只写 sidecar（不需要 torch 权重、不联网，最快）
    python -m scripts.export_onnx --ckpt-dir models/cls_dinov2 --sidecar-only

    # 2) 完整导出（torch -> onnx；缺 onnx 包时给出中文提示，sidecar 仍然有效）
    python -m scripts.export_onnx --ckpt-dir models/det_rtdetrv2 --imgsz 640 --opset 17

    # 3) 覆盖记录进 sidecar 的信息（例如换成 ultralytics 后端产出的权重）
    python -m scripts.export_onnx --ckpt-dir models/det_yolo --source ultralytics-rtdetr \
        --output-kind detect_xyxy_score_cls --note "ultralytics RT-DETR 微调"

依赖：torch（导出必需）；onnx / onnxruntime 可选（仅用于图校验，训练链路不依赖 onnxruntime）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.labels import load_labels
from src.core.paths import MODELS_DIR, PROJECT_ROOT, ensure_dir
from src.train.exporter import (
    ONNX_FILENAME,
    SIDE_CAR_FILENAME,
    WEIGHT_FILENAME,
    build_sidecar,
    dummy_input_for,
    export_onnx,
    load_sidecar,
    print_export_result,
    save_sidecar,
    validate_sidecar,
)

#: detect 任务的默认归一化：喂入 [0,1] 且不做通道标准化（RT-DETR 官方 processor 口径）
DETECT_MEAN = [0.0, 0.0, 0.0]
DETECT_STD = [255.0, 255.0, 255.0]
#: classify 任务的默认归一化（DINOv2/DINOv3 官方预处理）
CLASSIFY_MEAN = [0.485, 0.456, 0.406]
CLASSIFY_STD = [0.229, 0.224, 0.225]

DEFAULT_NAMES = {
    "detect_xyxy_score_cls": ["boxes", "scores", "labels"],
    "detect_detr": ["logits", "pred_boxes"],
    "logits": ["logits"],
}


def infer_sidecar_from_ckpt(ckpt_dir: Path, label_book, args: argparse.Namespace) -> Dict[str, Any]:
    """从 ``checkpoint.json``（训练脚本写的）推断 sidecar，缺失字段用约定默认值。"""
    info: Dict[str, Any] = {}
    cj = ckpt_dir / "checkpoint.json"
    if cj.is_file():
        # utf-8-sig：容忍手工编辑/另存为带来的 BOM（否则 json.load 直接失败）
        with open(cj, "r", encoding="utf-8-sig") as f:
            info = json.load(f)
    elif (ckpt_dir / SIDE_CAR_FILENAME).is_file():
        # 已有 sidecar：读取它作为基底，命令行参数再覆盖
        info = load_sidecar(ckpt_dir)

    task = str(args.task or info.get("task") or ("detect" if "detect" in str(info.get("source", "")) else "classify"))
    source = str(args.source or info.get("source") or info.get("meta", {}).get("source") or "")
    if not source:
        # 从 checkpoint.json 的结构信息推断来源标识，便于推理侧选择后端
        bb = str(info.get("backbone") or info.get("pretrained") or "").lower()
        if task == "classify":
            source = "dinov3-linear" if "dinov3" in bb else "dinov2-linear"
        else:
            source = "rtdetrv2" if "rtdetr" in bb else "unknown-detect"
    imgsz = int(args.imgsz or info.get("image_size") or info.get("imgsz") or info.get("meta", {}).get("imgsz")
                or (224 if task == "classify" else 640))
    note = args.note or str(info.get("note") or info.get("meta", {}).get("note") or "")
    mean = args.mean if args.mean else (info.get("mean") or info.get("meta", {}).get("mean")
                                       or (CLASSIFY_MEAN if task == "classify" else DETECT_MEAN))
    std = args.std if args.std else (info.get("std") or info.get("meta", {}).get("std")
                                    or (CLASSIFY_STD if task == "classify" else DETECT_STD))
    # 检测任务的默认输出口径必须按**模型来源**决定，不能一律用 detect_xyxy_score_cls。
    #
    # 真实事故（本机实测）：RT-DETR 这类 transformers 模型，forward 返回的是
    # RTDetrV2ObjectDetectionOutput，to_tuple() 的顺序是
    #   [0] logits (1,300,C)   [1] pred_boxes (1,300,4)   [2] last_hidden_state (1,300,256)
    # 而 torch.onnx.export(output_names=[...]) **只是按顺序改名，不做任何语义映射**，
    # 所以把契约名 ("boxes","scores","labels") 套上去后，图里变成：
    #   boxes  = logits            (1,300,2)
    #   scores = pred_boxes        (1,300,4)
    #   labels = last_hidden_state (1,300,256)
    # 解码层按 detect_xyxy_score_cls 去读，拿到的全是张冠李戴的数据 —— ONNX 后端
    # 因此每张图只吐 1 个框、主病害判反。**而单元测试用的是桩 session，一直没暴露。**
    # 正确做法：声明 detect_detr，让图直接输出原生的 logits + pred_boxes，
    # 后处理交给 onnx_backend 里已有的（且被测试锁住的）那条分支。
    raw_source = str(info.get("source") or info.get("meta", {}).get("source") or "")
    is_rtdetr = task == "detect" and "rtdetr" in raw_source.lower()
    default_kind = "logits" if task == "classify" else ("detect_detr" if is_rtdetr else "detect_xyxy_score_cls")
    kind = str(args.output_kind or info.get("output_kind") or info.get("meta", {}).get("output_kind") or default_kind)
    # RT-DETR 必须用原生的两个输出名，否则解码层 named.get() 取不到会退回按位置取
    default_names = DEFAULT_NAMES[kind]
    names = args.output_names or info.get("output_names") or info.get("meta", {}).get("output_names") or default_names
    if is_rtdetr and not (args.output_names or info.get("output_names")):
        names = default_names

    extra: Dict[str, Any] = {}
    for k in ("backbone", "pretrained", "head", "pooling", "trainer", "backend", "epochs",
              "best_epoch", "prompt_lang"):
        if k in info:
            extra[k] = info[k]

    # 置信度阈值必须**在重新导出时保留**。
    # 真实事故（本机实测）：阈值是在 val 集上标定后手工写进 sidecar 的
    # （本模型分数整体偏低，默认 0.25 会筛掉几乎所有框）。重跑一次 export_onnx
    # 就把 sidecar 从 checkpoint.json 重新生成，标定值被抹掉 → torch 后端
    # 立刻退化（每图 1 个框、置信度 0.000、主病害判成 healthy）。
    # 因此这里显式继承旧 sidecar 的 conf_threshold，除非命令行给了新值。
    prev_sc = ckpt_dir / SIDE_CAR_FILENAME
    prev_conf = None
    if prev_sc.is_file():
        try:
            prev_conf = (json.loads(prev_sc.read_text(encoding="utf-8")).get("meta") or {}).get("conf_threshold")
        except Exception:
            prev_conf = None
    conf = args.conf_threshold if getattr(args, "conf_threshold", None) is not None else prev_conf
    if conf is not None:
        extra["conf_threshold"] = float(conf)
    if args.device:
        extra["export_device"] = str(args.device)
    if args.backend_impl:
        extra["backend_impl"] = str(args.backend_impl)

    return build_sidecar(
        task=task,
        label_book=label_book,
        source=source,
        imgsz=imgsz,
        input_mean=mean,
        input_std=std,
        output_kind=kind,
        output_names=names,
        note=note or f"由 scripts/export_onnx.py 生成（task={task}, source={source}）",
        extra_meta=extra,
    )


def _sync_sidecar_input_name(ckpt_dir: Path, sidecar_path: Path) -> str | None:
    """把 sidecar 的 `input.name` 同步成 ONNX 图里真实的输入名。

    为什么要这一步（本机实测抓到的部署级 bug）：分类模型导出时
    `_build_model_for_export` 用的输入名是 ``pixel_values``（DINOv2 的惯例），
    而 sidecar 一直写死 ``images``。ONNX 后端按 sidecar 的名字喂数据，
    名字对不上会直接抛运行时错误 —— 而**因为本机没装 onnxruntime、
    测试用的是桩 session，这个不一致一直没被发现**，直到直接读 ONNX 图才暴露：

        cls_dinov2 : 图中 'pixel_values'  vs  sidecar 'images'   ✗
        det_rtdetrv2: 图中 'images'        vs  sidecar 'images'   ✓

    修法：导出后**从图里读回真实名字**再写入 sidecar，让两者不可能再分叉
    （图是唯一真源）。
    """
    onnx_path = ckpt_dir / ONNX_FILENAME
    if not onnx_path.is_file():
        return None
    try:
        import onnx  # type: ignore
    except Exception:
        return "未安装 onnx，跳过输入名同步（部署前务必确认 sidecar.input.name 与图一致）"
    try:
        graph = onnx.load(str(onnx_path)).graph
        real = graph.input[0].name if graph.input else None
    except Exception as exc:  # noqa: BLE001
        return f"读取 ONNX 图失败，未同步输入名：{exc}"
    if not real:
        return None
    try:
        sc = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    old = (sc.get("input") or {}).get("name")
    if old == real:
        return f"sidecar.input.name 与 ONNX 图一致：{real}"
    sc.setdefault("input", {})["name"] = real
    Path(sidecar_path).write_text(json.dumps(sc, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"已修正 sidecar.input.name：{old!r} -> {real!r}（以 ONNX 图为准）"


def _build_model_for_export(ckpt_dir: Path, sidecar: Dict[str, Any], device: str):
    """按 sidecar 的 source 重建可导出的模型（复用 predictors 的加载逻辑）。"""
    from src.train.predictors import build_predictor

    predictor = build_predictor(str(ckpt_dir), device=device)
    task = sidecar.get("task")
    if task == "classify":
        return predictor.clf.net, "pixel_values"
    if hasattr(predictor, "model"):
        return predictor.model, "images"
    raise RuntimeError(f"无法从 {ckpt_dir} 的推理器中找到可导出的 torch 模型（source={sidecar['meta'].get('source')}）")


def _torch_dummy(sidecar: Dict[str, Any], device: str):
    """构造导出用 dummy 输入（NCHW）。"""
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "缺少 PyTorch，无法导出 ONNX。\n"
            "  CPU：pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  RTX 5070（sm_120）：pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            f"（原始错误：{exc}）"
        ) from exc
    return dummy_input_for(sidecar, device=device)


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    ap = argparse.ArgumentParser(
        prog="python -m scripts.export_onnx",
        description="导出 ONNX + sidecar json（models/<name>/model.onnx 与 model.onnx.json）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m scripts.export_onnx --ckpt-dir models/cls_dinov2 --sidecar-only\n"
            "  python -m scripts.export_onnx --ckpt-dir models/det_rtdetrv2 --imgsz 640 --opset 17\n"
        ),
    )
    ap.add_argument("--ckpt-dir", required=True, help="模型目录（含 best.pt / checkpoint.json）")
    ap.add_argument("--imgsz", type=int, default=0, help="输入边长，默认取 checkpoint.json 里的 image_size")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset，默认 17（失败会自动试 13）")
    ap.add_argument("--device", default="cpu", help="导出设备，默认 cpu（导出用 CPU 更稳）")
    ap.add_argument("--task", default=None, choices=[None, "detect", "classify"], help="任务类型，默认从 checkpoint.json 推断")
    ap.add_argument("--source", default=None, help="meta.source，如 rtdetrv2 / dinov2-linear / ultralytics-rtdetr")
    ap.add_argument("--backend-impl", default=None, help="meta.backend_impl 备注（如 transformers-rtdetrv2）")
    ap.add_argument("--output-kind", default=None,
                    choices=[None, "detect_xyxy_score_cls", "detect_detr", "logits"],
                    help="outputs.kind，默认 detect->detect_xyxy_score_cls / classify->logits")
    ap.add_argument("--output-names", nargs="*", default=None, help="ONNX 输出名列表，需与图输出顺序一致")
    ap.add_argument("--conf-threshold", type=float, default=None,
                    help="写入 sidecar 的置信度阈值。不给则**继承旧 sidecar 的值**"
                         "（避免重新导出把标定好的阈值抹掉）")
    ap.add_argument("--mean", nargs=3, type=float, default=None, help="sidecar input.mean（3 个浮点数）")
    ap.add_argument("--std", nargs=3, type=float, default=None, help="sidecar input.std（3 个浮点数）")
    ap.add_argument("--note", default=None, help="写入 sidecar meta.note 的说明")
    ap.add_argument("--sidecar-only", action="store_true", help="只写 sidecar json，不导出 ONNX（不需要 torch/权重）")
    ap.add_argument("--no-simplify", action="store_true", help="跳过 ONNX 图校验/简化")
    ap.add_argument("--static-batch", action="store_true", help="导出静态 batch（默认动态 batch=1）")
    ap.add_argument("--print-sidecar", action="store_true", help="把生成的 sidecar 打印出来")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口：返回退出码（0 成功 / 1 导出失败 / 2 参数或依赖问题 / 3 标签缺失）。"""
    args = build_parser().parse_args(argv)
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        cand = PROJECT_ROOT / ckpt_dir
        ckpt_dir = cand if cand.exists() else ckpt_dir
    if not ckpt_dir.is_dir():
        print(f"[错误] 模型目录不存在：{ckpt_dir}", file=sys.stderr)
        return 2
    ensure_dir(ckpt_dir)

    try:
        lb = load_labels()
    except Exception as exc:
        print(f"[错误] 标签体系加载失败：{exc}", file=sys.stderr)
        return 3

    print("=" * 78)
    print("导出 ONNX + sidecar json")
    print("=" * 78)
    print(f"模型目录：{ckpt_dir.resolve()}")
    print(f"标签体系：{lb.num_classes} 类（{Path(lb.source).name if lb.source else 'labels.yaml'}）")

    sidecar = infer_sidecar_from_ckpt(ckpt_dir, lb, args)
    problems = validate_sidecar(sidecar)
    if problems:
        print("[错误] sidecar 字段校验失败：" + "；".join(problems), file=sys.stderr)
        return 2
    sidecar_path = save_sidecar(sidecar, model_dir=ckpt_dir)
    print(f"[完成] sidecar 已写出：{sidecar_path}")
    print(f"        task={sidecar['task']}  outputs.kind={sidecar['outputs']['kind']}  "
          f"input.size={sidecar['input']['size']}  source={sidecar['meta']['source']}")
    if args.print_sidecar:
        print(json.dumps(sidecar, ensure_ascii=False, indent=2))

    if args.sidecar_only:
        print("[信息] --sidecar-only：跳过 ONNX 导出（推理侧 torch 后端只需 best.pt + sidecar）")
        return 0

    wpath = ckpt_dir / WEIGHT_FILENAME
    if not wpath.is_file():
        print(
            f"[警告] 未找到权重 {wpath}，无法导出 ONNX。\n"
            "        请先跑训练脚本（scripts/train_det.py / train_cls.py）生成 best.pt，"
            "或使用 --sidecar-only 只生成 sidecar。",
            file=sys.stderr,
        )
        return 1

    try:
        model, input_name = _build_model_for_export(ckpt_dir, sidecar, args.device)
        dummy = _torch_dummy(sidecar, args.device)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[错误] 重建模型失败：{exc}", file=sys.stderr)
        return 2

    res = export_onnx(
        model,
        ckpt_dir,
        sidecar,
        opset=int(args.opset),
        device=args.device,
        sample_input=dummy,
        input_name=input_name,
        output_names=sidecar["outputs"]["names"],
        dynamic_batch=not args.static_batch,
        simplify=not args.no_simplify,
    )
    code = print_export_result(res)
    if res.get("exported"):
        print(f"        文件：{ckpt_dir / ONNX_FILENAME}（{res.get('size_mb')} MB）")
        sync_input_name = _sync_sidecar_input_name(ckpt_dir, sidecar_path)
        if sync_input_name:
            print(f"[信息] {sync_input_name}")
    print("\n推理侧用法：")
    print("  from src.train.predictors import build_predictor")
    print(f'  predictor = build_predictor(r"{ckpt_dir}", device="auto")')
    print("  results = predictor.predict([bgr_image])   # -> disease_name/confidence/severity/class_id/bbox")
    _ = MODELS_DIR
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
