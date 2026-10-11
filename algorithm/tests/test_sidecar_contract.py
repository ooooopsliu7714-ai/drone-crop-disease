"""跨侧契约校验：训练侧写出的 sidecar 元数据，推理侧必须能原样消费。

这是整个项目最容易"静默出错"的地方——训练同学写 task="detection"、
推理同学只认 task="detect"，结果就是生产环境悄悄退化成规则基线而没人发现。
本测试不依赖 onnxruntime：直接调训练侧的 sidecar 构造/保存函数，
再用推理侧的严格要求校验字段。

运行：
    python tests/test_sidecar_contract.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labels import LABELS  # noqa: E402

# 推理侧只认这几种解码方式（src/infer/onnx_backend.py::_decode）
ALLOWED_OUTPUT_KINDS = {"detect_xyxy_score_cls", "detect_detr", "logits"}
ALLOWED_TASKS = {"detect", "classify"}


def _validate_sidecar(meta: dict) -> list[str]:
    """按推理侧的严格要求校验 sidecar，返回问题列表（空列表 = 通过）。"""
    problems: list[str] = []
    if meta.get("task") not in ALLOWED_TASKS:
        problems.append(f"task={meta.get('task')!r} 不在 {sorted(ALLOWED_TASKS)}")

    inp = meta.get("input") or {}
    size = inp.get("size")
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        problems.append(f"input.size 必须是 [H, W]，实际 {size!r}")
    for key in ("name", "layout", "mean", "std", "color"):
        if key not in inp:
            problems.append(f"input 缺少字段 {key}")
    if len(inp.get("mean", [])) != 3 or len(inp.get("std", [])) != 3:
        problems.append("input.mean / input.std 必须是 3 个通道的值")

    outputs = meta.get("outputs") or {}
    if outputs.get("kind") not in ALLOWED_OUTPUT_KINDS:
        problems.append(
            f"outputs.kind={outputs.get('kind')!r} 推理侧无法解码，"
            f"只能是 {sorted(ALLOWED_OUTPUT_KINDS)}"
        )
    names = [c.get("disease_name") for c in sorted(meta.get("classes") or [], key=lambda c: c.get("class_id", 0))]
    if names != LABELS.names:
        problems.append(
            "classes 与 configs/labels.yaml 不一致：\n"
            f"    sidecar : {names}\n"
            f"    labels  : {LABELS.names}"
        )
    return problems


def _train_side_helpers():  # noqa: ANN202
    """惰性 import 训练侧的 sidecar 工具，缺失时返回 None。"""
    try:
        from src.train import exporter
    except Exception as exc:  # pragma: no cover
        print(f"[skip] 训练侧 exporter 不可用：{exc}")
        return None
    return exporter


def test_detect_sidecar_roundtrip() -> None:
    """检测任务的 sidecar：构造 -> 保存 -> 读回 -> 推理侧校验。"""
    exporter = _train_side_helpers()
    if exporter is None:
        return
    from src.core.labels import load_labels

    tmp = Path(tempfile.mkdtemp())
    sc = exporter.build_sidecar(
        task="detect",
        label_book=load_labels(),
        source="rtdetrv3",
        imgsz=640,
        input_mean=[0.0, 0.0, 0.0],
        input_std=[255.0, 255.0, 255.0],
        output_kind="detect_detr",
        output_names=["logits", "pred_boxes"],
        note="跨侧契约测试",
    )
    saved = exporter.save_sidecar(sc, model_dir=tmp)
    assert Path(saved).exists(), "sidecar 未落盘"
    assert Path(saved).name == "model.onnx.json", f"文件名必须是 model.onnx.json，实际 {Path(saved).name}"

    back = json.loads(Path(saved).read_text(encoding="utf-8"))
    problems = _validate_sidecar(back)
    assert not problems, "sidecar 与推理侧契约不一致：\n  - " + "\n  - ".join(problems)

    # 训练侧自己的校验必须也通过（两侧不能各说各话）
    own = exporter.validate_sidecar(back)
    assert not own, f"训练侧 validate_sidecar 报错：{own}"
    print(f"[PASS] detect sidecar 一致：kind={back['outputs']['kind']} size={back['input']['size']}")


def test_classify_sidecar_roundtrip() -> None:
    """分类任务的 sidecar 同样要能过推理侧校验（归一化参数不同）。"""
    exporter = _train_side_helpers()
    if exporter is None:
        return
    from src.core.labels import load_labels

    tmp = Path(tempfile.mkdtemp())
    sc = exporter.build_sidecar(
        task="classify",
        label_book=load_labels(),
        source="dinov3",
        imgsz=224,
        input_mean=[0.485, 0.456, 0.406],
        input_std=[0.229, 0.224, 0.225],
        output_kind="logits",
        output_names=["logits"],
        note="跨侧契约测试",
    )
    saved = exporter.save_sidecar(sc, model_dir=tmp)
    back = json.loads(Path(saved).read_text(encoding="utf-8"))
    problems = _validate_sidecar(back)
    assert not problems, "分类 sidecar 与推理侧契约不一致：\n  - " + "\n  - ".join(problems)
    assert back["input"]["color"].upper() == "RGB", "深度学习预处理按 RGB，sidecar 必须声明 color=RGB"
    print(f"[PASS] classify sidecar 一致：kind={back['outputs']['kind']} mean={back['input']['mean']}")


def test_infer_sidecar_from_ckpt_dir() -> None:
    """从 checkpoint.json 推断 sidecar（训练脚本与导出脚本之间的交接点）。"""
    try:
        import scripts.export_onnx as exp
    except Exception as exc:  # pragma: no cover
        print(f"[skip] scripts.export_onnx 不可用：{exc}")
        return
    from src.core.labels import load_labels

    tmp = Path(tempfile.mkdtemp())
    (tmp / "checkpoint.json").write_text(
        json.dumps(
            {
                "task": "detect",
                "source": "rtdetrv3",
                "image_size": 800,
                "output_kind": "detect_detr",
                "output_names": ["logits", "pred_boxes"],
            }
        ),
        encoding="utf-8",
    )
    # 只设置导出脚本自己声明的参数，避免 AttributeError
    defaults: dict[str, object] = {
        "ckpt_dir": str(tmp),
        "imgsz": 0,
        "task": None,
        "output_kind": None,
        "output_names": None,
    }
    for action in exp.build_parser()._actions:
        if action.dest and action.dest != "help":
            defaults.setdefault(action.dest, None)
    args = argparse.Namespace(**defaults)

    sc = exp.infer_sidecar_from_ckpt(tmp, load_labels(), args)
    problems = _validate_sidecar(sc)
    assert not problems, "从 checkpoint.json 推断的 sidecar 不合契约：\n  - " + "\n  - ".join(problems)
    assert sc["input"]["size"] == [800, 800], f"应沿用 checkpoint.json 的 image_size=800，实际 {sc['input']['size']}"
    assert sc["meta"]["source"] == "rtdetrv3"
    print("[PASS] checkpoint.json -> sidecar 推断正确（image_size 继承、source 继承）")


def test_labels_sidecar_helper_matches_names() -> None:
    """LABELS.to_sidecar_classes() 的输出必须与 labels.yaml 顺序一致。"""
    sc = LABELS.to_sidecar_classes()
    assert [c["disease_name"] for c in sc] == LABELS.names
    assert all({"class_id", "disease_name", "disease_name_cn"} <= set(c) for c in sc)


def test_onnx_graph_input_matches_sidecar_input_name() -> None:
    """已有模型产物：sidecar 的 input.name 必须等于 ONNX 图里真实的输入名。

    真实事故（本机实测）：分类模型导出时图里的输入名是 ``pixel_values``（DINOv2 惯例），
    而 sidecar 一直写死 ``images``。ONNX 后端按 sidecar 的名字喂数据 → 部署时直接报错。
    因为本机没装 onnxruntime、单元测试用的是桩 session，**这个不一致一直没被发现**，
    直到直接读 ONNX 图才暴露。本测试用 onnx 包直读图（不需要 onnxruntime）来兜住它。
    """
    try:
        import onnx  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] 未安装 onnx（{exc}），无法校验图与 sidecar 的一致性")
        return

    checked = 0
    for d in sorted((ROOT / "models").iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        onnx_p, sc_p = d / "model.onnx", d / "model.onnx.json"
        if not onnx_p.is_file() or not sc_p.is_file():
            continue
        graph = onnx.load(str(onnx_p)).graph
        assert graph.input, f"{d.name}: ONNX 图没有输入"
        real = graph.input[0].name
        declared = (json.loads(sc_p.read_text(encoding="utf-8")).get("input") or {}).get("name")
        assert real == declared, (
            f"{d.name}: sidecar.input.name={declared!r} 与 ONNX 图输入名={real!r} 不一致。\n"
            "  ONNX 后端按 sidecar 的名字喂数据，不一致会在部署时抛运行时错误。\n"
            "  修复：重跑 `python -m scripts.export_onnx --ckpt-dir <dir> --task <task>`，"
            "导出后会自动以图为准同步 sidecar。"
        )
        checked += 1
    print(f"[PASS] 已校验 {checked} 个模型产物的 ONNX 图与 sidecar 输入名一致")


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print(f"\n共 {len(tests)} 项，失败 {failed} 项")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
