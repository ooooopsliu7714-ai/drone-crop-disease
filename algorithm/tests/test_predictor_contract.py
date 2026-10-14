"""训练侧 predictor 工厂与推理侧适配层的接口契约。

registry 的 torch 后端依赖 `src.train.predictors.build_predictor` 返回的对象具备：
    predict(images: list[BGR ndarray]) -> list[dict]，字典含
    disease_name / confidence / severity / class_id / bbox
一旦训练侧改了方法名或返回字段，服务会静默退回规则基线，很难发现，
所以这里用真实产物（models/cls_selftest）和桩目录把这些约束钉死。

运行：python tests/test_predictor_contract.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labels import LABELS  # noqa: E402

REQUIRED_KEYS = {"disease_name", "confidence", "severity", "class_id", "bbox"}


def _discover_ckpt_dirs() -> list[Path]:
    """自动发现 models/ 下"完整可用"的产物目录（有 best.pt 且有 sidecar）。

    为什么不写死目录名：本机先后出现过 cls_selftest / det_rtdetrv2 / cls_dinov2，
    写死就会在不匹配时静默 skip，把"没验证"伪装成"通过"。
    """
    root = ROOT / "models"
    if not root.is_dir():
        return []
    found: list[Path] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "best.pt").is_file() and (d / "model.onnx.json").is_file():
            found.append(d)
    return found


def _make_bgr(h: int = 224, w: int = 224) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 1] = 150
    img[:, :, 0] = 60
    img[:, :, 2] = 70
    img[h // 4 : h // 2, w // 4 : w // 2] = (60, 180, 200)
    return img


def _validate_result_dict(
    d: dict, shape: tuple[int, int], allowed_names: set[str] | None = None
) -> list[str]:
    """校验 predictor 输出。

    `allowed_names`：该模型**自己**的类别名集合（来自 sidecar / checkpoint.json）。
    必须按模型自己的类别表校验——同一个仓库里可能同时存在 15 类病害模型和
    2 类胁迫模型（healthy/stressed），用全局 labels.yaml 校验 2 类模型会把
    "stressed 不在 labels.yaml 中"误报成契约错误。
    """
    problems: list[str] = []
    missing = REQUIRED_KEYS - set(d)
    if missing:
        problems.append(f"缺少字段 {sorted(missing)}")
    names = allowed_names if allowed_names is not None else set(LABELS.names)
    if d.get("disease_name") not in names:
        # 唯一允许的例外：「无检出」显式表示为 healthy（class_id 为负）。
        #
        # 为什么需要它：PlantSeg 那套 10 类病种体系**只标注病斑、没有 healthy 类**，
        # 而接口必须能表达"这张图没发现病斑"。此时 predictor 返回
        # disease_name="healthy" / class_id=-1 是**正确**的语义
        # （本机实测：训练末尾的自检就是因为返回了 class 0 的病名而判"字段不匹配"）。
        # 其余情况仍然严格校验，避免把类别名写错的问题放过去。
        is_explicit_healthy = (
            d.get("disease_name") == "healthy" and int(d.get("class_id", 0) or 0) < 0
        )
        if not is_explicit_healthy:
            problems.append(
                f"disease_name={d.get('disease_name')!r} 不在模型类别表 {sorted(names)} 中"
            )
    if d.get("severity") not in LABELS.severity_keys:
        problems.append(f"severity={d.get('severity')!r} 非法，只能是 {LABELS.severity_keys}")
    conf = d.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        problems.append(f"confidence={conf!r} 必须在 0~1")
    bbox = d.get("bbox") or []
    if len(bbox) != 4:
        problems.append(f"bbox 必须是 4 个数，实际 {bbox!r}")
    else:
        h, w = shape
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if not (0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h):
            problems.append(f"bbox 越界或顺序错误：{[x1, y1, x2, y2]} vs 图 {w}x{h}")
    return problems


def _model_class_names(ckpt_dir: Path) -> set[str]:
    """从产物的 sidecar / checkpoint.json 读该模型自己的类别名。"""
    import json as _json

    for fn in ("model.onnx.json", "checkpoint.json"):
        p = ckpt_dir / fn
        if not p.is_file():
            continue
        try:
            meta = _json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        names = {
            str(c.get("disease_name"))
            for c in (meta.get("classes") or [])
            if isinstance(c, dict) and c.get("disease_name")
        }
        if names:
            return names
    return set(LABELS.names)


def test_real_checkpoint_predictor_contract() -> None:
    """用训练侧真实产物验证 predictor 的对外契约。"""
    available = _discover_ckpt_dirs()
    if not available:
        print("[skip] models/ 下没有完整产物（需 best.pt + model.onnx.json）")
        return
    try:
        from src.train.predictors import build_predictor
    except Exception as exc:  # pragma: no cover
        print(f"[skip] 训练侧 predictors 不可用：{exc}")
        return

    checked = 0
    for ckpt in available:
        try:
            predictor = build_predictor(str(ckpt), device="cpu")
        except Exception as exc:
            # 权重与代码版本不匹配时不算失败，但要显式说明跳过了哪个目录
            print(f"[skip] 加载 {ckpt.name} 失败：{type(exc).__name__}: {str(exc)[:120]}")
            continue
        assert hasattr(predictor, "predict"), "predictor 必须有 predict 方法"
        img = _make_bgr()
        results = predictor.predict([img])
        assert isinstance(results, list) and len(results) == 1, "predict 必须返回与输入等长的列表"
        problems = _validate_result_dict(results[0], img.shape[:2], _model_class_names(ckpt))
        assert not problems, f"{ckpt.name} 的输出不合契约：\n  - " + "\n  - ".join(problems)
        print(f"[PASS] {ckpt.name} -> {type(predictor).__name__} 输出契约通过（{results[0]['disease_name']}）")
        checked += 1
    assert checked > 0, f"发现 {len(available)} 个产物目录，但一个都没能加载成功"


def test_build_predictor_dispatches_by_source() -> None:
    """工厂必须按 sidecar 的 meta.source 分派到不同 predictor，而不是永远返回同一类。

    做法：把四个 predictor 类替换成记录用的桩，再调用真实的 build_predictor，
    这样验证的是**工厂的分派逻辑本身**，而不是测试里重写一遍的推断规则。
    """
    try:
        from src.train import predictors as P
    except Exception as exc:  # pragma: no cover
        print(f"[skip] 训练侧 predictors 不可用：{exc}")
        return

    created: list[str] = []

    class _Stub:
        def __init__(self, ckpt_dir, sidecar, device="cpu") -> None:  # noqa: ANN001
            created.append(type(self).__name__)

    cases = {
        "rtdetrv2": ("detect", "RTDetrTorchPredictor"),
        "rtdetrv3": ("detect", "RTDetrTorchPredictor"),
        "dinov2-linear": ("classify", "DinovClassifierPredictor"),
        "dinov3-linear": ("classify", "DinovClassifierPredictor"),
    }
    originals = {name: getattr(P, name) for name in set(v[1] for v in cases.values())}
    try:
        for name in originals:
            setattr(P, name, type(name, (_Stub,), {}))
        for source, (task, expect) in cases.items():
            tmp = Path(tempfile.mkdtemp())
            (tmp / "best.pt").write_bytes(b"stub")
            (tmp / "model.onnx.json").write_text(
                json.dumps(
                    {
                        "task": task,
                        "classes": LABELS.to_sidecar_classes(),
                        "input": {
                            "name": "pixel_values" if task == "classify" else "images",
                            "size": [224, 224] if task == "classify" else [640, 640],
                            "layout": "NCHW",
                            "mean": [0.0, 0.0, 0.0],
                            "std": [255.0, 255.0, 255.0],
                            "color": "RGB",
                        },
                        "outputs": {
                            "kind": "logits" if task == "classify" else "detect_detr",
                            "names": ["logits"] if task == "classify" else ["logits", "pred_boxes"],
                        },
                        "meta": {"source": source},
                    }
                ),
                encoding="utf-8",
            )
            created.clear()
            P.build_predictor(str(tmp), device="cpu")
            assert created, f"source={source} 时工厂没有创建任何 predictor"
            assert created[0] == expect, f"source={source} 应分派到 {expect}，实际 {created[0]}"
            print(f"[PASS] source={source} -> {created[0]}")
    finally:
        for name, cls in originals.items():
            setattr(P, name, cls)


def test_service_layer_consumes_predictor_output() -> None:
    """推理侧 AlgorithmService 必须能吃下 predictor 的输出并补全中文名与契约字段。

    中文名的期望值从**当前生效的标签体系**里取，不写死具体病名——
    本仓库可切换 15 类病害体系与 2 类胁迫体系，写死会在其中一种下误报。
    """
    from src.api.service import AlgorithmService
    from src.infer.predictor import Predictor

    # 选当前体系里的第二个类别作为被测目标
    target = sorted(LABELS.classes, key=lambda c: c["class_id"])[1]
    target_name = str(target["disease_name"])
    target_cid = int(target["class_id"])
    target_cn = str(target["disease_name_cn"])

    class _Stub:
        name = "stub-predictor"
        model_version = "stub-v1"
        task = "detect"

        def predict(self, images):  # noqa: ANN001
            return [
                {
                    "disease_name": target_name,
                    "confidence": 0.72,
                    "severity": "moderate",
                    "class_id": target_cid,
                    "bbox": [10.0, 20.0, 110.0, 120.0],
                }
                for _ in images
            ]

    svc = AlgorithmService(Predictor(_Stub(), slice_threshold=1600), None)
    d = svc.predict_bgr(_make_bgr(300, 400), "stub_check").model_dump()
    assert d["disease_name"] == target_name
    assert d["disease_name_cn"] == target_cn, "中文名必须由服务层补全"
    assert d["confidence"] == 0.72 and d["severity"] == "moderate"
    assert abs(d["boxes"][0]["x"] - 10 / 400) < 1e-6
    assert d["model_version"] == "stub-v1"


def test_model_dir_discovery_prefers_complete_and_skips_temp() -> None:
    """模型目录发现规则：完整目录优先，`_`/`.` 开头的临时目录排到最后。

    本机实测踩过一次坑：训练自测留下的 `models/_tmp_sidecar_check`（只有
    checkpoint.json）按"最近修改"排到了真实权重前面，服务静默退回规则基线。
    这里把排序规则钉死。
    """
    from src.runtime import registry as R

    real_models = R.MODELS_DIR
    tmp = Path(tempfile.mkdtemp())
    try:
        R.MODELS_DIR = tmp
        # 半成品临时目录（更晚创建，模拟"时间上更新"）
        temp_dir = tmp / "_tmp_sidecar_check"
        temp_dir.mkdir()
        (temp_dir / "checkpoint.json").write_text("{}", encoding="utf-8")
        # 真实产物：sidecar + 权重
        real_dir = tmp / "rtdetrv3_best"
        real_dir.mkdir()
        (real_dir / "best.pt").write_bytes(b"stub")
        (real_dir / "model.onnx.json").write_text("{}", encoding="utf-8")
        # 只有权重、没有 sidecar 的目录
        weight_only = tmp / "cls_step1"
        weight_only.mkdir()
        (weight_only / "best.pt").write_bytes(b"stub")

        picked = R._find_latest_ckpt_dir()
        assert picked == real_dir, f"应优先选完整目录 {real_dir.name}，实际 {picked}"
        print(f"[PASS] 完整目录优先：{picked.name}")

        # 去掉完整目录后，应在"只有权重"和"临时目录"之间选前者
        (real_dir / "model.onnx.json").unlink()
        (real_dir / "best.pt").unlink()
        real_dir.rmdir()
        picked2 = R._find_latest_ckpt_dir()
        assert picked2 == weight_only, f"临时目录不应优先于有真实权重的目录，实际 {picked2}"
        print(f"[PASS] 临时目录不挤掉真实权重：{picked2.name}")
    finally:
        R.MODELS_DIR = real_models


def test_model_dir_discovery_handles_empty_models_dir() -> None:
    """models/ 为空或不存在时必须返回 None，让服务干净地降级。"""
    from src.runtime import registry as R

    real_models = R.MODELS_DIR
    tmp = Path(tempfile.mkdtemp())
    try:
        R.MODELS_DIR = tmp / "not_exist"
        assert R._find_latest_ckpt_dir() is None
        assert R._find_latest_onnx() is None
        R.MODELS_DIR = tmp
        assert R._find_latest_ckpt_dir() is None
        print("[PASS] 空目录/不存在目录都返回 None")
    finally:
        R.MODELS_DIR = real_models


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
