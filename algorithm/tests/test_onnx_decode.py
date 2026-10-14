"""ONNX 后端解码测试：不需要安装 onnxruntime，用桩 session 验证坐标还原与契约。

为什么值得测：ONNX 路径是生产路径，最容易出错的地方是
  ① letterbox 填充后的坐标还原；② 归一化 cxcywh -> 原图像素 xyxy；
  ③ 分数/类别阈值；④ sidecar 元数据解析。
这些错误不会报异常，只会让框"整体偏移或缩小一半"，肉眼很难发现。
运行：python tests/test_onnx_decode.py
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

from src.infer import onnx_backend as ob  # noqa: E402
from src.labels import LABELS  # noqa: E402


class _StubSession:
    """冒充 onnxruntime.InferenceSession：run() 直接返回预设张量。"""

    def __init__(self, outputs: list[np.ndarray], output_names: list[str]) -> None:
        self._outputs = outputs
        self._names = output_names

    def get_outputs(self):  # noqa: ANN201
        class _O:
            def __init__(self, name: str) -> None:
                self.name = name

        return [_O(n) for n in self._names]

    def run(self, names, feed):  # noqa: ANN001, ANN201
        return self._outputs


def _make_backend(tmp: Path, sidecar: dict, outputs: list[np.ndarray], input_size=(640, 640)) -> ob.OnnxDetector:
    """绕过 __init__ 的依赖检查，构造一个用桩 session 的 OnnxDetector。"""
    onnx_path = tmp / "model.onnx"
    onnx_path.write_bytes(b"stub")  # 只占位，不真读
    sidecar.setdefault("input", {})
    sidecar["input"].setdefault("size", [input_size[1], input_size[0]])
    sidecar["input"].setdefault("name", "images")
    sidecar["classes"] = LABELS.to_sidecar_classes()
    onnx_path.with_suffix(".onnx.json").write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")

    det = ob.OnnxDetector.__new__(ob.OnnxDetector)  # 跳过 __init__ 里的 onnxruntime 检查
    det.onnx_path = onnx_path
    det.meta = json.loads(onnx_path.with_suffix(".onnx.json").read_text(encoding="utf-8"))
    det.model_version = "stub"
    det.score_threshold = 0.25
    det.iou_threshold = 0.45
    det.max_det = 300
    det.input_size = (int(sidecar["input"]["size"][1]), int(sidecar["input"]["size"][0]))
    det.input_name = sidecar["input"]["name"]
    det.mean = np.asarray([0, 0, 0], np.float32)
    det.std = np.asarray([255, 255, 255], np.float32)
    det.color = "RGB"
    det.task = sidecar.get("task", "detect")
    det.out_kind = sidecar["outputs"]["kind"]
    det.output_names = ["logits", "pred_boxes"] if det.out_kind == "detect_detr" else ["output0"]
    det.session = _StubSession(outputs, det.output_names)
    det.classes = sidecar["classes"]
    det._build_class_map()  # 与真实 __init__ 保持一致，否则 _info() 找不到 _class_map
    return det


def _second_class(labels) -> tuple[int, str]:  # noqa: ANN001
    """返回当前标签体系的第二个类别 (class_id, disease_name)。

    为什么需要：本仓库可切换 15 类病害体系与 2 类胁迫体系，
    测试里写死 "class 1 = rice_blast" 会让 2 类体系下的用例误报失败。
    """
    ordered = sorted(labels.classes, key=lambda c: c["class_id"])
    c = ordered[1] if len(ordered) > 1 else ordered[0]
    return int(c["class_id"]), str(c["disease_name"])


def test_detr_decode_restores_original_coordinates() -> None:
    """RT-DETR 原生输出：cxcywh 归一化到 **letterbox 画布**，图是 1280x720（非正方形）。

    关键点：归一化坐标的基准是 640x640 画布，不是原图。
    期望值必须从 letterbox 参数推导，不能直接按原图比例写——这正是本模块早期
    把框压缩到左上角的错误来源，所以测试要用真实 letterbox 参数反推。
    """
    from src.infer.onnx_backend import letterbox

    img = np.zeros((720, 1280, 3), np.uint8)
    # 用真实 letterbox 求 scale/pad，避免手算与实现不一致
    _, scale, pad_x, pad_y = letterbox(img, (640, 640))

    # 类别 id 从 sidecar 动态取（第二类），不要写死 1——
    # 2 类胁迫体系下只有 class 0/1，写死索引会 IndexError 或误判。
    # 画布上的框：中心 (320,320)、宽高各 64 -> 归一化 0.5,0.5,0.1,0.1
    cx, cy, bw, bh = 0.5, 0.5, 0.1, 0.1
    target_cid, target_name = _second_class(LABELS)
    logits = np.zeros((1, 1, LABELS.num_classes), np.float32)
    logits[0, 0, target_cid] = 5.0
    boxes = np.array([[[cx, cy, bw, bh]]], np.float32)
    det = _make_backend(
        Path(tempfile.mkdtemp()),
        {"task": "detect", "outputs": {"kind": "detect_detr", "names": ["logits", "pred_boxes"]}},
        [logits, boxes],
    )
    out = det.predict([img])[0]
    assert out["disease_name"] == target_name, out

    # 期望：画布坐标 -> 去填充 -> 除 scale
    def expect(v_canvas: float, pad: float) -> float:
        return (v_canvas - pad) / scale

    exp_x1 = expect(320 - 32, pad_x)
    exp_x2 = expect(320 + 32, pad_x)
    exp_y1 = expect(320 - 32, pad_y)
    exp_y2 = expect(320 + 32, pad_y)

    x1, y1, x2, y2 = out["bbox"]
    assert abs(x1 - exp_x1) < 1.5, f"x1 期望 {exp_x1}，实际 {x1}"
    assert abs(x2 - exp_x2) < 1.5, f"x2 期望 {exp_x2}，实际 {x2}"
    assert abs(y1 - exp_y1) < 1.5, f"y1 期望 {exp_y1}，实际 {y1}"
    assert abs(y2 - exp_y2) < 1.5, f"y2 期望 {exp_y2}，实际 {y2}"
    # 画布上 64x64 的框 -> 原图上 128x128（scale=0.5）
    assert abs((x2 - x1) - 128) < 2, f"宽度错误：{x2 - x1}"
    assert abs((y2 - y1) - 128) < 2, f"高度错误：{y2 - y1}"
    assert out["detections"], "检测任务应返回 detections 列表"
    assert out["detections"][0]["class_id"] == target_cid


def test_yolo_style_decode_and_threshold() -> None:
    """常见导出格式 (1,N,6)：xyxy + score + class，验证阈值过滤。"""
    img = np.zeros((640, 640, 3), np.uint8)
    target_cid, _ = _second_class(LABELS)
    other_cid = 0 if target_cid != 0 else min(1, LABELS.num_classes - 1)
    pred = np.array(
        [[[100, 100, 200, 200, 0.9, target_cid], [300, 300, 400, 400, 0.1, other_cid]]],
        np.float32,
    )
    det = _make_backend(
        Path(tempfile.mkdtemp()),
        {"task": "detect", "outputs": {"kind": "detect_xyxy_score_cls", "names": ["output0"]}},
        [pred],
    )
    out = det.predict([img])[0]
    dets = out["detections"]
    assert len(dets) == 1, f"0.1 分的框应被阈值 0.25 过滤，实际 {len(dets)}"
    assert dets[0]["class_id"] == target_cid, f"应保留 {target_cid} 类，实际 {dets[0]['class_id']}"
    assert abs(dets[0]["bbox"][0] - 100) < 2 and abs(dets[0]["bbox"][2] - 200) < 2


def test_classify_logits_decode() -> None:
    """分类任务：整图一个框，且给出 topk 供复核端使用。"""
    img = np.zeros((300, 400, 3), np.uint8)
    target_cid, target_name = _second_class(LABELS)
    logits = np.zeros((1, LABELS.num_classes), np.float32)
    logits[0, target_cid] = 8.0
    if target_cid != 0:
        logits[0, 0] = 2.0
    det = _make_backend(
        Path(tempfile.mkdtemp()),
        {"task": "classify", "outputs": {"kind": "logits", "names": ["logits"]}},
        [logits],
    )
    out = det.predict([img])[0]
    assert out["disease_name"] == target_name, out
    assert out["bbox"] == [0.0, 0.0, 400.0, 300.0]
    assert len(out["topk"]) >= 1 and out["topk"][0]["disease_name"] == target_name


def test_missing_sidecar_raises_model_unavailable() -> None:
    """缺 sidecar 时必须抛 ModelUnavailable，让 registry 降级而不是 500。

    注意：不走 OnnxDetector(...) 构造——那会先因为本机没装 onnxruntime 抛错，
    掩盖掉我们真正想验证的 sidecar 检查。这里直接调 sidecar 读取函数。
    """
    tmp = Path(tempfile.mkdtemp())
    onnx_path = tmp / "m.onnx"
    onnx_path.write_bytes(b"x")
    try:
        ob._read_sidecar(onnx_path)
    except ob.ModelUnavailable as exc:
        assert "元数据" in str(exc), str(exc)
    else:
        raise AssertionError("缺少 sidecar 应抛 ModelUnavailable")

    # sidecar 存在但不是合法 JSON 时，也要给出可读的中文错误
    onnx_path.with_suffix(".onnx.json").write_text("{ 不是 json", encoding="utf-8")
    try:
        ob._read_sidecar(onnx_path)
    except ob.ModelUnavailable as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("非法 sidecar 应抛 ModelUnavailable")


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
