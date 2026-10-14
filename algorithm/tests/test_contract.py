"""接口契约测试：确保 /algo/predict 的字段名、坐标范围、降级行为都不会被改坏。

运行（在项目根目录）：
    python -m pytest tests -q          # 有 pytest 时
    python tests/test_contract.py      # 无 pytest 也能跑（内置简易断言）

为什么这份测试重要：算法服务的消费方是 Java 后端 + 小程序 + Apifox mock，
字段名一旦漂移，联调就会无声地失败。这里把契约钉死。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.api.schema import compute_statistics, top_disease  # noqa: E402
from src.api.service import AlgorithmService  # noqa: E402
from src.infer.postprocess import class_wise_nms, iou_matrix, tile_origins  # noqa: E402
from src.labels import LABELS  # noqa: E402
from src.runtime.registry import build_predictor  # noqa: E402

# 契约要求必须存在的顶层字段（来自 Apifox 文档）
REQUIRED_TOP_FIELDS = {
    "code",
    "message",
    "image_id",
    "disease_name",
    "disease_name_cn",
    "confidence",
    "severity",
    "boxes",
    "processing_time",
}
REQUIRED_BOX_FIELDS = {
    "disease_name",
    "disease_name_cn",
    "confidence",
    "severity",
    "class_id",
    "x",
    "y",
    "w",
    "h",
    "x1",
    "y1",
    "x2",
    "y2",
}


def synthetic_field(shape=(720, 960), diseased=True) -> np.ndarray:
    """合成一张"农田图"：绿色植被底 + 黄化/褐变斑块。"""
    h, w = shape
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 1] = 150
    img[:, :, 0] = 60
    img[:, :, 2] = 70
    if diseased:
        img[h // 4 : h // 2, w // 4 : w // 2] = (60, 180, 200)     # 黄化
        img[h // 2 : h // 2 + h // 6, w // 2 : w // 2 + w // 4] = (25, 45, 70)  # 褐变
    return img


def _service() -> AlgorithmService:
    pred, status = build_predictor(backend="heuristic")
    return AlgorithmService(pred, status)


# ---------------------------------------------------------------------------
def test_contract_fields_present() -> None:
    svc = _service()
    d = svc.predict_bgr(synthetic_field(), "test_001").model_dump()
    missing = REQUIRED_TOP_FIELDS - set(d)
    assert not missing, f"响应缺少契约字段：{missing}"
    assert d["code"] == 200
    assert 0.0 <= d["confidence"] <= 1.0
    assert d["severity"] in LABELS.severity_keys
    for b in d["boxes"]:
        miss = REQUIRED_BOX_FIELDS - set(b)
        assert not miss, f"boxes[] 缺少契约字段：{miss}"
        assert 0.0 <= b["x"] <= 1.0 and 0.0 <= b["y"] <= 1.0
        assert 0.0 <= b["w"] <= 1.0 and 0.0 <= b["h"] <= 1.0
        assert b["x1"] <= b["x2"] and b["y1"] <= b["y2"]
        assert b["label"] == b["disease_name"], "兼容字段 label 必须等于 disease_name"
        assert abs(b["score"] - b["confidence"]) < 1e-9, "兼容字段 score 必须等于 confidence"
        assert len(b["bbox"]) == 4


def test_healthy_image_reports_healthy() -> None:
    """健康判定的稳定口径：disease_name == "healthy" 且 severity == "healthy"。

    注意 boxes 的形状会随后端变化（分类后端给空列表、检测/基线后端给一个全图框），
    所以**前端必须用 disease_name 判断健康**，不能用 len(boxes)==0。
    这条口径同时写在 docs/algorithm-design.md §6.1。
    """
    svc = _service()
    d = svc.predict_bgr(synthetic_field(diseased=False), "test_healthy").model_dump()
    assert d["disease_name"] == "healthy"
    assert d["severity"] == "healthy"
    assert d["statistics"]["diseased_ratio"] == 0.0


def test_diseased_image_detects_and_advises() -> None:
    svc = _service()
    d = svc.predict_bgr(synthetic_field(), "test_diseased").model_dump()
    assert d["disease_name"] != "healthy"
    assert d["severity"] in ("light", "moderate", "severe")
    assert d["statistics"]["disease_index"] > 0
    assert d["advice"]["priority"] in ("immediate", "within_3_days", "observe", "none")


def test_response_is_json_serializable() -> None:
    svc = _service()
    d = svc.predict_bgr(synthetic_field(), "test_json").model_dump()
    text = json.dumps(d, ensure_ascii=False)
    assert "disease_name" in text and "confidence" in text


def test_sliced_inference_on_large_image() -> None:
    """大图必须走切片推理：缩到 640 会丢小病斑，所以这里验证切片路径能跑通。"""
    import gc

    svc = _service()
    big = np.tile(synthetic_field(shape=(700, 700)), (3, 3, 1))  # 2100x2100 > slice_threshold
    try:
        d = svc.predict_bgr(big, "test_big").model_dump()
    finally:
        del big
        gc.collect()  # 测试机内存紧张（可用内存 <3GB），主动回收避免 OpenCV 分配失败
    assert d["code"] == 200
    for b in d["boxes"]:
        assert 0 <= b["x1"] <= 2100 and 0 <= b["x2"] <= 2100


def test_severity_and_stats_math() -> None:
    from src.api.schema import bbox_to_out

    boxes = [
        bbox_to_out(
            disease_name="rice_blast",
            disease_name_cn="稻瘟病",
            confidence=0.9,
            severity="severe",
            class_id=1,
            xyxy=[0, 0, 100, 100],
            image_w=1000,
            image_h=1000,
        )
    ]
    st = compute_statistics(boxes, 1000, 1000)
    assert abs(st.diseased_ratio - 0.01) < 1e-9
    # 病情指数 = 3级 * 1% 面积 / (3 * 100%) * 100 = 1.0
    assert abs(st.disease_index - 1.0) < 1e-6
    name, cn, conf, sev = top_disease(boxes)
    assert (name, cn, sev) == ("rice_blast", "稻瘟病", "severe") and conf == 0.9


def test_nms_merges_duplicate_boxes() -> None:
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
    scores = [0.9, 0.8, 0.7]
    classes = [1, 1, 2]
    keep = class_wise_nms(boxes, scores, classes, iou_threshold=0.5)
    assert len(keep) == 2, "重叠的两个同类框应只留一个"


def test_iou_matrix_and_tiles() -> None:
    import numpy as np_

    a = np_.array([[0, 0, 10, 10]], dtype=np_.float32)
    b = np_.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np_.float32)
    m = iou_matrix(a, b)
    assert abs(m[0, 0] - 1.0) < 1e-6 and abs(m[0, 1]) < 1e-6
    tiles = tile_origins(1000, 800, tile=640, overlap=128)
    assert tiles and all(t[2] <= 1000 and t[3] <= 800 for t in tiles)
    # 必须覆盖到右边界与下边界
    assert max(t[2] for t in tiles) == 1000
    assert max(t[3] for t in tiles) == 800


def test_missing_image_id_returns_404() -> None:
    svc = _service()
    try:
        svc.predict_by_id("not_exist_12345")
    except FileNotFoundError as exc:
        assert "not_exist_12345" in str(exc)
    else:
        raise AssertionError("不存在的 image_id 应抛 FileNotFoundError -> 由接口层转 404")


def test_two_label_implementations_agree() -> None:
    """训练侧(src.core.labels) 与推理侧(src.labels) 必须读出同一套类别与严重度。

    这两个模块各自解析 configs/labels.yaml（训练侧有自己的 LabelBook 便于训练脚本使用），
    一旦解析口径漂移，就会出现"训练用 class_id=5 是条锈病、推理当作别的病"这种
    极难排查的错误。这里把一致性钉死。
    """
    from src.core import labels as core_labels

    book = core_labels.load_labels()
    assert book.names == LABELS.names, (
        f"类别顺序不一致：\n  推理侧 {LABELS.names}\n  训练侧 {book.names}"
    )
    assert list(core_labels.SEVERITIES) == LABELS.severity_keys
    for s in LABELS.severity_levels:
        assert core_labels.SEVERITY_INDEX[s["key"]] == int(s["severity_index"])
    # 严重度换算规则也必须一致（同为面积占比口径）
    for ratio in (0.0, 0.01, 0.05, 0.2, 0.5):
        assert core_labels.severity_from_area_ratio(ratio) == LABELS.severity_from_ratio(ratio), (
            f"比例 {ratio} 的严重度分级不一致"
        )


def test_result_stable_across_slice_settings() -> None:
    """同一张图在不同切片配置下，**主病害与严重度必须一致**。

    这条是切片推理最容易出的静默错误，早期版本踩过两次：
    ① 切片内的局部严重度被当成整块地严重度；
    ② 干净切片返回的 healthy 占位框（置信度 0.9）把真实病斑挤成主病害。
    两者都会让"换个切片尺寸结论就变了"。

    适用范围说明（重要）：本测试只用**单一症状类型**的图。
    规则基线每个切片只报一个"主症状"，当一张图里同时存在黄化与坏死病斑、
    且病斑尺寸与切片尺寸同量级时，某片可能只看到坏死、另一片只看到黄化，
    不同切片配置的主症状就可能不同；真实检测模型逐病斑输出、不受此限制，
    因此这里不去强求规则基线在所有切片配置下的多症状一致性。
    """
    import os

    from src.runtime.registry import build_predictor as bp

    # 绿色底色 + 一块黄化病斑（单一症状），病斑明显大于切片以覆盖"占满切片"的情形
    img = np.zeros((1200, 1200, 3), np.uint8)
    img[:, :, 1] = 150
    img[:, :, 0] = 60
    img[:, :, 2] = 70
    img[300:800, 300:800] = (60, 180, 200)

    configs = [("1600", "640", "128"), ("800", "512", "64"), ("1000000000", "640", "128")]
    saved = {k: os.environ.get(k) for k in ("ALGO_SLICE_THRESHOLD", "ALGO_TILE", "ALGO_OVERLAP")}
    outcomes = []
    try:
        for th, tile, ov in configs:
            os.environ["ALGO_SLICE_THRESHOLD"] = th
            os.environ["ALGO_TILE"] = tile
            os.environ["ALGO_OVERLAP"] = ov
            pred, status = bp(backend="heuristic")
            svc = AlgorithmService(pred, status)
            d = svc.predict_bgr(img, f"slice_{th}").model_dump()
            outcomes.append((d["disease_name"], d["severity"]))
            # 健康占位框不允许出现在结果里（否则前端会画一堆"健康块"）
            assert all(b["disease_name"] != "healthy" for b in d["boxes"]), (
                f"切片配置 {th}/{tile} 下结果里混入了 healthy 占位框：{d['boxes'][:2]}"
            )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    assert len(set(outcomes)) == 1, f"不同切片配置下结论不一致：{outcomes}"
    assert outcomes[0][0] != "healthy", f"带病图不应判为健康：{outcomes}"
    # 规则基线只做"症状族"判别，会映射到当前标签体系里真实存在的类别：
    # 15 类病害体系下是 wheat_stripe_rust，2 类胁迫体系下是 stressed。
    # 所以这里校验"落在当前类别表内"，而不是写死某个病名（否则换数据集就误报）。
    assert outcomes[0][0] in LABELS.names, (
        f"主病害 {outcomes[0][0]!r} 不在当前类别表 {LABELS.names} 中"
    )


def test_http_layer_two_request_modes() -> None:
    """用 FastAPI TestClient 验证 JSON 与 multipart 两种调用方式。

    TestClient 依赖 httpx；缺失时跳过（不阻塞其它测试）。
    """
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:  # pragma: no cover
        print(f"[skip] TestClient 不可用：{exc}")
        return

    from src.api.app import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

    ok, buf = cv2.imencode(".jpg", synthetic_field())
    assert ok
    r = client.post(
        "/algo/predict",
        files={"file": ("field.jpg", buf.tobytes(), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert REQUIRED_TOP_FIELDS <= set(payload), payload.keys()

    # JSON 模式：先不存在的 image_id -> 404；再把图写进 uploads 目录后应 200
    r404 = client.post("/algo/predict", json={"image_id": "definitely_missing"})
    assert r404.status_code == 404, r404.text

    from src.runtime.registry import UPLOADS_DIR

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    name = "contract_test_field.jpg"
    (UPLOADS_DIR / name).write_bytes(buf.tobytes())
    r200 = client.post("/algo/predict", json={"image_id": name})
    assert r200.status_code == 200, r200.text
    assert REQUIRED_TOP_FIELDS <= set(r200.json())

    rbad = client.post("/algo/predict", json={"no_image_id": 1})
    assert rbad.status_code == 422, rbad.text


# ---------------------------------------------------------------------------
def _run_all() -> int:
    """无 pytest 时的简易执行器。"""
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
