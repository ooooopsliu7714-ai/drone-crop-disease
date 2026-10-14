"""一键冒烟验证：检查环境 -> 起服务逻辑 -> 跑契约测试 -> 打印结果。

用途：换电脑、重新装依赖、答辩前 5 分钟的自检。
用法（项目根目录）：
    python scripts/verify_all.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, str, str]] = []  # (名称, 状态, 说明)


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, "OK" if ok else "MISSING", detail))


def main() -> int:
    print("=" * 72)
    print("无人机农作物病害识别 —— 算法侧环境自检")
    print("=" * 72)
    print(f"Python: {sys.version.split()[0]} ({platform.machine()})")
    print(f"项目根: {ROOT}\n")

    # ---- 依赖 ----
    for mod, why in [
        ("numpy", "数值计算（必需）"),
        ("cv2", "图像读写与预处理（必需）"),
        ("yaml", "读取 configs/labels.yaml（必需）"),
        ("fastapi", "HTTP 接口（必需）"),
        ("uvicorn", "启动服务（必需）"),
        ("pydantic", "契约模型（必需）"),
        ("torch", "训练与 PyTorch 推理后端"),
        ("transformers", "RT-DETR / DINOv2-v3 官方实现"),
        ("onnxruntime", "ONNX 部署推理（生产推荐）"),
        ("timm", "DINOv 骨干网络可选实现"),
        ("albumentations", "训练期数据增强"),
        ("ultralytics", "RT-DETR / YOLO-World 训练后端（可选）"),
        ("openpyxl", "统计报表导出 Excel"),
    ]:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            check(f"依赖 {mod}", True, f"{ver} — {why}")
        except Exception:
            check(f"依赖 {mod}", False, f"未安装 — {why}")

    # ---- GPU ----
    try:
        import torch

        if torch.cuda.is_available():
            check("CUDA", True, f"{torch.cuda.get_device_name(0)} / torch {torch.__version__}")
        else:
            check(
                "CUDA",
                False,
                f"torch {torch.__version__} 为 CPU 版，GPU 训练不可用。"
                "装 cu128 版本：pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision",
            )
    except Exception as exc:
        check("CUDA", False, f"torch 不可用：{exc}")

    # ---- 标签体系 ----
    try:
        from src.labels import LABELS

        check("标签体系", LABELS.num_classes > 0, f"{LABELS.num_classes} 个类别，严重度 {LABELS.severity_keys}")
    except Exception as exc:
        check("标签体系", False, str(exc))

    # ---- 后端 ----
    try:
        from src.runtime.registry import build_predictor

        pred, status = build_predictor()
        check("推理后端", True, f"{status.backend} (loaded={status.loaded}, device={status.device})")
        for note in status.notes:
            print(f"    · {note}")
    except Exception as exc:
        check("推理后端", False, str(exc))

    # ---- 模型文件 ----
    models = list((ROOT / "models").rglob("*.onnx")) if (ROOT / "models").exists() else []
    check("ONNX 模型", bool(models), f"找到 {len(models)} 个" if models else "models/ 下暂无 .onnx（未训练前正常）")

    # ---- 端到端一次推理 ----
    try:
        import numpy as np

        from src.api.service import AlgorithmService

        pred, status = build_predictor()
        svc = AlgorithmService(pred, status)
        img = np.zeros((720, 960, 3), np.uint8)
        img[:, :, 1] = 150
        img[:, :, 0] = 60
        img[:, :, 2] = 70
        img[180:360, 240:480] = (60, 180, 200)
        t0 = time.perf_counter()
        resp = svc.predict_bgr(img, "verify_smoke")
        dt = time.perf_counter() - t0
        payload = resp.model_dump()
        required = {
            "code", "message", "image_id", "disease_name", "disease_name_cn",
            "confidence", "severity", "boxes", "processing_time",
        }
        ok = required <= set(payload)
        check("端到端推理", ok, f"{dt*1000:.0f} ms, disease={payload['disease_name']}, severity={payload['severity']}")
        print("    响应样例：")
        print(
            "      "
            + json.dumps(
                {
                    "code": payload["code"],
                    "image_id": payload["image_id"],
                    "disease_name": payload["disease_name"],
                    "confidence": payload["confidence"],
                    "severity": payload["severity"],
                    "boxes": len(payload["boxes"]),
                    "processing_time": payload["processing_time"],
                },
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        check("端到端推理", False, str(exc))

    # ---- 汇总 ----
    print("\n" + "-" * 72)
    missing = [c for c in CHECKS if c[1] == "MISSING"]
    for name, st, detail in CHECKS:
        mark = "✓" if st == "OK" else "✗"
        print(f"  {mark} {name:16s} {detail}")
    print("-" * 72)
    if missing:
        print(f"有 {len(missing)} 项未就绪。CPU 联调通常只需前 6 项 + 端到端推理。")
        print("要跑真实训练，请先按 REQUIREMENTS.md 装 GPU 版 torch 与训练依赖。")
    else:
        print("全部就绪。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
