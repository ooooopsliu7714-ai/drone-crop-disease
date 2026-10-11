"""服务端快速自检：直接用 AlgorithmService 跑几张测试图，打印框数与主病害。

用途：确认"服务端返回的框数是否合理"。本机曾出现服务端几乎不返回框的情况——
模型分数整体偏低（最高 0.13~0.26）而默认阈值是 0.25，导致接口形同失效。
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import cv2

    from src.api.service import AlgorithmService
    from src.runtime.registry import build_predictor

    predictor, status = build_predictor(backend="torch")
    print(f"后端 = {status.backend}")
    print(f"置信度阈值 = {getattr(predictor, 'conf_threshold', None)}")
    print(f"模型版本 = {getattr(predictor, 'model_version', None)}")
    print("-" * 70)

    svc = AlgorithmService(predictor, status)
    files = sorted(glob.glob(str(ROOT / "data" / "processed_stress2" / "test" / "images" / "*.jpg")))[:4]
    if not files:
        print("[错误] 找不到测试图")
        return 2

    for f in files:
        img = cv2.imread(f)
        d = svc.predict_bgr(img, os.path.basename(f)).model_dump()
        print(
            "  {:26s} 主病害={:9s} 框数={:3d} 严重度={:9s} 主置信度={:.3f}".format(
                os.path.basename(f)[:24],
                d["disease_name"],
                len(d["boxes"]),
                d["severity"],
                d["confidence"],
            )
        )
    print("-" * 70)
    print("说明：框数应与标注密度同量级（本数据集约 13.8 框/图）。")
    print("      若框数为 0 或远低于该值，优先怀疑服务端置信度阈值过高。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
