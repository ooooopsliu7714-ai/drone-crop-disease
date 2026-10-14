"""一键跑全部离线测试（不依赖 pytest）。

用法（项目根目录）：
    python tests/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = [
    "test_contract.py",
    "test_det_targets.py",
    "test_onnx_decode.py",
    "test_sidecar_contract.py",
    "test_predictor_contract.py",
]


def main() -> int:
    failed: list[str] = []
    for name in SUITES:
        path = HERE / name
        print("=" * 72)
        print(f"运行 {name}")
        print("=" * 72)
        code = subprocess.call([sys.executable, str(path)], cwd=str(HERE.parent))
        if code != 0:
            failed.append(name)
    print("=" * 72)
    if failed:
        print(f"失败的测试文件：{failed}")
        return 1
    print(f"全部通过（{len(SUITES)} 个测试文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
