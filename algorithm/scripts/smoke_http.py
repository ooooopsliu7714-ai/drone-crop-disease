"""真实 HTTP 冒烟：起 uvicorn 子进程，用 HTTP 请求打 /algo/predict 并测延迟。

为什么不能只用 TestClient：`fastapi.testclient.TestClient` 走的是**进程内 ASGI**，
不经过 socket、不经过 uvicorn 的中间件与生命周期钩子。它是有效的契约测试，
但**不能证明"服务真的能起来并被外部调用"**——端口占用、启动期模型加载失败、
启动事件没跑到之类的问题它都看不见。

用法：
    python scripts/smoke_http.py --port 8077
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _wait_ready(base: str, timeout: float = 180.0):
    import httpx

    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{base}/health", timeout=5.0)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = type(exc).__name__
        time.sleep(2.0)
    raise RuntimeError(f"服务在 {timeout:.0f}s 内未就绪（最后错误：{last}）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="真实 HTTP 冒烟测试")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--rounds", type=int, default=6, help="延迟测量的请求次数")
    ap.add_argument("--log", default="runs/http_smoke_server.log")
    args = ap.parse_args(argv)

    import httpx

    base = f"http://{args.host}:{args.port}"
    log_path = ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 注意：沙箱下不能给子进程用管道（named pipe 会被拒），所以输出重定向到文件
    logf = log_path.open("w", encoding="utf-8", errors="replace")
    env = dict(os.environ)
    env.setdefault("ALGO_LABELS_FILE", "configs/labels_stress2.yaml")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:app",
         "--host", args.host, "--port", str(args.port), "--log-level", "warning"],
        cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT, env=env,
    )
    print(f"[信息] uvicorn 已启动 pid={proc.pid}，日志 -> {log_path}")
    ok = True
    try:
        health = _wait_ready(base)
        print(f"[PASS] /health 就绪：backend={health.get('backend')} "
              f"model={health.get('model_version')} labels={health.get('labels_version')}")
        print(f"       {json.dumps(health, ensure_ascii=False)[:300]}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] /health 未就绪：{exc}")
        print("---- 服务日志尾部 ----")
        try:
            print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]))
        except Exception:
            pass
        proc.terminate()
        return 1

    # 找测试图：优先数据集目录，**回退到 data/uploads/**
    #
    # 为什么必须回退（本机实测）：交付包里不含数据集（体积原因），只带了 3 张示例图
    # 放在 data/uploads/。若只查数据集目录，接收方一运行就 IndexError，
    # 明明包是好的却以为交付失败。回退后"收到包即可自测"才成立。
    img_dir = ROOT / "data/processed_stress2/val/images"
    if not img_dir.is_dir():
        img_dir = ROOT / "data/uploads"
    samples = sorted(img_dir.glob("*.jpg"))[:3] if img_dir.is_dir() else []
    if not samples:
        print(f"[警告] 在 {img_dir} 找不到测试图，只做 /health 与错误路径检查")

    latency = []
    try:
        with httpx.Client(base_url=base, timeout=60.0) as c:
            # 1) JSON + image_id
            for p in samples:
                r = c.post("/algo/predict", json={"image_id": p.name})
                j = r.json()
                n = len(j.get("boxes") or [])
                print(f"[{'PASS' if r.status_code == 200 else 'FAIL'}] JSON  image_id={p.name[:26]:28s} "
                      f"HTTP {r.status_code}  {j.get('disease_name')} conf={j.get('confidence')} 框数={n}")
                ok &= r.status_code == 200 and isinstance(j.get("disease_name"), str) and n >= 0

            # 2) multipart 上传
            for p in samples[:2]:
                with p.open("rb") as fh:
                    r = c.post("/algo/predict", files={"file": (p.name, fh, "image/jpeg")})
                j = r.json()
                print(f"[{'PASS' if r.status_code == 200 else 'FAIL'}] multipart {p.name[:22]:24s} "
                      f"HTTP {r.status_code}  {j.get('disease_name')} 框数={len(j.get('boxes') or [])}")
                ok &= r.status_code == 200

            # 3) 错误路径：缺 image_id 应 4xx 而不是 5xx
            r = c.post("/algo/predict", json={"no_image_id": 1})
            print(f"[{'PASS' if r.status_code < 500 else 'FAIL'}] 缺字段 -> HTTP {r.status_code}（应 4xx，不能 5xx）")
            ok &= r.status_code < 500

            # 4) 延迟（没有测试图时跳过，避免 IndexError 把整个自测判成失败）
            if samples:
                p = samples[0]
                for _ in range(args.rounds):
                    t0 = time.perf_counter()
                    c.post("/algo/predict", json={"image_id": p.name})
                    latency.append((time.perf_counter() - t0) * 1000)
                print(f"[信息] 端到端延迟（{len(latency)} 次）：中位 {statistics.median(latency):.0f} ms  "
                      f"最小 {min(latency):.0f} ms  最大 {max(latency):.0f} ms")
            else:
                print("[跳过] 没有测试图，未测延迟")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 请求过程异常：{type(exc).__name__}: {exc}")
        ok = False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        logf.close()

    print("\n" + "=" * 70)
    print("✓ 真实 HTTP 冒烟通过（uvicorn + JSON + multipart + 错误路径 + 延迟）" if ok
          else "✗ 真实 HTTP 冒烟有问题，见上")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
