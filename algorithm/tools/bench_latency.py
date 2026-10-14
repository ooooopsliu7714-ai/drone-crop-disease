#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HTTP 接口压测工具（POST /algo/predict 性能证据，答辩用）。

【它解决什么】
答辩和评审都会问"你这套东西到底多快、能不能上生产"。本工具对算法服务
`POST /algo/predict` 打真实 HTTP 请求，测出 p50 / p95 / 平均耗时、QPS、成功率，
并采集显存与内存占用，输出 JSON（机器可读）+ 中文报告（直接贴 PPT）。

【接口契约（与 configs/labels.yaml 完全一致，本工具只做客户端，不改契约）】
  请求 A（JSON）      : {"image_id": "abc123"}
  请求 B（文件上传）  : multipart/form-data, 字段名 file
  响应               : code / message / image_id / disease_name / disease_name_cn /
                       confidence / severity / boxes / processing_time

【为什么同时记录两种耗时】
  latency_ms      —— 客户端测的端到端耗时（含网络、排队、序列化），决定用户体验；
  server_ms       —— 响应体 processing_time 字段，服务端自报的处理耗时，只含模型推理；
  两者差值是「网络 + 排队 + 调度」开销，答辩时用来解释 QPS 上不去的原因。

【依赖降级策略】
  - HTTP 客户端：优先 `requests`（本机已装 2.32.5），未装则降级到标准库 urllib/http.client；
  - 内存：优先 `psutil`（本机已装 7.2.2），未装则用 ctypes 读本进程 RSS，再不行只报系统内存；
  - 显存：调用 `nvidia-smi`；无 NVIDIA GPU 或命令缺失时跳过并提示，不影响整体压测。

【用法示例】
  python tools/bench_latency.py --help

  # 1) 服务已在 127.0.0.1:8000 跑着，按 image_id 压测（推荐，排除上传带宽影响）
  python tools/bench_latency.py --url http://127.0.0.1:8000/algo/predict \\
      --image-id demo_001 --n 100 --warmup 10 --concurrency 4 --out runs/bench

  # 2) 直接上传图片文件（测端到端含上传）
  python tools/bench_latency.py --url http://127.0.0.1:8000/algo/predict \\
      --image data/demo/field.jpg --n 50 --concurrency 2 --out runs/bench

  # 3) 先做冒烟测试确认服务活着（1 次请求，打印完整响应）
  python tools/bench_latency.py --url http://127.0.0.1:8000/algo/predict \\
      --image-id demo_001 --n 1 --warmup 0 --smoke

【输出】
  <out>/bench_latency.json  原始数据 + 汇总指标（每请求一条明细）
  <out>/bench_latency.md    中文报告（含目标达成对照）

【性能目标（写进答辩的验收线）】
  单张 640 图：CPU ≤ 1500 ms，GPU ≤ 200 ms；4096 大图切片推理 ≤ 30 s。
  本工具会把实测值与这些目标自动对照并在报告里标 ✅/❌。
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# ---- 可选依赖：requests / psutil ----
try:
    import requests

    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

try:
    import psutil

    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


# --------------------------------------------------------------------------
# 资源采集
# --------------------------------------------------------------------------
def read_rss_mb() -> Optional[float]:
    """读当前进程 RSS（MB）。优先 psutil，缺失时用 ctypes 读 Windows 进程内存。"""
    if HAS_PSUTIL:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / 1024.0 / 1024.0
        except Exception:
            pass
    try:  # Windows 下用 ctypes 兜底，不依赖任何第三方库
        import ctypes
        import ctypes.wintypes as wt

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if ok:
            return counters.WorkingSetSize / 1024.0 / 1024.0
    except Exception:
        pass
    return None


def read_system_mem() -> Dict[str, Any]:
    """系统内存概况。psutil 缺失时退回 os 提供的有限信息并给出提示。"""
    if HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            return {
                "source": "psutil",
                "total_mb": round(vm.total / 1024 / 1024, 1),
                "used_mb": round((vm.total - vm.available) / 1024 / 1024, 1),
                "percent": float(vm.percent),
            }
        except Exception:
            pass
    return {
        "source": "unavailable",
        "note": "未安装 psutil，无法读取系统内存占用；本进程 RSS 已用 ctypes 兜底。"
                "安装后可获得完整数据：pip install psutil",
    }


def read_gpu_mem() -> Optional[Dict[str, Any]]:
    """用 nvidia-smi 读整卡显存与利用率。无 GPU / 无驱动时返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        first = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) < 4:
            return None
        return {
            "name": parts[0],
            "memory_used_mb": float(parts[1]),
            "memory_total_mb": float(parts[2]),
            "utilization_percent": float(parts[3]),
        }
    except Exception:
        return None


class ResourceSampler:
    """后台线程采样资源占用，给出峰值与均值。"""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append({
                "t": time.time(),
                "rss_mb": read_rss_mb(),
                "gpu": read_gpu_mem(),
            })
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._th:
            self._th.join(timeout=2.0)
        rss = [s["rss_mb"] for s in self.samples if s.get("rss_mb") is not None]
        gpu_used = [s["gpu"]["memory_used_mb"] for s in self.samples
                    if s.get("gpu") and s["gpu"].get("memory_used_mb") is not None]
        gpu_util = [s["gpu"]["utilization_percent"] for s in self.samples
                    if s.get("gpu") and s["gpu"].get("utilization_percent") is not None]
        return {
            "samples": len(self.samples),
            "rss_mb": {
                "min": round(min(rss), 1) if rss else None,
                "max": round(max(rss), 1) if rss else None,
                "mean": round(statistics.fmean(rss), 1) if rss else None,
                "note": "本进程（压测客户端）RSS；服务端若为独立进程请另行监控",
            },
            "gpu_memory_used_mb": {
                "min": round(min(gpu_used), 1) if gpu_used else None,
                "max": round(max(gpu_used), 1) if gpu_used else None,
                "mean": round(statistics.fmean(gpu_used), 1) if gpu_used else None,
                "note": "整卡显存占用（含其他程序），需与 baseline 对比看增量",
            },
            "gpu_utilization_percent": {
                "max": round(max(gpu_util), 1) if gpu_util else None,
                "mean": round(statistics.fmean(gpu_util), 1) if gpu_util else None,
            },
        }


# --------------------------------------------------------------------------
# 请求构造
# --------------------------------------------------------------------------
def build_multipart(image_path: str, field: str = "file") -> Tuple[bytes, str]:
    """手工拼 multipart/form-data 请求体，避免依赖 requests。"""
    boundary = "----dshbench" + uuid.uuid4().hex
    fname = os.path.basename(image_path)
    ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    with open(image_path, "rb") as f:
        payload = f.read()
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{fname}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + payload + tail, f"multipart/form-data; boundary={boundary}"


def do_request(
    url: str,
    image_path: Optional[str],
    image_id: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    """发一次请求，返回 {ok, status, latency_ms, server_ms, code, body_head, error}。"""
    t0 = time.perf_counter()
    rec: Dict[str, Any] = {"ok": False, "status": None, "latency_ms": None,
                           "server_ms": None, "code": None, "error": None}
    try:
        if image_path:  # ---- 文件上传 ----
            body, ctype = build_multipart(image_path)
            if HAS_REQUESTS:
                r = requests.post(url, data=body, headers={"Content-Type": ctype}, timeout=timeout)
                status, text = r.status_code, r.text
            else:
                req = urllib.request.Request(url, data=body, method="POST",
                                             headers={"Content-Type": ctype})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    status, text = resp.status, resp.read().decode("utf-8", "replace")
        else:  # ---- JSON image_id ----
            payload = json.dumps({"image_id": image_id or "bench"}).encode("utf-8")
            if HAS_REQUESTS:
                r = requests.post(url, data=payload,
                                  headers={"Content-Type": "application/json"}, timeout=timeout)
                status, text = r.status_code, r.text
            else:
                req = urllib.request.Request(url, data=payload, method="POST",
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    status, text = resp.status, resp.read().decode("utf-8", "replace")
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        rec["status"] = status
        try:
            data = json.loads(text)
            rec["code"] = data.get("code")
            sm = data.get("processing_time")
            rec["server_ms"] = round(float(sm) * 1000.0, 3) if sm is not None else None
            rec["body_head"] = {
                k: data.get(k) for k in ("code", "message", "image_id", "disease_name",
                                         "disease_name_cn", "confidence", "severity")
            }
            rec["n_boxes"] = len(data.get("boxes") or [])
            rec["ok"] = (status == 200 and data.get("code") == 200)
            if not rec["ok"]:
                rec["error"] = f"HTTP {status} / code={data.get('code')} message={data.get('message')}"
        except json.JSONDecodeError:
            rec["error"] = f"响应不是合法 JSON（HTTP {status}）：{text[:120]!r}"
    except urllib.error.HTTPError as e:
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        rec["status"] = e.code
        rec["error"] = f"HTTPError {e.code}: {e.reason}"
    except Exception as e:
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def pct(values: Sequence[float], q: float) -> Optional[float]:
    """百分位数（线性插值），与 numpy.percentile 口径一致。"""
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return round(xs[0], 3)
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return round(xs[lo] * (1 - frac) + xs[hi] * frac, 3)


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def build_markdown(res: dict) -> str:
    """中文压测报告。"""
    s = res["summary"]
    lat = res["latency_client_ms"]
    srv = res["latency_server_ms"]
    L: List[str] = []
    L.append("# /algo/predict 接口压测报告")
    L.append("")
    L.append(f"- 目标地址：`{res['url']}`")
    L.append(f"- 请求方式：**{res['request_mode']}**"
             f"{'（' + os.path.basename(res['image']) + '）' if res.get('image') else ''}")
    L.append(f"- 并发：**{res['concurrency']}**，预热 {res['warmup']} 次，"
             f"正式压测 **{res['n']}** 次，超时 {res['timeout_s']}s")
    L.append(f"- HTTP 客户端：**{res['http_client']}**，资源采集：**{res['mem_backend']}**")
    L.append(f"- 压测时间：{res['started_at']}")
    L.append("")
    L.append("## 一、核心结论")
    L.append("")
    L.append("| 指标 | 数值 |")
    L.append("| --- | ---: |")
    L.append(f"| 成功率 | **{s['success_rate']:.2%}**（{s['success']}/{s['total']}） |")
    L.append(f"| QPS（吞吐） | **{s['qps']:.2f}** req/s |")
    L.append(f"| 单张平均耗时（客户端） | **{lat['mean']} ms** |")
    L.append(f"| p50（客户端） | {lat['p50']} ms |")
    L.append(f"| p95（客户端） | **{lat['p95']} ms** |")
    L.append(f"| p99（客户端） | {lat['p99']} ms |")
    L.append(f"| 最小 / 最大 | {lat['min']} / {lat['max']} ms |")
    L.append(f"| 标准差 | {lat['stdev']} ms |")
    if srv.get("count"):
        L.append(f"| 服务端自报 processing_time 均值 | {srv['mean']} ms |")
        L.append(f"| 服务端 p95 | {srv['p95']} ms |")
        L.append(f"| 网络+排队开销（客户端均值 − 服务端均值） | {s['overhead_ms']} ms |")
    L.append("")
    L.append("## 二、性能目标对照")
    L.append("")
    L.append("| 场景 | 目标 | 实测 | 结论 |")
    L.append("| --- | ---: | ---: | --- |")
    for row in res["targets"]:
        L.append(f"| {row['name']} | {row['target']} | {row['measured']} | {row['verdict']} |")
    L.append("")
    L.append("## 三、资源占用")
    L.append("")
    rss = res["resources"]["rss_mb"]
    gpu = res["resources"].get("gpu_memory_used_mb") or {}
    L.append(f"- 压测进程 RSS：min {rss['min']} / mean {rss['mean']} / max {rss['max']} MB")
    if gpu.get("max") is not None:
        L.append(f"- 整卡显存占用：min {gpu['min']} / mean {gpu['mean']} / max {gpu['max']} MB"
                 f"（baseline {res['resources'].get('gpu_baseline_mb')} MB，"
                 f"增量 ≤ {res['resources'].get('gpu_delta_mb')} MB）")
    else:
        L.append("- 整卡显存：未采集到（无 NVIDIA GPU 或 nvidia-smi 不可用）")
    sysmem = res["resources"].get("system_memory") or {}
    if sysmem.get("percent") is not None:
        L.append(f"- 系统内存：{sysmem['used_mb']} / {sysmem['total_mb']} MB"
                 f"（占用 {sysmem['percent']}%）")
    elif sysmem.get("note"):
        L.append(f"- 系统内存：{sysmem['note']}")
    L.append("")
    notes = res["resource_notes"]
    if notes:
        L.append("## 四、采集口径说明")
        L.append("")
        for n in notes:
            L.append(f"- {n}")
        L.append("")
    if res.get("failures"):
        L.append("## 五、失败样本（前 5 条）")
        L.append("")
        L.append("| # | 耗时(ms) | HTTP | 错误 |")
        L.append("| ---: | ---: | ---: | --- |")
        for i, f in enumerate(res["failures"][:5], 1):
            L.append(f"| {i} | {f.get('latency_ms')} | {f.get('status')} | {f.get('error')} |")
        L.append("")
    L.append("---")
    L.append("原始逐请求数据见同目录 `bench_latency.json`。")
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="bench_latency.py",
        description="对 POST /algo/predict 做压测，输出 p50/p95/均值/QPS/成功率 与显存内存占用。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python tools/bench_latency.py --url http://127.0.0.1:8000/algo/predict "
               "--image-id demo_001 --n 100 --concurrency 4\n"
               "  python tools/bench_latency.py --url http://127.0.0.1:8000/algo/predict "
               "--image field.jpg --n 50 --smoke\n",
    )
    p.add_argument("--url", default="http://127.0.0.1:8000/algo/predict",
                   help="接口地址（默认 http://127.0.0.1:8000/algo/predict）")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-id", help="JSON 模式：请求体 {\"image_id\": ...}，排除上传带宽影响")
    g.add_argument("--image", help="上传模式：multipart/form-data 字段 file")
    p.add_argument("--n", type=int, default=50, help="正式压测请求数（默认 50）")
    p.add_argument("--warmup", type=int, default=5,
                   help="预热请求数，不计入统计（默认 5，避开首次加载/CUDA 初始化）")
    p.add_argument("--concurrency", type=int, default=1, help="并发线程数（默认 1，串行）")
    p.add_argument("--timeout", type=float, default=60.0, help="单请求超时秒数（默认 60）")
    p.add_argument("--out", default="runs/bench", help="输出目录（默认 runs/bench）")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                   help="被测服务跑在 CPU 还是 GPU 上，用于选择对照目标（默认 cpu）")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟测试：只发 1 次并打印完整响应体，确认服务可用")
    p.add_argument("--quiet", action="store_true", help="安静模式")
    args = p.parse_args(argv)

    if args.n <= 0:
        print("[错误] --n 必须为正整数。", file=sys.stderr)
        return 2
    if args.concurrency <= 0:
        print("[错误] --concurrency 必须为正整数。", file=sys.stderr)
        return 2
    if args.image and not os.path.isfile(args.image):
        print(f"[错误] 图片不存在：{args.image}", file=sys.stderr)
        return 2

    http_client = "requests" if HAS_REQUESTS else "urllib(标准库)"
    mem_backend = "psutil" if HAS_PSUTIL else "ctypes/系统兜底"
    if not HAS_REQUESTS and not args.quiet:
        print("[提示] 未安装 requests，已降级为标准库 urllib 实现；"
              "安装可获得更稳定的连接复用：pip install requests", file=sys.stderr)
    if not HAS_PSUTIL and not args.quiet:
        print("[提示] 未安装 psutil，系统内存与显存采集能力受限；"
              "安装后可获得完整资源数据：pip install psutil", file=sys.stderr)

    request_mode = "文件上传 multipart/form-data" if args.image else "JSON image_id"
    if not args.quiet:
        print(f"[信息] 目标 {args.url}")
        print(f"[信息] 模式 {request_mode}，预热 {args.warmup}，压测 {args.n}，"
              f"并发 {args.concurrency}，HTTP 客户端 {http_client}")

    # ---- 冒烟测试 ----
    if args.smoke:
        rec = do_request(args.url, args.image, args.image_id, args.timeout)
        print("=" * 68)
        print(f"[冒烟] HTTP {rec.get('status')}，耗时 {rec.get('latency_ms')} ms，"
              f"服务端 processing_time={rec.get('server_ms')} ms")
        print(f"       响应字段：{json.dumps(rec.get('body_head'), ensure_ascii=False)}")
        print(f"       框数：{rec.get('n_boxes')}")
        if rec.get("error"):
            print(f"[冒烟失败] {rec['error']}", file=sys.stderr)
            print("       排查建议：① 服务是否已启动（uvicorn src.api.main:app --port 8000）；"
                  "② 路径是否为 /algo/predict；③ image_id 是否已在服务端注册。", file=sys.stderr)
            return 1
        print("[冒烟通过] 接口可按契约正常返回。")
        return 0

    # ---- 预热 ----
    if args.warmup > 0 and not args.quiet:
        print(f"[信息] 预热 {args.warmup} 次（不计入统计）…")
    for _ in range(max(0, args.warmup)):
        do_request(args.url, args.image, args.image_id, args.timeout)

    # ---- 正式压测 ----
    gpu_baseline = read_gpu_mem()
    sampler = ResourceSampler(interval=0.4)
    sampler.start()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    t_wall0 = time.perf_counter()

    records: List[dict] = []
    if args.concurrency == 1:
        for i in range(args.n):
            records.append(do_request(args.url, args.image, args.image_id, args.timeout))
            if not args.quiet and (i + 1) % max(1, args.n // 10) == 0:
                print(f"       已完成 {i + 1}/{args.n}")
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(do_request, args.url, args.image, args.image_id, args.timeout)
                    for _ in range(args.n)]
            done = 0
            for fu in as_completed(futs):
                records.append(fu.result())
                done += 1
                if not args.quiet and done % max(1, args.n // 10) == 0:
                    print(f"       已完成 {done}/{args.n}")

    t_wall = time.perf_counter() - t_wall0
    resources = sampler.stop()

    # ---- 统计 ----
    ok_recs = [r for r in records if r.get("ok")]
    lat_all = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    lat_ok = [r["latency_ms"] for r in ok_recs if r.get("latency_ms") is not None]
    srv_ok = [r["server_ms"] for r in ok_recs if r.get("server_ms") is not None]

    summary = {
        "total": len(records),
        "success": len(ok_recs),
        "failed": len(records) - len(ok_recs),
        "success_rate": (len(ok_recs) / len(records)) if records else 0.0,
        "wall_time_s": round(t_wall, 3),
        "qps": round(len(ok_recs) / t_wall, 3) if t_wall > 0 else 0.0,
        "overhead_ms": (round(statistics.fmean(lat_ok) - statistics.fmean(srv_ok), 3)
                        if lat_ok and srv_ok else None),
    }

    def _stats(vals: List[float]) -> dict:
        if not vals:
            return {k: None for k in ("count", "mean", "p50", "p90", "p95", "p99",
                                      "min", "max", "stdev")}
        return {
            "count": len(vals),
            "mean": round(statistics.fmean(vals), 3),
            "p50": pct(vals, 50), "p90": pct(vals, 90), "p95": pct(vals, 95), "p99": pct(vals, 99),
            "min": round(min(vals), 3), "max": round(max(vals), 3),
            "stdev": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0,
        }

    latency_client = _stats(lat_ok)
    latency_server = _stats(srv_ok)

    # ---- 与目标对照 ----
    targets: List[dict] = []
    if args.device == "gpu":
        targets.append({
            "name": "单张 640 图（GPU）", "target": "≤ 200 ms",
            "measured": f"{latency_client['mean']} ms（均值）",
            "verdict": "✅ 达标" if (latency_client["mean"] or 1e9) <= 200 else "❌ 未达标",
        })
    else:
        targets.append({
            "name": "单张 640 图（CPU）", "target": "≤ 1500 ms",
            "measured": f"{latency_client['mean']} ms（均值）",
            "verdict": "✅ 达标" if (latency_client["mean"] or 1e9) <= 1500 else "❌ 未达标",
        })
    targets.append({
        "name": "p95 稳定性", "target": "p95 ≤ 2 × 均值",
        "measured": f"p95={latency_client['p95']} ms，均值={latency_client['mean']} ms",
        "verdict": "✅ 稳定" if (latency_client["p95"] is not None
                                and latency_client["mean"]
                                and latency_client["p95"] <= 2 * latency_client["mean"])
        else "❌ 抖动大",
    })
    targets.append({
        "name": "大图切片推理（4096）", "target": "≤ 30000 ms",
        "measured": "本工具只压 640 单图；大图请用 tools/tile_infer.py 计时",
        "verdict": "—（另测）",
    })

    # ---- 资源口径说明 ----
    res_notes = [
        "RSS 是压测客户端进程的内存占用；若服务端是独立进程（uvicorn 单独启动），"
        "请用任务管理器或 psutil 单独观察服务端进程。",
        "GPU 显存是整卡读数，包含桌面/浏览器等其他程序；"
        "请用 max 与 baseline 的差值判断本服务真实增量。",
        "client latency 含网络与排队；server processing_time 只含服务端处理。"
        "两者差值即网络+排队开销，QPS 上不去时优先看它。",
    ]
    if not HAS_PSUTIL:
        res_notes.append("未安装 psutil：系统内存无法读取，已用 ctypes 读本进程 RSS 兜底。"
                         "pip install psutil 可获得完整数据。")
    if gpu_baseline is None:
        res_notes.append("未采集到 NVIDIA GPU 信息（无卡、无驱动或 nvidia-smi 不在 PATH），"
                         "GPU 相关指标留空。")
    gpu_max = resources["gpu_memory_used_mb"]["max"]
    resources["gpu_baseline_mb"] = round(gpu_baseline["memory_used_mb"], 1) if gpu_baseline else None
    resources["gpu_delta_mb"] = (round(gpu_max - gpu_baseline["memory_used_mb"], 1)
                                 if (gpu_max is not None and gpu_baseline) else None)
    resources["system_memory"] = read_system_mem()

    failures = [r for r in records if not r.get("ok")]
    result = {
        "url": args.url,
        "request_mode": request_mode,
        "image": os.path.abspath(args.image) if args.image else None,
        "image_id": args.image_id,
        "n": args.n, "warmup": args.warmup,
        "concurrency": args.concurrency, "timeout_s": args.timeout,
        "device": args.device,
        "started_at": started_at,
        "http_client": http_client,
        "mem_backend": mem_backend,
        "summary": summary,
        "latency_client_ms": latency_client,
        "latency_server_ms": latency_server,
        "resources": resources,
        "resource_notes": res_notes,
        "targets": targets,
        "failures": failures,
        "records": records,
    }

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "bench_latency.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    md_path = os.path.join(args.out, "bench_latency.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(result))

    print("=" * 68)
    print(f"[完成] 成功率 {summary['success_rate']:.2%}（{summary['success']}/{summary['total']}），"
          f"QPS {summary['qps']:.2f}，总墙钟 {summary['wall_time_s']}s")
    if latency_client["count"]:
        print(f"       客户端耗时 ms：均值 {latency_client['mean']} | "
              f"p50 {latency_client['p50']} | p95 {latency_client['p95']} | "
              f"p99 {latency_client['p99']} | max {latency_client['max']}")
    if latency_server["count"]:
        print(f"       服务端 processing_time 均值 {latency_server['mean']} ms，"
              f"网络+排队开销 {summary['overhead_ms']} ms")
    print(f"       资源：RSS max {resources['rss_mb']['max']} MB，"
          f"GPU 显存 max {resources['gpu_memory_used_mb']['max']} MB")
    for t in targets[:2]:
        print(f"       目标「{t['name']}」{t['target']} → {t['measured']} {t['verdict']}")
    print(f"       JSON → {json_path}")
    print(f"       报告 → {md_path}")
    if failures:
        print(f"[警告] 有 {len(failures)} 个请求失败，前 3 条：", file=sys.stderr)
        for f_ in failures[:3]:
            print(f"        {f_.get('error')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
