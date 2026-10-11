"""带进度与断点续传的 wheel 下载器（国内镜像网络不稳时用）。

为什么需要它：pip 下载 2.7 GB 的 torch 时没有可靠进度输出，
卡住/限速时看不出是"在下载"还是"已经死了"（本机实测踩过：
4 分钟磁盘零增长，无法判断状态）。这个脚本每 3 秒打印一次进度与瞬时速度，
中断后重跑会自动从已下载的字节继续（HTTP Range）。

用法：
    python scripts/dl_wheel.py <url> <输出文件>

示例：
    python scripts/dl_wheel.py `
      https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp310-cp310-win_amd64.whl `
      .venv/_wheels/torch.whl
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, out: Path, *, timeout: int = 60, report_every: float = 3.0) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    have = out.stat().st_size if out.is_file() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"[续传] 已有 {human(have)}，从该位置继续", flush=True)

    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # 请求范围不满足：说明已下载完整
            print(f"[完成] 文件已完整（{human(have)}）")
            return 0
        print(f"[错误] HTTP {exc.code}: {exc.reason}")
        return 2
    except Exception as exc:
        print(f"[错误] 连接失败：{type(exc).__name__}: {exc}")
        return 3

    total = None
    cl = resp.headers.get("Content-Length")
    cr = resp.headers.get("Content-Range")
    if cr and "/" in cr:
        total = int(cr.split("/")[-1])
    elif cl:
        total = int(cl) + have

    # 服务端不支持 Range 时会返回 200 并从头开始
    mode = "ab" if resp.status == 206 else "wb"
    if mode == "wb" and have:
        print("[注意] 服务端不支持断点续传，将从头下载")
        have = 0

    t0 = time.time()
    last = t0
    last_bytes = have
    done = have
    with open(out, mode) as fh:
        while True:
            try:
                chunk = resp.read(1024 * 256)
            except Exception as exc:
                print(f"\n[中断] 读取失败：{type(exc).__name__}: {exc}")
                print(f"       已下载 {human(done)}，重跑本命令会自动续传。")
                return 4
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last >= report_every:
                inst = (done - last_bytes) / max(1e-6, now - last)
                avg = (done - have) / max(1e-6, now - t0)
                if total:
                    pct = done * 100.0 / total
                    eta = (total - done) / max(inst, 1.0)
                    print(
                        f"  {human(done)}/{human(total)} ({pct:5.1f}%)  "
                        f"{human(inst)}/s  均速 {human(avg)}/s  剩余约 {eta/60:.1f} 分钟",
                        flush=True,
                    )
                else:
                    print(f"  {human(done)}  {human(inst)}/s", flush=True)
                last, last_bytes = now, done

    size = out.stat().st_size
    dt = time.time() - t0
    print(f"[完成] {out.name}  {human(size)}  用时 {dt/60:.1f} 分钟  均速 {human((size-have)/max(1e-6,dt))}/s")
    if total and size != total:
        print(f"[警告] 大小不符：期望 {human(total)}，实际 {human(size)}，请重跑续传")
        return 5
    return 0


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="带进度与断点续传的 wheel 下载器（国内镜像网络不稳时用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python scripts/dl_wheel.py <url> <输出文件>

  python scripts/dl_wheel.py `
    https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp310-cp310-win_amd64.whl `
    .venv/_wheels/torch.whl

下完可本地安装（不再依赖网络）:
  pip install --no-index --no-deps .venv/_wheels/torch.whl
""",
    )
    ap.add_argument("url", help="wheel 的完整下载地址")
    ap.add_argument("out", help="保存路径")
    ap.add_argument("--timeout", type=int, default=60, help="单次读取超时秒数，默认 60")
    ap.add_argument("--report-every", type=float, default=3.0, help="进度打印间隔秒数，默认 3")
    args = ap.parse_args(argv[1:])
    return download(args.url, Path(args.out), timeout=args.timeout, report_every=args.report_every)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
