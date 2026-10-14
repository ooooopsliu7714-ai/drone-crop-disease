"""一键安装 GPU 版 PyTorch（cu128）并校验 —— RTX 5070 (Blackwell sm_120) 专用。

为什么必须用 cu128：
    RTX 50 系是 Blackwell 架构、算力 **sm_120**。只有 CUDA 12.8（cu128）及以上的
    PyTorch 轮子编译了 sm_120 支持；装 cu118/cu121 会出现
    `no kernel image is available for execution on the device`。
    本机现状是 `torch 2.10.0+cpu`，GPU 完全闲置。

⚠️ 这个脚本会**替换掉 CPU 版 torch/torchvision**（约 2.5~3.5 GB 下载）：
    - 替换后现有推理与测试仍可用（代码不依赖 CPU 版特有行为）；
    - torch 与 torchvision **必须一起装**，只换 torch 会导致版本不匹配。

用法（项目根目录，PowerShell）::

    # 常规：装到当前 Python（先自动体检磁盘空间，不够就中止且不动环境）
    python scripts/setup_gpu_env.py

    # 系统盘（C:）空间不足时：把 torch 装到 D: 的 venv 里，复用已装好的其它包
    python scripts/setup_gpu_env.py --venv .venv --torch-mirror sjtu

    # 只体检 / 不锁版本 / 免交互
    python scripts/setup_gpu_env.py --check-only
    python scripts/setup_gpu_env.py --latest --yes

⚠️ 两个来自实测的坑（脚本已内置防护）：

1. **磁盘空间**：cu128 版 torch 装上要 4~5 GB，下载时还要同量级的临时空间。
   早期版本先卸载旧 torch 再安装，遇到 C: 盘满就**把环境搞坏了**（torch 没了、
   新版也没装上）。现在会**先查空间、不够就直接中止且不卸载任何东西**。
2. **下载源**：`download.pytorch.org` 在国内实测只有 0.14 MB/s，
   下 2.7 GB 要 5 小时以上；SJTU 镜像 2.6 MB/s，约 18 分钟。
   空间/网络都不顺时用 `--torch-mirror sjtu`。

退出码：0 全部就绪 / 1 校验未通过或空间不足 / 2 被用户取消 / 3 安装失败
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
#: 国内镜像（实测 download.pytorch.org 0.14 MB/s，SJTU 2.6 MB/s，差约 19 倍）
TORCH_INDEX_MIRRORS = {
    "sjtu": "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128",
    "aliyun": "https://mirrors.aliyun.com/pytorch-wheels/cu128",
}
#: 与当前环境匹配的版本（实测 torch 2.10.0+cpu / torchvision 0.25.0+cpu）。
#: 不写死版本时可传 --latest；写死的目的是保证与已验证组合一致。
PINNED_TORCH = ["torch==2.10.0", "torchvision==0.25.0"]
EXTRA_PKGS = ["onnx", "onnxruntime"]
#: 训练必需的其它依赖（镜像加速可用 pip 的 -i 参数，见 --pip-index）
TRAIN_PKGS = ["transformers", "pyyaml", "opencv-python", "numpy"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], *, timeout: int | None = None) -> tuple[int, str]:
    """执行命令并回显；返回 (退出码, 输出文本)。"""
    log("$ " + " ".join(cmd))
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore",
            timeout=timeout, cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return 124, "命令超时"
    out = (p.stdout or "") + (p.stderr or "")
    if out.strip():
        print(out.strip()[-4000:])
    return p.returncode, out


def probe_torch() -> dict:
    """体检：torch 版本、CUDA 可用性、算力、能否真的跑一次 GPU 矩阵乘。"""
    import importlib

    info: dict = {"torch": None, "torchvision": None, "cuda": False, "cap": None,
                  "device": None, "smoke": None, "error": None}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception as exc:  # pragma: no cover
        info["error"] = f"torch 不可用：{exc}"
        return info
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        info["torchvision"] = "未安装"
    info["cuda"] = bool(torch.cuda.is_available())
    if info["cuda"]:
        info["device"] = torch.cuda.get_device_name(0)
        info["cap"] = torch.cuda.get_device_capability(0)
        try:
            a = torch.randn(256, 256, device="cuda")
            b = torch.randn(256, 256, device="cuda")
            c = (a @ b).sum().item()
            torch.cuda.synchronize()
            info["smoke"] = f"GPU 矩阵乘通过（校验和 {c:.2f}）"
        except Exception as exc:
            # 这一步能抓出"装了 CUDA 版但没编进 sm_120"的情况
            info["smoke"] = f"失败：{type(exc).__name__}: {exc}"
    _ = importlib
    return info


def print_report(info: dict) -> bool:
    print("\n" + "=" * 74)
    print("GPU 环境体检")
    print("=" * 74)
    print(f"  torch        : {info['torch']}")
    print(f"  torchvision  : {info['torchvision']}")
    print(f"  cuda.is_available : {info['cuda']}")
    print(f"  设备         : {info['device']}")
    print(f"  算力         : {info['cap']}")
    print(f"  GPU 自检     : {info['smoke']}")
    if info["error"]:
        print(f"  错误         : {info['error']}")

    ok = True
    if not info["torch"] or "+cu" not in str(info["torch"]):
        print("\n  ✗ 当前不是 CUDA 版 torch（应是 2.x.x+cu128，现在是 +cpu）")
        ok = False
    if not info["cuda"]:
        print("  ✗ cuda.is_available() 为 False")
        ok = False
    if info["cap"] and tuple(info["cap"]) != (12, 0):
        print(f"  ✗ 算力 {info['cap']} 不是 (12, 0)/sm_120 —— 可能装错了轮子")
        ok = False
    if info["smoke"] and str(info["smoke"]).startswith("失败"):
        print("  ✗ GPU 前向/矩阵乘失败：轮子里没编进 sm_120，重装 cu128")
        ok = False
    if ok:
        print("\n  ✓ 全部通过：可以开始训练")
    else:
        print("\n  详见 REQUIREMENTS.md 第 4 节「环境故障排查」")
    print("=" * 74)
    return ok


def pip_install(pkgs: list[str], index: str | None = None, extra_index: str | None = None,
                python: str | None = None) -> int:
    cmd = [python or PY, "-m", "pip", "install", "-U", *pkgs]
    if index:
        cmd += ["--index-url", index]
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    code, _ = run(cmd)
    return code


# ---------------------------------------------------------------------------
# 磁盘空间体检 —— 必须在"卸载旧 torch"之前做
# ---------------------------------------------------------------------------
#: cu128 版 torch+torchvision 需要的空间（下载 + 解包安装，实测量级）
NEED_FREE_GB = 6.0


def _disk_free_gb(path: Path) -> float:
    """返回 path 所在盘的可用 GB；取不到返回 -1。"""
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return shutil.disk_usage(str(p)).free / (1024 ** 3)
    except Exception:
        return -1.0


def install_target_dir() -> Path:
    """pip 的实际安装目标目录（site-packages），用于判断要占哪个盘。"""
    import site

    try:
        paths = list(site.getsitepackages())
    except Exception:
        paths = []
    for cand in paths:
        if cand:
            return Path(cand)
    return Path(sys.prefix) / "Lib" / "site-packages"


def temp_dir() -> Path:
    """pip 下载临时目录所在位置（也要占空间）。"""
    return Path(os.environ.get("TMP") or os.environ.get("TEMP") or tempfile.gettempdir())


def disk_report() -> tuple[bool, list[str]]:
    """检查安装目标盘与临时目录盘的空间。返回 (是否够用, 说明)。"""
    notes: list[str] = []
    tgt = install_target_dir()
    tmp = temp_dir()
    free_tgt = _disk_free_gb(tgt)
    free_tmp = _disk_free_gb(tmp)
    notes.append(f"安装目标：{tgt}")
    notes.append(f"  所在盘可用 {free_tgt:.2f} GB（需要 ≈{NEED_FREE_GB:.0f} GB）" if free_tgt >= 0
                 else "  所在盘可用空间读取失败")
    notes.append(f"pip 临时目录：{tmp}")
    notes.append(f"  所在盘可用 {free_tmp:.2f} GB" if free_tmp >= 0 else "  可用空间读取失败")

    ok = True
    same_drive = False
    try:
        same_drive = tgt.drive.lower() == tmp.drive.lower()
    except Exception:
        pass
    # 同盘时下载与安装叠加占用，要求更高
    need_tgt = NEED_FREE_GB * (1.8 if same_drive else 1.0)
    if free_tgt >= 0 and free_tgt < need_tgt:
        ok = False
        notes.append(
            f"  ✗ 安装盘空间不足：可用 {free_tgt:.2f} GB < 需要 {need_tgt:.1f} GB"
            + ("（pip 临时目录同盘，占用叠加）" if same_drive else "")
        )
    if free_tmp >= 0 and free_tmp < NEED_FREE_GB:
        ok = False
        notes.append(f"  ✗ 临时目录盘空间不足：可用 {free_tmp:.2f} GB < 需要 {NEED_FREE_GB:.0f} GB")
    if ok:
        notes.append("  ✓ 空间充足")
    return ok, notes


def make_venv(venv_dir: Path, *, reuse_system: bool = True) -> tuple[bool, str]:
    """在指定目录建 venv（默认复用系统已装好的包），返回 (是否成功, 说明)。

    用途：系统盘（C:）被占满、装不下 CUDA 版 torch 时，把 torch 装到别的盘。
    `--system-site-packages` 让 venv 直接复用系统里的 opencv/transformers/fastapi，
    只需要为新装的 torch 付磁盘空间。
    """
    if (venv_dir / "Scripts" / "python.exe").is_file():
        return True, f"venv 已存在：{venv_dir}"
    cmd = [PY, "-m", "venv"]
    if reuse_system:
        cmd.append("--system-site-packages")
    cmd.append(str(venv_dir))
    code, _ = run(cmd)
    if code != 0:
        return False, f"创建 venv 失败（退出码 {code}）"
    return True, f"venv 已创建：{venv_dir}（复用系统包）"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="安装并校验 GPU 版 PyTorch（cu128，RTX 50 系 sm_120 必需）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python scripts/setup_gpu_env.py                # 完整安装 + 校验
  python scripts/setup_gpu_env.py --check-only   # 只体检
  python scripts/setup_gpu_env.py --latest       # 不锁版本，装最新 cu128 轮子
  python scripts/setup_gpu_env.py --pip-index https://pypi.tuna.tsinghua.edu.cn/simple
""",
    )
    ap.add_argument("--check-only", action="store_true", help="只体检，不安装")
    ap.add_argument("--latest", action="store_true", help="不锁 torch 版本，装该渠道最新")
    ap.add_argument("--skip-onnx", action="store_true", help="不装 onnx/onnxruntime")
    ap.add_argument("--skip-extra", action="store_true", help="不装 transformers 等训练依赖")
    ap.add_argument("--pip-index", default=None, help="额外的 pip 镜像（仅用于非 torch 的包）")
    ap.add_argument("--yes", action="store_true", help="跳过交互确认")
    ap.add_argument("--venv", default=None,
                    help="把 torch 装到该目录的 venv 里（系统盘空间不足时用），"
                         "如 --venv .venv 或 --venv D:\\venvs\\drone")
    ap.add_argument("--no-uninstall", action="store_true",
                    help="安装前不卸载旧 torch（默认会卸载，避免 CPU/CUDA 两版冲突）")
    ap.add_argument("--torch-mirror", choices=sorted(TORCH_INDEX_MIRRORS), default=None,
                    help="用国内镜像下 torch 轮子（官方源在国内实测慢约 19 倍）")
    ap.add_argument("--torch-index", default=None,
                    help="自定义 torch 下载渠道（默认官方 cu128）")
    args = ap.parse_args(argv)

    torch_index = args.torch_index or (
        TORCH_INDEX_MIRRORS[args.torch_mirror] if args.torch_mirror else TORCH_INDEX
    )

    print("=" * 74)
    print("无人机病害识别 —— GPU 环境安装向导")
    print("=" * 74)
    print(f"Python : {PY}")
    print(f"项目根 : {ROOT}")
    print(f"目标渠道: {torch_index}\n")

    before = probe_torch()
    print(f"当前 torch = {before['torch']} | cuda 可用 = {before['cuda']}")

    if args.check_only:
        return 0 if print_report(before) else 1

    # ---- 可选：先建 venv（系统盘装不下时的正解）----
    install_py = PY
    if args.venv:
        venv_dir = Path(args.venv)
        if not venv_dir.is_absolute():
            venv_dir = ROOT / venv_dir
        print(f"\n将在 venv 里安装：{venv_dir}")
        ok, msg = make_venv(venv_dir)
        print(f"  {msg}")
        if not ok:
            return 3
        install_py = str(venv_dir / "Scripts" / "python.exe")
        # venv 存在盘上通常空间充足；仍按新目标再算一次
        print(f"  使用解释器：{install_py}")

    # ---- 磁盘空间体检（必须在卸载之前！）----
    print("\n磁盘空间体检：")
    space_ok, notes = disk_report()
    for n in notes:
        print(f"  {n}")
    if not space_ok:
        print("\n[中止] 空间不足，**没有卸载任何东西**，环境保持原样。可选做法：")
        print("  1) 把 torch 装到别的盘（推荐，无需管理员）：")
        print(r"     python scripts\setup_gpu_env.py --venv D:\venvs\drone")
        print("  2) 清理 C: 盘（需要管理员）：Windows 设置 -> 系统 -> 存储 -> 临时文件；")
        print(r"     或管理员运行：Dism /Online /Cleanup-Image /StartComponentCleanup")
        print("  3) 清理 Windows 更新残留（本次实测占 8.99 GB）：")
        print(r"     管理员运行：net stop wuauserv & net stop bits & rd /s /q "
              r"C:\Windows\SoftwareDistribution\Download & net start wuauserv & net start bits")
        return 1
    print("  ✓ 空间检查通过")

    torch_pkgs = ["torch", "torchvision"] if args.latest else list(PINNED_TORCH)
    print("\n将要执行：")
    if not args.no_uninstall:
        print(f"  1) 卸载旧版 torch/torchvision（当前解释器 {Path(install_py).parent.parent.name}）")
    print(f"  2) 安装 {', '.join(torch_pkgs)}（来自 {torch_index}，约 2.5~3.5 GB）")
    if not args.skip_extra:
        print(f"  3) 安装训练依赖 {', '.join(TRAIN_PKGS)}")
    if not args.skip_onnx:
        print(f"  4) 安装 ONNX 部署依赖 {', '.join(EXTRA_PKGS)}")
    print("\n注意：替换 torch 后现有推理与测试仍可用，但**磁盘会多占约 2~3 GB**。")
    if args.venv:
        print("装完后请用 venv 的解释器跑项目命令（脚本结束时会打印现成命令）。")

    if not args.yes:
        try:
            ans = input("\n继续？(y/N) ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("已取消。")
            return 2

    # ---- 1) 卸载旧 torch（空间已确认充足，不会把环境搞坏）----
    if not args.no_uninstall:
        log("卸载旧版 torch / torchvision ...")
        run([install_py, "-m", "pip", "uninstall", "-y", "torch", "torchvision"])

    # ---- 2) 装 cu128 ----
    log("安装 cu128 版 torch / torchvision（耗时取决于网速，请耐心等）...")
    code = pip_install(torch_pkgs, index=torch_index, python=install_py)
    if code != 0:
        # 区分"磁盘满"和"网络问题"——早期版本一律按网络问题提示，误导过使用者
        _, out = run([install_py, "-m", "pip", "--version"])
        print("\n[错误] torch 安装失败。按下面的现象对症处理：")
        print("  · 出现 'No space left on device' / OSError(28) -> **磁盘满**，不是网络问题：")
        print(r"      改用别的盘：python scripts\setup_gpu_env.py --venv D:\venvs\drone")
        print("  · 出现 timeout / ConnectionError -> 网络问题：")
        print("      换镜像重试：python scripts/setup_gpu_env.py --pip-index "
              "https://pypi.tuna.tsinghua.edu.cn/simple")
        print("  · 出现 'Could not find a version' -> 版本/渠道不对：加 --latest 不锁版本")
        print("  另外可先把 pip 临时目录指到空间大的盘：")
        print(r"      $env:TMP='D:\tmp'; $env:TEMP='D:\tmp'")
        return 3

    # ---- 3) 训练依赖 ----
    if not args.skip_extra:
        log("安装训练依赖 ...")
        code = pip_install(TRAIN_PKGS, index=args.pip_index, python=install_py)
        if code != 0:
            print("[警告] 部分训练依赖安装失败，请手动重试：pip install " + " ".join(TRAIN_PKGS))

    # ---- 4) ONNX ----
    if not args.skip_onnx:
        log("安装 onnx / onnxruntime ...")
        code = pip_install(EXTRA_PKGS, index=args.pip_index, python=install_py)
        if code != 0:
            print("[警告] onnx/onnxruntime 安装失败；训练不受影响，只影响 ONNX 导出。")

    # ---- 5) 校验 ----
    print()
    after = probe_torch()
    ok = print_report(after)

    if ok:
        print("\n下一步（全量训练）：")
        print(f"  {Path(PY).name} scripts/train_full_gpu.py --data-root data/processed_stress2")
    else:
        print("\n建议排查顺序：")
        print("  1) pip list | Select-String torch   -> 确认版本带 +cu128")
        print("  2) python -c \"import torch;print(torch.cuda.is_available())\"")
        print("  3) 仍是 False 时重装：pip install -U --index-url "
              "https://download.pytorch.org/whl/cu128 torch torchvision")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
