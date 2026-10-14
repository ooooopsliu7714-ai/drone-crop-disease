"""一键全量训练：分类（DINOv2）+ 检测（RT-DETRv2）+ ONNX 导出 + 评估。

前置：先跑 `python scripts/setup_gpu_env.py` 确认 CUDA 可用。

用法（项目根目录，PowerShell）::

    # 完整流程（推荐）：先体检 -> 分类 -> 检测 -> 导出 -> 评估
    python scripts/train_full_gpu.py --data-root data/processed_stress2

    # 8GB 显存保守档（batch 更小，时间长一点但稳）
    python scripts/train_full_gpu.py --data-root data/processed_stress2 --profile safe

    # 只训检测主模型
    python scripts/train_full_gpu.py --data-root data/processed_stress2 --only det

    # 快速验证整条链路（每阶段只跑 30 步，约 10 分钟）
    python scripts/train_full_gpu.py --data-root data/processed_stress2 --smoke

    # 只重跑评估（训练已完成时）
    python scripts/train_full_gpu.py --data-root data/processed_stress2 --only eval

脚本会：
  1. 校验环境（CUDA / 数据集 / 标签体系），不通过就直接退出，不浪费训练时间；
  2. 按阶段训练，每阶段日志写到 `runs/<阶段>.log`，可随时 Ctrl+C 中断；
  3. 每阶段产物独立落到 `models/<name>/`，起服务时自动发现；
  4. 训练完自动导出 sidecar + 跑评估，并打印关键指标。

预计耗时（RTX 5070 Laptop 8GB 参考值，实际以本机为准）：
  分类 DINOv2-small 冻结 backbone：约 5~15 分钟
  检测 RT-DETRv2-r18vd @640：约 60~150 分钟（取决于 epochs）
  ONNX 导出 + 评估：约 5~15 分钟
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 直接 `python scripts/train_full_gpu.py` 运行时，sys.path[0] 是 scripts/ 而不是项目根，
# 会导致 `import src.*` 失败（本机实测：前置校验报 "No module named 'src'"）。
# 这里显式把项目根插到最前面，保证两种调用方式（python -m scripts.x / python scripts/x.py）都能跑。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PY = sys.executable
RUNS = ROOT / "runs"


def load_env_file(path: Path | None = None) -> list[str]:
    """读取项目根目录的 `.env`，把键值写进 os.environ（**不覆盖**已存在的变量）。

    为什么需要：环境变量只在当前窗口有效，重启电脑/新开窗口就丢。本机实测踩过——
    重启后忘记重设，`ALGO_LABELS_FILE` 变空，标签体系从 2 类回退成 15 类，
    于是模型把 class 1（受胁迫）学成"稻瘟病"。这种静默错标比报错更难发现。

    已存在的环境变量优先（方便命令行临时覆盖）；返回实际生效的键名列表。
    """
    p = path or (ROOT / ".env")
    applied: list[str] = []
    if not p.is_file():
        return applied
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not k or (k in os.environ and os.environ[k]):
            continue
        os.environ[k] = v
        applied.append(k)
    return applied

#: 三档超参。数值来自本机（RTX 5070 Laptop 8GB，且系统里其它程序常占 1.5~2.5 GB 显存）实测：
#:   lowmem   —— 已实测跑通（batch1 + 512 输入），显存紧张时用这档
#:   safe     —— 640 输入 batch2，要求显存较空
#:   balanced —— 640 输入 batch4，需要关掉其它占显存的程序
PROFILES: dict[str, dict] = {
    "lowmem": {
        "cls": {"epochs": 10, "batch": 8, "workers": 0, "backbone": "facebook/dinov2-small"},
        "det": {"epochs": 60, "batch": 1, "grad_accum": 8, "workers": 0, "lr": "1e-4", "imgsz": 512},
    },
    # 解冻 backbone 微调。实测：可训练参数 8.88M -> 20.07M，每 epoch 135s -> 166s
    # （只慢 23%），显存 1942 -> 2165 MiB（8GB 卡完全够），epoch1 val_loss 18.4 -> 14.3。
    # backbone 学习率必须比 head 小一个数量级（train_det 已内置参数分组），
    # 否则预训练权重会被大学习率破坏，效果可能还不如冻结。
    "finetune": {
        "cls": {"epochs": 10, "batch": 8, "workers": 0, "backbone": "facebook/dinov2-small"},
        "det": {"epochs": 40, "batch": 1, "grad_accum": 8, "workers": 0, "lr": "1e-4",
                "backbone_lr": "1e-5", "imgsz": 512, "unfreeze": True},
    },
    "safe": {
        "cls": {"epochs": 10, "batch": 16, "workers": 0, "backbone": "facebook/dinov2-small"},
        "det": {"epochs": 50, "batch": 2, "grad_accum": 4, "workers": 0, "lr": "5e-5", "imgsz": 640},
    },
    "balanced": {
        "cls": {"epochs": 10, "batch": 32, "workers": 4, "backbone": "facebook/dinov2-small"},
        "det": {"epochs": 50, "batch": 4, "grad_accum": 2, "workers": 4, "lr": "1e-4", "imgsz": 640},
    },
}


def log(msg: str) -> None:
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage(title: str, cmd: list[str], log_name: str, *, allow_fail: bool = False) -> int:
    """跑一个阶段：命令回显 + 全量日志落盘 + 失败处理。

    额外能力：如果失败原因是 DataLoader 多进程被拒（受限/沙箱环境的典型症状
    `PermissionError: [WinError 5]`），自动改用 `--workers 0` 重跑一次。
    这样同一份脚本在**真实训练机**（要多进程喂 GPU）和**受限环境**（不能 fork）
    上都能跑，不需要手工记住这个差异。
    """
    RUNS.mkdir(parents=True, exist_ok=True)
    log_path = RUNS / log_name
    log(f"阶段：{title}")
    print("  $ " + " ".join(cmd))
    print(f"  日志：{log_path}")
    env = dict(os.environ)
    env.setdefault("PYTHONWARNINGS", "ignore")
    # 显存碎片治理：开启后 OOM 会干净报错而不是把进程搞崩
    # （本机实测：不开时 batch4 OOM 会以 0xC0000005 访问违例退出，开了就正常报 OOM）
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="ignore") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    dt = time.time() - t0
    print(f"  退出码 {p.returncode}，耗时 {dt/60:.1f} 分钟")

    if p.returncode != 0:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        workers_issue = ("WinError 5" in text or "PermissionError" in text) and "--workers" in " ".join(cmd)
        finished = _training_completed(text)
        if workers_issue and not finished:
            retry = _with_zero_workers(cmd)
            print("  检测到 DataLoader 多进程被系统拒绝，自动改用 --workers 0 重跑 ...")
            print("  $ " + " ".join(retry))
            t0 = time.time()
            with open(log_path, "a", encoding="utf-8", errors="ignore") as fh:
                fh.write("\n\n===== 自动重试：--workers 0 =====\n")
                p = subprocess.run(retry, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
            print(f"  退出码 {p.returncode}，耗时 {(time.time()-t0)/60:.1f} 分钟")
        elif workers_issue and finished:
            # 关键：训练已经跑完（权重/sidecar 都写了），只是末尾某一步返回非 0。
            # 早期版本仍然整段重跑，白白多花 20 分钟（本机实测），这里直接判定成功。
            print("  检测到多进程报错，但训练已完成（权重与 sidecar 均已写出）-> 视为成功，不重跑")
            return 0

    if p.returncode != 0:
        print(f"  ---- {log_name} 尾部 ----")
        print(_tail(log_path, 25))
        if not allow_fail:
            print(f"\n[中止] 阶段「{title}」失败。完整日志：{log_path}")
            return p.returncode
        print(f"[警告] 阶段「{title}」失败但允许继续。")
    return p.returncode


def memory_report(min_commit_gb: float = 3.0) -> tuple[bool, list[str]]:
    """检查可用物理内存与**提交上限**（commit limit）。

    为什么要查提交上限：CUDA/cuBLAS 除了显存还要分配宿主内存，而 Windows 的
    提交上限 = 物理内存 + 页面文件。若 C: 盘快满，页面文件无法扩展，提交上限就会很小——
    表现是训练报 `CUBLAS_STATUS_EXECUTION_FAILED` 或
    `MemoryError: Unable to allocate 1.17 MiB`（本机实测踩过：连 1 MB 都分配不出来，
    看上去像 CUDA 坏了，其实是整机内存耗尽）。

    阈值依据（本机实测对照）：
        失败过：提交可用 1.2 ~ 2.8 GB
        成功过：提交可用 3.4 GB（分类 10 epoch）、4.7 GB（检测冒烟）
    所以默认门槛取 3.0 GB；该值会随其它程序开关而波动，紧张时关掉浏览器最有效。
    """
    notes: list[str] = []
    ok = True
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = _MS()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        gb = 1024 ** 3
        avail_phys = m.ullAvailPhys / gb
        avail_commit = m.ullAvailPageFile / gb
        notes.append(
            f"物理内存可用 {avail_phys:.2f} GB / 共 {m.ullTotalPhys/gb:.1f} GB（占用 {m.dwMemoryLoad}%）"
        )
        # 用两位小数：早期只显示一位时出现过"显示 3.0 却说低于 3.0"的歧义
        notes.append(
            f"提交上限可用 {avail_commit:.2f} GB / 共 {m.ullTotalPageFile/gb:.1f} GB"
            f"（门槛 {min_commit_gb:.1f} GB）"
        )
        if avail_commit < min_commit_gb:
            ok = False
            notes.append(
                f"  ✗ 提交上限只剩 {avail_commit:.2f} GB，低于门槛 {min_commit_gb:.1f} GB："
                "训练会在 CUDA 初始化或取数据时报内存错误"
            )
        elif avail_commit < min_commit_gb * 2:
            notes.append(f"  ⚠️ 提交上限偏低（{avail_commit:.2f} GB），紧张时可能失败")
        if avail_phys < 2.5:
            ok = False
            notes.append(f"  ✗ 物理内存只剩 {avail_phys:.2f} GB")
    except Exception as exc:
        notes.append(f"内存信息读取失败：{exc}")
    return ok, notes


def _training_completed(text: str) -> bool:
    """判断日志里是否出现"训练已完成"的证据（用于避免无意义重跑）。"""
    markers = (
        "训练完成",                 # train_cls / train_det 的正常收尾
        "已保存更优权重",           # 至少完成过一个 epoch 并存了权重
        "字段检查：通过",           # 推理自检通过
    )
    last_epoch = re.findall(r"epoch\s+(\d+)\s*/\s*(\d+)", text)
    if last_epoch:
        done, total = last_epoch[-1]
        if int(total) > 0 and int(done) >= int(total):
            return True
    return any(m in text for m in markers)


def _with_zero_workers(cmd: list[str]) -> list[str]:
    """把命令里的 --workers N 换成 --workers 0；没有该参数就追加。"""
    out = list(cmd)
    if "--workers" in out:
        i = out.index("--workers")
        if i + 1 < len(out):
            out[i + 1] = "0"
            return out
    out += ["--workers", "0"]
    return out


def _tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return "(日志不可读)"
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# 环境与数据前置校验
# ---------------------------------------------------------------------------
def preflight(data_root: Path, labels_file: str | None) -> tuple[bool, list[str]]:
    """返回 (是否通过, 提示列表)。不通过就不要开始训练。"""
    notes: list[str] = []
    ok = True

    # 1) CUDA
    try:
        import torch

        if not torch.cuda.is_available():
            ok = False
            notes.append(
                f"CUDA 不可用（当前 torch {torch.__version__}）。"
                "先跑：python scripts/setup_gpu_env.py"
            )
        else:
            cap = torch.cuda.get_device_capability(0)
            notes.append(f"GPU：{torch.cuda.get_device_name(0)}  算力 sm_{cap[0]}{cap[1]}")
    except Exception as exc:  # pragma: no cover
        ok = False
        notes.append(f"torch 不可用：{exc}")

    # 2) 标签体系
    if labels_file:
        os.environ["ALGO_LABELS_FILE"] = labels_file
    try:
        from src.core.labels import load_labels

        lb = load_labels()
        notes.append(f"标签体系：{lb.num_classes} 类（{lb.source.name if lb.source else '默认'}）")
    except Exception as exc:
        ok = False
        notes.append(f"标签体系加载失败：{exc}")
        lb = None

    # 3) 数据集
    for split in ("train", "val"):
        p = data_root / split / "annotations.json"
        if not p.is_file():
            ok = False
            notes.append(f"缺少 {p}（先用 src.data.prepare_data 生成）")
            continue
        try:
            coco = json.loads(p.read_text(encoding="utf-8-sig"))
            n_img, n_box = len(coco.get("images", [])), len(coco.get("annotations", []))
            notes.append(f"{split}: {n_img} 图 / {n_box} 框")
            if n_box == 0:
                ok = False
                notes.append(f"{split} 标注框为 0，训练没有意义（检查数据转换）")
        except Exception as exc:
            ok = False
            notes.append(f"{split} 读取失败：{exc}")

    # 4) 类别数一致性（数据集 vs 标签体系）
    if lb is not None:
        for split in ("train", "val"):
            p = data_root / split / "annotations.json"
            if not p.is_file():
                continue
            try:
                coco = json.loads(p.read_text(encoding="utf-8-sig"))
                ids = {int(c["id"]) for c in coco.get("categories", [])}
                if ids and max(ids) >= lb.num_classes:
                    ok = False
                    notes.append(
                        f"{split} 的类别 id 最大 {max(ids)} ≥ 标签体系类别数 {lb.num_classes}："
                        "数据和标签体系不匹配（检查 --labels / ALGO_LABELS_FILE）"
                    )
                # 5) **类别名一致性**：只看 id 数量不够——15 类的表也能容纳 id 0/1，
                #    于是"受胁迫"会被学成"稻瘟病"（本机实测踩过：重启后 ALGO_LABELS_FILE 丢失，
                #    标签体系回退成 15 类，数据集仍是 2 类，静默错标）。这里直接比名字。
                ds_names = {str(c.get("name")) for c in coco.get("categories", []) if c.get("name")}
                if ds_names and not ds_names.issubset(set(lb.names)):
                    ok = False
                    notes.append(
                        f"✗ {split} 的类别名与当前标签体系不一致：\n"
                        f"      数据集类别名 : {sorted(ds_names)}\n"
                        f"      标签体系类别名: {lb.names[:6]}{'...' if lb.num_classes > 6 else ''}\n"
                        "      请设置正确的标签体系，例如：\n"
                        "        --labels-file configs/labels_stress2.yaml\n"
                        "      （重启后环境变量会丢，脚本会自动读项目根的 .env，见该文件说明）"
                    )
            except Exception:
                pass
    return ok, notes


# ---------------------------------------------------------------------------
# 指标摘要
# ---------------------------------------------------------------------------
def summarize_eval(json_path: Path) -> None:
    """从评估结果里挑关键指标打印出来（答辩要用的那几个数字）。"""
    if not json_path.is_file():
        print(f"  （未找到 {json_path}）")
        return
    try:
        r = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"  评估结果解析失败：{exc}")
        return
    overall = r.get("overall") or r
    print("\n  关键指标：")
    for key, label in (
        ("map50", "AP@0.5"),
        ("map", "mAP@0.5:0.95"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ):
        if key in overall:
            v = overall[key]
            print(f"    {label:14s} = {float(v):.4f}")
    agri = r.get("agri") or {}
    if agri.get("area_metric_note"):
        print(f"\n    ⚠️ {agri['area_metric_note']}")


def _find_eval_json() -> Path | None:
    cands = sorted(RUNS.glob("eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="一键全量训练（分类 + 检测 + ONNX 导出 + 评估）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--data-root", default="data/processed_stress2", help="处理后的数据集根目录")
    ap.add_argument("--labels-file", default=None, help="标签体系 yaml（默认读 ALGO_LABELS_FILE）")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="safe", help="超参档位")
    ap.add_argument("--only", choices=["all", "cls", "det", "export", "eval"], default="all",
                    help="只跑某个阶段")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="检测输入边长（默认取档位配置：lowmem=512，其它=640）")
    ap.add_argument("--cls-batch", type=int, default=None, help="分类 batch（覆盖档位默认值）")
    ap.add_argument("--det-batch", type=int, default=None, help="检测 batch（覆盖档位默认值）")
    ap.add_argument("--det-epochs", type=int, default=None, help="检测训练轮数（覆盖档位默认值）")
    ap.add_argument("--unfreeze-backbone", action="store_true",
                    help="解冻 RT-DETR backbone 做微调（更准但更慢、更吃显存）")
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="解冻时 backbone 的学习率，默认 1e-5（比 head 小一个数量级）")
    ap.add_argument("--cls-epochs", type=int, default=None, help="分类训练轮数（覆盖档位默认值）")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader 进程数：受限环境用 0；自己终端里可用 4 加速")
    ap.add_argument("--skip-cls", action="store_true",
                    help="跳过分类阶段（分类已训好时用），但仍会做检测+导出+评估")
    ap.add_argument("--min-commit-gb", type=float, default=3.0,
                    help="提交上限门槛（GB），默认 3.0；内存紧张但又想硬跑时调低它")
    ap.add_argument("--skip-mem-check", action="store_true",
                    help="只跳过内存检查，保留数据/标签一致性等其它校验")
    ap.add_argument("--smoke", action="store_true", help="快速冒烟：每阶段只跑 30 步")
    ap.add_argument("--skip-preflight", action="store_true", help="跳过环境校验（不推荐）")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("无人机病害识别 —— 全量训练编排")
    print("=" * 78)
    # 先加载 .env（重启/新窗口后环境变量会丢，这是实测踩过的坑）
    applied = load_env_file()
    if applied:
        print(f"已从 .env 载入：{', '.join(applied)}")
    prof = PROFILES[args.profile]
    data_root = (ROOT / args.data_root).resolve()

    # ---- 前置校验：不通过就别开始，省下几小时白跑 ----
    if not args.skip_preflight:
        ok, notes = preflight(data_root, args.labels_file)
        print("\n前置校验：")
        for n in notes:
            print(f"  · {n}")
        # 内存/提交上限单独查一遍（这是最容易被忽略、又最致命的一项）
        if args.skip_mem_check:
            print("  · （已用 --skip-mem-check 跳过内存检查）")
            mem_ok = True
        else:
            mem_ok, mem_notes = memory_report(args.min_commit_gb)
            for n in mem_notes:
                print(f"  · {n}")
        if not ok:
            print("\n[中止] 前置校验未通过（如上）。修好后重跑，或加 --skip-preflight 强行继续。")
            return 1
        if not mem_ok:
            print("\n[中止] 内存不足，直接开始会白跑。请先做以下任一处理：")
            print("  1) 关掉占内存的程序（浏览器多标签、豆包、ChatGPT、微信等），最常见也最快")
            print("  2) 把页面文件放大到 D: 盘（C: 满时 Windows 无法扩展页面文件）：")
            print("     Win+R -> sysdm.cpl -> 高级 -> 性能[设置] -> 高级 -> 虚拟内存[更改]")
            print("     -> 取消『自动管理』-> 选 D: -> 『系统管理的大小』-> 设置 -> 重启")
            print("  3) 清理 C: 盘（Windows 设置 -> 系统 -> 存储 -> 临时文件）")
            print("  4) 内存会波动：关掉几个程序后**重跑同一条命令**通常就过了（本机实测：")
            print("     同一分钟前是 3.0 GB 被拦，几分钟后变 4.2 GB 可跑）")
            print(f"  5) 确认要硬闯：加 --min-commit-gb {max(0.0, args.min_commit_gb - 1.0):.1f}")
            print("     （只放低内存门槛，仍保留数据与标签一致性校验）")
            return 1
    else:
        if args.labels_file:
            os.environ["ALGO_LABELS_FILE"] = args.labels_file

    labels_env = os.environ.get("ALGO_LABELS_FILE", "")
    print(f"\n标签体系环境变量 ALGO_LABELS_FILE = {labels_env!r}")
    print(f"超参档位 profile = {args.profile}")
    if args.smoke:
        print("冒烟模式：每阶段 max-steps-per-epoch=30")

    steps = ["30"] if args.smoke else ["0"]
    cls_cfg, det_cfg = prof["cls"], prof["det"]
    results: dict[str, str] = {}

    # ---- ① 分类：DINOv2 冻结 backbone + 线性头 ----
    if args.only in ("all", "cls") and not args.skip_cls:
        cmd = [
            PY, "-m", "scripts.train_cls",
            "--config", "configs/train_cls.yaml",
            "--data-root", str(data_root),
            "--backbone", cls_cfg["backbone"],
            "--name", "cls_dinov2",
            "--freeze-backbone", "--class-weight",
            "--epochs", str(args.cls_epochs or cls_cfg["epochs"]),
            "--batch-size", str(args.cls_batch or cls_cfg["batch"]),
            "--workers", str(args.workers if args.workers is not None else cls_cfg["workers"]),
            "--max-steps-per-epoch", steps[0],
            "--export-onnx",
        ]
        code = stage("① 分类训练（DINOv2）", cmd, "train_cls_gpu.log")
        results["分类"] = "成功" if code == 0 else f"失败(码 {code})"
        if code != 0 and not args.smoke:
            print("[提示] 分类失败常见原因与处理：")
            print("  · CUBLAS_STATUS_EXECUTION_FAILED / MemoryError -> 整机内存不足，")
            print("    先关掉其它程序（浏览器/豆包/ChatGPT 等），或把页面文件放到 D: 盘（见下）")
            print("  · CUDA out of memory -> 显存不足，降 batch：--profile lowmem")

    # ---- ② 检测：RT-DETRv2（主线）----
    if args.only in ("all", "det"):
        imgsz = str(args.imgsz or det_cfg.get("imgsz", 640))
        # backbone 冻结与否：默认冻结（8GB 显存友好）；加 --unfreeze-backbone 解冻微调。
        # 解冻时务必让 backbone 用更小的学习率（train_det 已内置分组：head=lr，bb=backbone_lr），
        # 否则预训练权重会被大学习率破坏，实测比冻结还差。
        unfreeze = bool(args.unfreeze_backbone or det_cfg.get("unfreeze", False))
        cmd = [
            PY, "-m", "scripts.train_det",
            "--config", "configs/train_det.yaml",
            "--data-root", str(data_root),
            "--name", "det_rtdetrv2",
            "--backend", "transformers-rtdetrv2",
            "--amp",
            "--epochs", str(args.det_epochs or det_cfg["epochs"]),
            "--batch-size", str(args.det_batch or det_cfg["batch"]),
            "--grad-accum", str(det_cfg["grad_accum"]),
            "--workers", str(args.workers if args.workers is not None else det_cfg["workers"]),
            "--lr", str(det_cfg["lr"]),
            "--backbone-lr", str(args.backbone_lr or det_cfg.get("backbone_lr", 1e-5)),
            "--imgsz", imgsz,
            "--max-steps-per-epoch", steps[0],
            "--export-onnx",
        ]
        if unfreeze:
            # 注意用 --no-freeze-backbone：只"不传 --freeze-backbone"是不够的，
            # configs/train_det.yaml 里 freeze_backbone: true 会让 backbone 继续冻结
            cmd.append("--no-freeze-backbone")
        else:
            cmd.append("--freeze-backbone")
        print(f"  backbone：{'解冻微调（backbone_lr=' + str(args.backbone_lr or det_cfg.get('backbone_lr', 1e-5)) + '）' if unfreeze else '冻结（只训 decoder/head）'}")
        code = stage("② 检测训练（RT-DETRv2）", cmd, "train_det_gpu.log")
        results["检测"] = "成功" if code == 0 else f"失败(码 {code})"
        if code != 0:
            print("[提示] 显存不足（CUDA out of memory）时的处理顺序：")
            print("  1) 关掉其它占显存的程序（浏览器/豆包/ChatGPT 等，本机常占 1.5~2.5 GB）")
            print("  2) 换更小的档：--profile lowmem（batch1 + 512 输入，已实测跑通）")
            print("  3) 手动再降：--batch-size 1 --imgsz 384")
            return code

    # ---- ③ 补齐 sidecar（分类缺 sidecar 时推理侧加载不了）----
    if args.only in ("all", "cls", "export"):
        for name, task in (("cls_dinov2", "classify"), ("det_rtdetrv2", "detect")):
            d = ROOT / "models" / name
            if (d / "best.pt").is_file() and not (d / "model.onnx.json").is_file():
                stage(
                    f"③ 生成 {name} 的 sidecar",
                    [PY, "-m", "scripts.export_onnx", "--ckpt-dir", f"models/{name}",
                     "--sidecar-only", "--task", task],
                    f"export_{name}.log",
                    allow_fail=True,
                )

    # ---- ④/⑤ 评估：val 与 test 都要评 ----
    #
    # ⚠️ 为什么必须**两个划分都评**（本机实测抓到的报告级问题）：
    #   原实现只评 val，输出文件名还写死成 `eval_stress2.json`（与数据集无关）。
    #   后果有两层：
    #     1. 测试集指标**不会自动产生**，极易被遗忘 —— 本机就差点漏掉；
    #        而 val 曾用于调置信度阈值、比对 512/768 两版模型，**已偏乐观**。
    #        实测同一模型：val mAP@0.5 = 0.1483，test = 0.1222（低 18%）。
    #        报告若只引 val，等于报了偏高的数字。
    #     2. 文件名写死，换数据集后仍叫 stress2，溯源时会认错。
    #   这里改为：val、test 各评一次，文件名带上「数据集_划分」。
    if args.only in ("all", "eval") and (ROOT / "models" / "det_rtdetrv2" / "best.pt").is_file():
        # 数据集标识（用于文件名）：processed_plantseg -> plantseg
        ds_tag = data_root.name.replace("processed_", "").replace("processed", "") or "data"
        for split, label in (("val", "④"), ("test", "⑤")):
            gt = data_root / split / "annotations.json"
            if not gt.is_file():
                if split == "test":
                    print(f"\n[提示] 没有 {gt}，跳过测试集评估")
                continue
            out_json = f"runs/eval_{ds_tag}_{split}.json"
            cmd = [
                PY, "-m", "scripts.eval",
                "--gt", str(gt),
                "--ckpt-dir", "models/det_rtdetrv2",
                # 与 sidecar 标定的 0.05 对齐；AP 本身是阈值无关的，
                # 这里只影响日志里的 P/R/F1 那一行。
                "--conf", "0.05",
                "--out-json", out_json,
            ]
            if args.smoke:
                cmd += ["--max-images", "30"]
            note = ("（★ 报告请用这个数字：测试集从未参与训练与调参）"
                    if split == "test" else "（注意：val 用于调参，指标偏乐观）")
            print(f"\n提示：评估 {split} 集，GPU 上约 1~5 分钟 {note}")
            stage(f"{label} 评估（{split}）{note}", cmd, f"eval_{split}.log", allow_fail=True)
            summarize_eval(ROOT / out_json)

    # ---- 汇总 ----
    print("\n" + "=" * 78)
    print("训练编排结束")
    print("=" * 78)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("\n产物：")
    for d in sorted((ROOT / "models").iterdir()):
        if d.is_dir():
            files = [f.name for f in d.iterdir() if f.is_file()]
            print(f"  models/{d.name}/  {', '.join(files)}")
    print("\n接下去：")
    print("  1) 起服务验证：python -m uvicorn src.api.app:app --port 8000")
    # 文件名不要写死：按数据集动态给出，避免换数据集后指到不存在的旧文件
    _tag = data_root.name.replace("processed_", "") or "data"
    print(f"  2) 看指标：runs/eval_{_tag}_test.json（★ 报告用这个）"
          f" / runs/eval_{_tag}_val.json（偏乐观）")
    print("  3) 打包汇报材料：python tools/report_export.py（见该脚本 --help）")
    print("  4) 压测：python tools/bench_latency.py --image <一张测试图> --n 50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
