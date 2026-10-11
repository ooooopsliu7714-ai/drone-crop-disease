"""生成交付压缩包（部署形态 B：后端自己跑算法服务）。

设计原则：**用显式白名单，而不是排除法**。
排除法（"打包全部、排除 .venv/data/runs..."）在仓库变脏时会悄悄把垃圾带进交付物——
本机实测：项目根目录积累了 185 个 `tmp*` 临时目录（里面是 0 字节的占位 .onnx/.pt）
和 284 个 `__pycache__`，排除法稍有不慎就会把它们打包进去。
白名单则保证"只有列出来的东西会进包"。

同时做**交付前自检**：
  1. 清掉垃圾（tmp* / __pycache__ / 空目录）；
  2. 校验 sidecar 与 ONNX 图的输入名一致（这是会静默出错的坑）；
  3. 校验 ONNX 图能被 onnxruntime 加载并前向一次；
  4. 打印包内文件清单与体积。

用法：
    python scripts/make_delivery_package.py --out-dir deliverables
    python scripts/make_delivery_package.py --with-weights     # 额外带上 best.pt（重训用）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: 交付白名单：目录（整目录递归收录，但会过滤 __pycache__ 等）
INCLUDE_DIRS = ["src", "scripts", "configs", "tests", "docs", "tools", "deliverables"]
#: 交付白名单：单文件（会放到压缩包根目录，接收方解压后第一眼就能看到）
INCLUDE_FILES = [
    "README.md", "REQUIREMENTS.md", "pyproject.toml", ".gitignore",
    # 部署专用最短路径说明。放在**包根**而不是 docs/ 下：
    # REQUIREMENTS.md 是训练视角写的（300+ 行，含 GPU 训练内容），
    # 且部署安装那行漏了 onnxruntime —— 接收方照它装会缺依赖。
    # 这份 2 分钟能读完的说明放在最外层，避免他翻错文档。
    "docs/部署与验收.md",
]
#: 模型目录：只收这些文件名（.pt 是否收由 --with-weights 决定）
MODEL_DIRS = ["models/det_rtdetrv2", "models/cls_dinov2"]
MODEL_KEEP = {
    "model.onnx", "model.onnx.json", "checkpoint.json", "config.json", "train_log.jsonl",
}
MODEL_KEEP_WITH_WEIGHTS = MODEL_KEEP | {"best.pt", "last.pt"}

#: 随包附带少量示例图，让接收方**能立刻自测**
#
# 为什么必须有：交付包里原本没有图片，接收方跑 `smoke_http.py` 只会得到
# "找不到测试图"，然后就得回来问你 —— 一次交付变成两轮沟通。
# 附 3 张 640×640 的示例图（约 0.3 MB），放在 `data/uploads/`，
# 与接口"JSON + image_id 从 data/uploads/ 读图"的约定一致，收到即可跑通自测。
SAMPLE_IMAGES_FROM = "data/processed_stress2/val/images"
SAMPLE_IMAGES_TO = "data/uploads"
SAMPLE_IMAGE_COUNT = 3

#: 随包生成一份**脱敏 `.env`**，让接收方开箱即用。
#
# 为什么必须有（本机实测抓到的交付级事故）：不设 `ALGO_LABELS_FILE` 时，
# 服务回退到 `configs/labels.yaml`（15 类病害表），于是二分类模型的 class_id=1
# 被解释成 **`rice_blast` / 稻瘟病**。实测在干净解压目录里直接起服务：
#     disease_name    = rice_blast
#     disease_name_cn = 稻瘟病
# 而正确结果应是 `stressed` / `受胁迫`。**HTTP 200、字段齐全、没有任何报错**，
# 接收方很难发现病名是错的。
#
# 注意不能直接打包开发机上的 `.env`：里面含 `HF_HOME=D:\Users\jibei\...`
# 这类机器专属绝对路径。所以这里**只写与路径无关的必要项**。
DELIVERY_ENV = """# 运行期环境变量（本项目自动读取仓库根目录的 .env，与当前工作目录无关）
#
# ⚠️ 这一行是必需的：不设它，标签体系会回退到 configs/labels.yaml（15 类病害表），
#    二分类模型的类别会被解释成具体病种（实测输出"稻瘟病"而不是"受胁迫"），
#    且接口仍返回 HTTP 200，属于静默错标。
#
# 本模型是 2 类：healthy（健康）/ stressed（受胁迫）
ALGO_LABELS_FILE=configs/labels_stress2.yaml

# 置信度阈值（可选）。模型分数整体偏低，sidecar 里已标定为 0.10；
# 早筛场景要高召回可设为 0.05（召回约 0.81，误报增多）。
# ALGO_CONF_THRESHOLD=0.10

# 训练用：HuggingFace 国内镜像（仅训练阶段需要，部署可忽略）
HF_ENDPOINT=https://hf-mirror.com
"""
#: 一律丢弃的文件名/后缀
DROP_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"}
DROP_SUFFIX = (".pyc", ".pyo", ".log")


def clean_junk() -> list[str]:
    """清理仓库垃圾（tmp* 占位目录、__pycache__、空目录）。返回清理项摘要。"""
    removed: list[str] = []

    n = 0
    for d in ROOT.glob("tmp*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    if n:
        removed.append(f"删除 {n} 个 tmp* 临时目录（内含 0 字节占位 .onnx/.pt）")

    n = 0
    for d in list(ROOT.rglob("__pycache__")):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    if n:
        removed.append(f"删除 {n} 个 __pycache__")

    for name in ("torchinductor_jibei", ".pytest_cache"):
        p = ROOT / name
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(f"删除 {name}")
    return removed


def self_check() -> tuple[bool, list[str]]:
    """交付前自检：sidecar 与 ONNX 图一致 + 图能真正加载前向。"""
    notes: list[str] = []
    ok = True

    need_ort = True
    try:
        import onnxruntime as ort  # noqa: F401
    except Exception:
        need_ort = False
        notes.append("  ⚠️ 本机没有 onnxruntime，跳过'真实加载前向'检查")

    for rel in MODEL_DIRS:
        d = ROOT / rel
        onnx_p, sc_p = d / "model.onnx", d / "model.onnx.json"
        if not onnx_p.is_file() or not sc_p.is_file():
            ok = False
            notes.append(f"  ✗ {rel}: 缺 model.onnx 或 model.onnx.json")
            continue
        sc = json.loads(sc_p.read_text(encoding="utf-8"))
        try:
            import onnx

            graph = onnx.load(str(onnx_p)).graph
            real_in = graph.input[0].name
            real_out = [o.name for o in graph.output]
            decl_in = (sc.get("input") or {}).get("name")
            decl_out = (sc.get("outputs") or {}).get("names") or []
            if real_in != decl_in:
                ok = False
                notes.append(f"  ✗ {rel}: 输入名不一致 图={real_in!r} sidecar={decl_in!r}")
            else:
                notes.append(f"  ✓ {rel}: 输入名 {real_in!r} 一致")
            # 声明的输出名必须是图输出的**前缀**（RT-DETR 图会多带中间张量，属正常）
            missing = [n for n in decl_out if n not in real_out]
            if missing:
                ok = False
                notes.append(f"  ✗ {rel}: sidecar 声明的输出 {missing} 不在图中（图输出前几个：{real_out[:4]}）")
            else:
                notes.append(f"  ✓ {rel}: 声明的输出名 {decl_out} 都在图中")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"  ⚠️ {rel}: 读图失败 {exc}")

        if need_ort:
            try:
                import numpy as np
                import onnxruntime as ort

                sess = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
                iname = sess.get_inputs()[0].name
                shape = sess.get_inputs()[0].shape
                hw = [d for d in shape[2:] if isinstance(d, int)] or [512, 512]
                x = np.zeros((1, 3, int(hw[0]), int(hw[1])), dtype=np.float32)
                sess.run(None, {iname: x})
                notes.append(f"  ✓ {rel}: onnxruntime 加载并前向一次成功（输入 {hw}）")
            except Exception as exc:  # noqa: BLE001
                ok = False
                notes.append(f"  ✗ {rel}: onnxruntime 前向失败 {type(exc).__name__}: {str(exc)[:120]}")
    return ok, notes


def collect(include_weights: bool) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []

    def add_file(p: Path, arc: str):
        if p.name in DROP_NAMES or p.suffix in DROP_SUFFIX:
            return
        if "__pycache__" in p.parts:
            return
        items.append((p, arc))

    for rel in INCLUDE_FILES:
        p = ROOT / rel
        if p.is_file():
            # 只取文件名放到包根：INCLUDE_FILES 里可能写成 docs/xxx.md（源文件在子目录），
            # 但接收方需要在解压第一眼就看到它，所以归档路径用 basename。
            add_file(p, p.name)

    for rel in INCLUDE_DIRS:
        d = ROOT / rel
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file():
                add_file(p, str(p.relative_to(ROOT)).replace("\\", "/"))

    keep = MODEL_KEEP_WITH_WEIGHTS if include_weights else MODEL_KEEP
    for rel in MODEL_DIRS:
        d = ROOT / rel
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.name in keep:
                add_file(p, f"{rel}/{p.name}")

    # 示例图：让接收方开箱即可自测（见 SAMPLE_IMAGES_* 的说明）
    src_dir = ROOT / SAMPLE_IMAGES_FROM
    if src_dir.is_dir():
        # 排除测试夹具：`tests/test_contract.py` 会往图片目录写 `contract_test_field.jpg`
        # 之类的夹具图，本机实测它会被选进示例图，然后出现在接收方的自测输出里
        # （表现为一行 conf=0.0、框数=1 的怪结果），看起来像服务有问题。
        def _is_fixture(p: Path) -> bool:
            low = p.name.lower()
            return "contract" in low or "fixture" in low or low.startswith("test")

        picked = [p for p in sorted(src_dir.glob("*.jpg")) if not _is_fixture(p)]
        picked = picked[:SAMPLE_IMAGE_COUNT]
        for p in picked:
            add_file(p, f"{SAMPLE_IMAGES_TO}/{p.name}")
        if picked:
            readme = ROOT / "deliverables" / "_示例图说明.txt"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "本目录中的 3 张 640x640 图片用于验收自测。\n"
                "接口的 JSON 调用方式（{\"image_id\": \"<文件名>\"}）从 data/uploads/ 读图，\n"
                "因此这些图放在这里，收到交付包后可直接运行：\n\n"
                "  python scripts/smoke_http.py --port 8077\n\n"
                "来验证服务可用（会测 /health、JSON、multipart、错误路径与延迟）。\n",
                encoding="utf-8",
            )
            items.append((readme, f"{SAMPLE_IMAGES_TO}/_示例图说明.txt"))

    # 生成随包 .env（脱敏；见 DELIVERY_ENV 的说明）
    env_tmp = ROOT / "deliverables" / "_delivery.env"
    env_tmp.parent.mkdir(parents=True, exist_ok=True)
    env_tmp.write_text(DELIVERY_ENV, encoding="utf-8")
    items.append((env_tmp, ".env"))
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成交付压缩包")
    ap.add_argument("--out-dir", default="deliverables")
    ap.add_argument("--name", default="drone-crop-disease-algo_交付包")
    ap.add_argument("--with-weights", action="store_true",
                    help="额外包含 best.pt / last.pt（约 +241 MB，仅重训需要）")
    ap.add_argument("--skip-clean", action="store_true", help="不清理仓库垃圾")
    args = ap.parse_args(argv)

    print("=" * 74)
    print("交付包生成")
    print("=" * 74)

    if not args.skip_clean:
        removed = clean_junk()
        if removed:
            print("【清理仓库垃圾】")
            for r in removed:
                print(f"  · {r}")
        else:
            print("【清理仓库垃圾】无需清理")

    print("\n【交付前自检】")
    ok, notes = self_check()
    for n in notes:
        print(n)
    if not ok:
        print("\n[中止] 自检未通过，先修复再打包（否则会把静默错误交付出去）")
        return 1

    items = collect(args.with_weights)
    total = sum(p.stat().st_size for p, _ in items)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_含权重" if args.with_weights else ""
    zip_path = out_dir / f"{args.name}{suffix}.zip"

    print(f"\n【打包】{len(items)} 个文件，原始 {total/(1024*1024):.1f} MB")
    # 用 ZIP_STORED（不压缩）：ONNX 权重几乎压不动（实测压缩率仅 92%，即只省 8%），
    # 而 deflate 处理 161 MB 权重极慢 —— 本机实测卡住 50 分钟未完成。
    # 交付靠 U 盘/网盘传输时，8% 的体积收益不值得这个耗时。
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for p, arc in items:
            z.write(p, arc)
    zsize = zip_path.stat().st_size / (1024 * 1024)
    print(f"  已生成：{zip_path}")
    print(f"  包体积：{zsize:.1f} MB（不压缩存储，避免 ONNX 上浪费 CPU）")

    # 按顶层目录汇总，便于人工核对
    print("\n【包内构成】")
    agg: dict[str, list[int]] = {}
    for p, arc in items:
        top = arc.split("/")[0] if "/" in arc else "(根文件)"
        a = agg.setdefault(top, [0, 0])
        a[0] += 1
        a[1] += p.stat().st_size
    for top, (cnt, sz) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {top:<28} {cnt:>4} 个文件  {sz/(1024*1024):>8.1f} MB")

    print("\n【必查项】")
    names = {arc for _, arc in items}
    checks = [
        ("models/det_rtdetrv2/model.onnx", "检测 ONNX"),
        ("models/det_rtdetrv2/model.onnx.json", "检测 sidecar"),
        ("models/cls_dinov2/model.onnx", "分类 ONNX"),
        ("models/cls_dinov2/model.onnx.json", "分类 sidecar"),
        ("src/api/app.py", "服务入口"),
        ("scripts/smoke_http.py", "HTTP 冒烟脚本"),
        ("tests/run_all.py", "回归测试入口"),
        ("docs/接口联调说明.md", "联调文档"),
    ]
    all_ok = True
    for rel, label in checks:
        hit = rel in names
        all_ok &= hit
        print(f"  {'✓' if hit else '✗'} {label:<16} {rel}")
    print("\n" + "=" * 74)
    print("✓ 交付包就绪" if all_ok else "✗ 缺少必查项")
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
