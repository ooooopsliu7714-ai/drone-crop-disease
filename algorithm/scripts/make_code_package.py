"""生成"仅代码"包（用于上传到团队 GitHub 仓库的 算法/ 目录）。

为什么不能直接传交付包（本机实测体积）：
    .venv            7544 MB   绝对不能进 git
    .pt 权重          554 MB   不进
    ONNX 模型         315 MB   单文件 77/84 MB —— 网页上传限 25 MB，且会让仓库永久膨胀
    交付包 zip        165 MB   同上
    data 数据集       144 MB   仓库另有「数据」目录负责
    ---- 分界线 ----
    代码+配置+测试        0.86 MB
    文档                 0.09 MB
    PPT                  2 MB

结论：**代码进 git，模型走 GitHub Releases**（或直接用交付包发给队友）。
本脚本产出仅代码包，网页/CLI 都能轻松上传。

用法：
    python scripts/make_code_package.py --out-dir deliverables
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 进 git 的内容（白名单）
DIRS = ["src", "scripts", "configs", "tests", "tools", "docs"]
FILES = ["README.md", "REQUIREMENTS.md", "pyproject.toml", "部署指引.md"]
#: 模型侧只带上 sidecar（几 KB）——它描述模型契约，是代码的一部分；
#: ONNX/权重本身走 Releases。
SIDECARS = ["models/det_rtdetrv2/model.onnx.json", "models/cls_dinov2/model.onnx.json"]

GITIGNORE = """# ---- 环境与缓存 ----
.venv/
venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/

# ---- 大文件：走 GitHub Releases，不进仓库 ----
# ONNX 模型单文件 77~84 MB，接近 GitHub 100 MB 硬限；
# 网页上传更是只有 25 MB 上限。模型请用 Releases 附件分发。
models/**/*.onnx
models/**/*.pt
deliverables/*.zip

# ---- 数据与运行产物 ----
data/
runs/
*.log
tmp*/
torchinductor_*/

# ---- 本地机专属配置 ----
# .env 需要进仓库（它决定标签体系，属于配置而非密钥），
# 但里面不得出现机器专属绝对路径（如 HF_HOME=D:\\Users\\xxx\\...）。
# 若确实需要本机专属变量，写进 .env.local（已被忽略）。
.env.local
HF_HOME/
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="deliverables")
    ap.add_argument("--name", default="算法_仅代码_可传GitHub")
    args = ap.parse_args(argv)

    items: list[tuple[Path, str]] = []

    def add(p: Path, arc: str):
        if "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo", ".log"):
            return
        items.append((p, arc))

    for rel in FILES:
        p = ROOT / rel
        if p.is_file():
            add(p, p.name)
    for rel in DIRS:
        d = ROOT / rel
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    add(p, str(p.relative_to(ROOT)).replace("\\", "/"))
    for rel in SIDECARS:
        p = ROOT / rel
        if p.is_file():
            add(p, rel)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 同时写出 .gitignore 供团队仓库使用
    gi = out_dir / "_算法目录.gitignore"
    gi.write_text(GITIGNORE, encoding="utf-8")

    zip_path = out_dir / f"{args.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, arc in items:
            z.write(p, arc)
        z.write(gi, ".gitignore")

    total = sum(p.stat().st_size for p, _ in items)
    zsize = zip_path.stat().st_size
    print(f"已生成：{zip_path}")
    print(f"  {len(items)} 个文件，原始 {total/1024:.0f} KB，压缩后 {zsize/1024:.0f} KB")
    print(f"  .gitignore 也一并写入包内：{gi.name}")
    print("\n  包含的顶层目录：")
    tops: dict[str, int] = {}
    for _, arc in items:
        t = arc.split("/")[0] if "/" in arc else "(根文件)"
        tops[t] = tops.get(t, 0) + 1
    for t, c in sorted(tops.items()):
        print(f"    {t:<24} {c} 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
