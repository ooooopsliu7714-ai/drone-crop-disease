"""生成交付清单：按部署形态列出该给/不该给的文件，并附 SHA256 版本指纹。

为什么要做这个：模型文件**不自证语义**。一个 .onnx 配一个 sidecar，再由代码解释
sidecar；三者版本不一致时会**静默出错**而不报错——本机实测就发生过：
导出的图输出名与 sidecar 声明不符、解码用了 softmax 而非 sigmoid、模型发现按 mtime
选错成分类模型，全都返回 HTTP 200 和合法 JSON，只是结果是错的。

所以交付的关键不是"把文件拷过去"，而是**让双方能确认拿到的是同一份**。
本脚本给出：
  1. 两种部署形态下分别该交付什么；
  2. 每个关键文件的 SHA256 指纹（双方各算一次即可确认一致）；
  3. 明确列出**不应交付**的目录（含一个必须删掉的坏模型）。

用法：
    python scripts/make_delivery_manifest.py --out deliverables/交付清单.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: 部署一个算法服务所需的最小文件集（相对仓库根）
SERVE_ESSENTIAL = [
    "models/det_rtdetrv2/model.onnx",
    "models/det_rtdetrv2/model.onnx.json",
    "models/cls_dinov2/model.onnx",
    "models/cls_dinov2/model.onnx.json",
]
#: 只用于重新训练/微调，部署不需要
TRAIN_ONLY = [
    "models/det_rtdetrv2/best.pt",
    "models/det_rtdetrv2/last.pt",
    "models/cls_dinov2/best.pt",
]
#: 对话/对比用，交付不必带
NOT_FOR_DELIVERY = [
    "models/_baseline_det_rtdetrv2_frozen60",
    "models/_BROKEN_det_rtdetrv2_训练目标编码错误_勿用",
]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _size_mb(path: Path) -> str:
    return f"{path.stat().st_size / (1024*1024):.1f} MB"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "(无 git 提交)"
    except Exception:
        return "(非 git 仓库或 git 不可用)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deliverables/交付清单.md")
    args = ap.parse_args(argv)

    det_sc = json.loads((ROOT / "models/det_rtdetrv2/model.onnx.json").read_text(encoding="utf-8"))
    cls_sc = json.loads((ROOT / "models/cls_dinov2/model.onnx.json").read_text(encoding="utf-8"))

    rows = []
    for rel in SERVE_ESSENTIAL + TRAIN_ONLY:
        p = ROOT / rel
        if p.is_file():
            rows.append((rel, _size_mb(p), _sha256(p), "部署必需" if rel in SERVE_ESSENTIAL else "仅重训需要"))

    lines = [
        "# 交付清单（算法侧 → 后端/部署）",
        "",
        "> 由 `python scripts/make_delivery_manifest.py` 生成。",
        "> **核对方式**：双方各自对同名文件算一次 SHA256，指纹一致即确认拿到同一份产物。",
        "",
        "## 〇、先选部署形态（这决定要交付什么）",
        "",
        "| 形态 | 谁跑算法服务 | 需要交付什么 | 风险 |",
        "|---|---|---|---|",
        "| **A（推荐）** | 算法侧自己部署 | **只需一个 URL**，不给任何模型文件 | 最低；模型与代码始终同版本 |",
        "| **B** | 后端部署 | 整个仓库（`src/` + `scripts/` + `models/` 下的 ONNX + sidecar） | 中；必须整包交付，不能只给模型 |",
        "",
        "**为什么 B 形态不能只给 `.onnx` 文件**：模型文件不自证语义——",
        "sidecar 描述它，代码解释 sidecar。三者版本不一致时会**静默给出错误结果**而不是报错。",
        "本项目实测踩过 5 个这类坑，全部返回 HTTP 200 + 合法 JSON：",
        "图输出名与 sidecar 不符、解码用 softmax 而非 sigmoid、模型发现按时间选错成分类模型等。",
        "**只发模型文件、配上旧代码，等于把这 5 个坑原样复制给对方。**",
        "",
        "## 一、部署必需文件（形态 B 才需要）",
        "",
        "| 文件 | 体积 | SHA256 | 用途 |",
        "|---|---|---|---|",
    ]
    for rel, size, sha, use in rows:
        lines.append(f"| `{rel}` | {size} | `{sha[:16]}…` | {use} |")

    lines += [
        "",
        "完整指纹：",
        "",
        "```text",
    ]
    for rel, size, sha, use in rows:
        lines.append(f"{sha}  {rel}")
    lines += ["```", ""]

    lines += [
        "## 二、模型自描述信息（交付时请一并对齐）",
        "",
        "| 项 | det_rtdetrv2 | cls_dinov2 |",
        "|---|---|---|",
        f"| task | `{det_sc.get('task')}` | `{cls_sc.get('task')}` |",
        f"| 输入名 / 尺寸 | `{det_sc['input'].get('name')}` / {det_sc['input'].get('size')} | "
        f"`{cls_sc['input'].get('name')}` / {cls_sc['input'].get('size')} |",
        f"| 输出 kind | `{det_sc['outputs'].get('kind')}` | `{cls_sc['outputs'].get('kind')}` |",
        f"| 输出名 | `{det_sc['outputs'].get('names')}` | `{cls_sc['outputs'].get('names')}` |",
        f"| 置信度阈值 | `{det_sc.get('meta', {}).get('conf_threshold')}` | "
        f"`{cls_sc.get('meta', {}).get('conf_threshold')}` |",
        f"| 类别数 | {len(det_sc.get('classes') or [])} | {len(cls_sc.get('classes') or [])} |",
        "",
        "## 三、**请勿交付**的内容",
        "",
        "| 路径 | 体积 | 原因 |",
        "|---|---|---|",
    ]
    for rel in NOT_FOR_DELIVERY:
        p = ROOT / rel
        if p.is_dir():
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            reason = ("对比基线，仅本机消融实验用" if "baseline" in rel
                      else "**训练目标编码错误的废模型，必须删除**")
            lines.append(f"| `{rel}` | {total/(1024*1024):.1f} MB | {reason} |")

    lines += [
        "",
        "`best.pt` / `last.pt` 只在重新训练/微调时需要；若后端只是部署推理，",
        "可以只给 ONNX + sidecar，省下约 246 MB。",
        "",
        "## 四、版本指纹",
        "",
        f"- git commit：`{_git_commit()}`",
        "- 标签体系：`configs/labels_stress2.yaml`（2 类：healthy / stressed）",
        "- 检测指标：val mAP@0.5 = 0.3346（无泄漏子集 224 图 / 3142 框）",
        "- 验证命令（对方可自行复现）：",
        "",
        "```bash",
        ".venv/Scripts/python.exe tests/run_all.py                # 29 项回归",
        ".venv/Scripts/python.exe scripts/smoke_http.py           # 真实 HTTP 冒烟",
        ".venv/Scripts/python.exe scripts/bench_onnx_vs_torch.py  # 两后端对拍",
        "```",
        "",
        "---",
        "",
        "## 五、一句话建议",
        "",
        "**优先选形态 A：算法服务由你部署，只把 URL 给后端。**",
        "这样模型、sidecar、解码代码永远是同一版本，不存在版本漂移问题；",
        "也避免把 234 MB 的废模型误交付出去。",
        "",
    ]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成：{out}")
    print(f"  部署必需文件 {len(SERVE_ESSENTIAL)} 个，指纹已写入")
    print(f"  明确标注不交付目录 {len(NOT_FOR_DELIVERY)} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
