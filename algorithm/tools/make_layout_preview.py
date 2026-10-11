"""为版式检查生成"预补偿"副本：绕开 LibreOffice Kit 对 16:9 页面的截取放大缺陷。

背景（本机实测）：LibreOffice Kit 渲染 13.333×7.5 in 的 PPTX 时，会把页面内容
放大 1.5 倍并只保留左上 2/3，因此画面右侧看起来"内容溢出被裁掉"。
用最小对照实验验证过：一个全新的、只含一个矩形的 PPTX 也会出现同样的放大，
说明**文件本身没问题，是渲染器缺陷**。

做法：把所有形状几何与字号同乘 1/k（k=1.5），渲染器的放大正好把它还原，
于是能拿到完整页面的真实版面用于检查。**这个副本只用于检查，不交付。**

用法：
    python tools/make_layout_preview.py --src deliverables/算法方案汇报.pptx \
        --out runs/_preview_src.pptx --k 1.5
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Pt


def scale_presentation(src: Path, dst: Path, k: float, keep_page_size: bool = True) -> dict:
    prs = Presentation(str(src))
    inv = 1.0 / k
    n_shapes = 0
    n_runs = 0

    # 关键：**默认保持页面尺寸不变**，只缩小内容。
    # 若连页面尺寸一起缩，渲染器仍会按它自己的逻辑放大并裁切，补偿就失效了
    # （本机实测：一起缩还会被裁掉右侧）。
    if not keep_page_size:
        prs.slide_width = Emu(int(prs.slide_width * inv))
        prs.slide_height = Emu(int(prs.slide_height * inv))

    for slide in prs.slides:
        for shape in slide.shapes:
            try:
                shape.left = Emu(int(shape.left * inv))
                shape.top = Emu(int(shape.top * inv))
                shape.width = Emu(int(shape.width * inv))
                shape.height = Emu(int(shape.height * inv))
                n_shapes += 1
            except Exception:
                pass
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    if run.font.size is not None:
                        run.font.size = Pt(max(1.0, run.font.size.pt * inv))
                        n_runs += 1

    prs.save(str(dst))
    return {"shapes": n_shapes, "runs": n_runs, "k": k,
            "slide_cm": (prs.slide_width / 360000, prs.slide_height / 360000)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=1.5, help="渲染器的放大倍数（实测 1.49~1.52）")
    args = ap.parse_args(argv)

    info = scale_presentation(Path(args.src), Path(args.out), args.k)
    print(f"已生成预补偿副本：{args.out}")
    print(f"  缩放 {info['shapes']} 个形状、{info['runs']} 个文字 run（1/{info['k']}）")
    print(f"  页面变为 {info['slide_cm'][0]:.2f} x {info['slide_cm'][1]:.2f} cm")
    print("  提示：该副本仅用于渲染检查，不要交付。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
