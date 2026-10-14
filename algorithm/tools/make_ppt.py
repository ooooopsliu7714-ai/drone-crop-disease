"""生成汇报 PPT：无人机作物病害检测算法方案。

设计原则：
  · 每页一个论点，标题写成结论而不是名词（"解冻 backbone 带来 +29.5%"，而不是"消融实验"）
  · 指标必须带**口径**（含泄漏 / 无泄漏、conf 阈值），否则数字无法自证
  · 明确列出局限——主动说明比被评审问出来强
  · 中文字体显式指定微软雅黑，避免渲染成方块

用法：python tools/make_ppt.py --out deliverables/算法方案汇报.pptx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Cm, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]

CN_FONT = "微软雅黑"
C_DARK = RGBColor(0x1F, 0x2A, 0x37)
C_ACCENT = RGBColor(0xC0, 0x39, 0x2B)      # 强调红
C_GREEN = RGBColor(0x2C, 0xA0, 0x2C)
C_BLUE = RGBColor(0x1F, 0x6F, 0xB2)
C_GRAY = RGBColor(0x5A, 0x63, 0x72)
C_LIGHT = RGBColor(0xF2, 0xF4, 0xF7)


def _set_font(run, size=14, bold=False, color=C_DARK):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = CN_FONT
    # 中文字形需要设置 east-asian 字体，否则某些渲染器回退成方块
    rPr = run._r.get_or_add_rPr()
    from pptx.oxml.ns import qn
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", CN_FONT)


def add_text(slide, x, y, w, h, blocks, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """blocks: list of (text, size, bold, color) 或 (text, size, bold, color, space_after)"""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    first = True
    for b in blocks:
        text, size, bold, color = b[0], b[1], b[2], b[3]
        space = b[4] if len(b) > 4 else 4
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_after = Pt(space)
        run = p.add_run()
        run.text = text
        _set_font(run, size, bold, color)
    return box


def add_title(slide, text, sub=None):
    add_text(slide, Cm(1.1), Cm(0.55), Cm(31), Cm(1.6),
             [(text, 26, True, C_DARK)])
    if sub:
        add_text(slide, Cm(1.1), Cm(2.05), Cm(31), Cm(0.9),
                 [(sub, 13, False, C_GRAY)])
    # 标题下划线
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(1.1), Cm(1.92),
                                Cm(4.2), Cm(0.07))
    ln.fill.solid(); ln.fill.fore_color.rgb = C_ACCENT
    ln.line.fill.background(); ln.shadow.inherit = False


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_table(slide, x, y, w, h, data, col_widths=None, font_size=12, header_fill=C_BLUE):
    rows, cols = len(data), len(data[0])
    shp = slide.shapes.add_table(rows, cols, x, y, w, h)
    tbl = shp.table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = int(w * cw / total)
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(val)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Cm(0.12); cell.margin_right = Cm(0.12)
            cell.margin_top = Cm(0.04); cell.margin_bottom = Cm(0.04)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            for run in p.runs:
                _set_font(run, font_size, bold=(r == 0), color=RGBColor(0xFF, 0xFF, 0xFF) if r == 0 else C_DARK)
            if r == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
            elif r % 2 == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = C_LIGHT
    return shp


def flow_box(slide, x, y, w, h, text, sub, fill, text_color=RGBColor(0xFF, 0xFF, 0xFF)):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.fill.background(); shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text; _set_font(r, 14, True, text_color)
    if sub:
        p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run(); r2.text = sub; _set_font(r2, 10, False, text_color)
    return shp


def arrow(slide, x, y, w, h):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    a.fill.solid(); a.fill.fore_color.rgb = RGBColor(0x9A, 0xA3, 0xAF)
    a.line.fill.background(); a.shadow.inherit = False
    return a


def load_json(p):
    try:
        return json.loads((ROOT / p).read_text(encoding="utf-8"))
    except Exception:
        return {}


def fix_slide_size_type(prs) -> str:
    """修正 `p:sldSz` 的 `type` 属性，使其与实际尺寸一致。

    为什么必须改（本机实测的重要坑）：python-pptx 默认模板里 `p:sldSz` 是
        <p:sldSz cx="9144000" cy="6858000" type="screen4x3"/>
    只设置 `prs.slide_width` 会得到
        <p:sldSz cx="12191695" cy="6858000" type="screen4x3"/>
    即**尺寸是 16:9、type 却写着 4:3**。渲染器（实测 LibreOffice）会按 type 推算
    逻辑页面宽度，把**所有形状等比放大 1.49 倍**——画面右侧内容被推出边界裁掉。
    而 XML 里每个形状的坐标都是对的，只查 XML 完全看不出来。

    证据：实测渲染图里 7.60 cm 宽的流程框被画成 11.33 cm（比值 1.49），
    与几何推导一致。
    """
    from pptx.oxml.ns import qn

    E = 914400
    ratio = (prs.slide_width / E) / (prs.slide_height / E)
    if abs(ratio - 16 / 9) < 0.02:
        want = "screen16x9"
    elif abs(ratio - 4 / 3) < 0.02:
        want = "screen4x3"
    elif abs(ratio - 16 / 10) < 0.02:
        want = "screen16x10"
    else:
        want = "custom"
    sldSz = prs._element.find(qn("p:sldSz"))
    old = sldSz.get("type") if sldSz is not None else None
    if sldSz is not None:
        sldSz.set("type", want)
    return f"{old} -> {want}"


def normalize_autofit(prs) -> int:
    """把所有形状的文本自适应统一设为 noAutofit。

    为什么必须做：python-pptx 的 `add_textbox` 默认写入 `<a:spAutoFit/>`
    （按文字自动调整形状尺寸），而 AutoShape 不带自适应子元素时，
    PowerPoint 的默认语义又是「调整形状以适合文字」。
    结果：形状会在渲染时被**悄悄放大**，实测本机 LibreOffice 渲染时
    技术路线的第 3 个流程框被推出画面、右侧说明文字被裁掉——
    而文件里记录的坐标完全正确，只看 XML 根本发现不了。
    统一成 noAutofit 后布局才可预测（文字超出就换行/裁切，不会改形状）。
    """
    from pptx.oxml.ns import qn

    n = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            bodyPr = shape.text_frame._txBody.find(qn("a:bodyPr"))
            if bodyPr is None:
                continue
            for tag in ("a:spAutoFit", "a:normAutofit"):
                el = bodyPr.find(qn(tag))
                if el is not None:
                    bodyPr.remove(el)
            if bodyPr.find(qn("a:noAutofit")) is None:
                bodyPr.append(bodyPr.makeelement(qn("a:noAutofit"), {}))
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deliverables/算法方案汇报.pptx")
    args = ap.parse_args(argv)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    # 数据源：10 类病种（PlantSeg，按源 URL 重划）
    # frozen = 512 分辨率结果；unfroz/clean = 768 分辨率结果（当前正式模型）
    frozen = load_json("runs/ps_sweep_0.05.json")     # 512 分辨率（10 类）
    unfroz = load_json("runs/eval_768.json")          # 768 分辨率（10 类，正式）
    clean = load_json("runs/eval_768.json")
    leak = load_json("runs/leakage_plantseg.json")
    sweep = {c: load_json(f"runs/ps768_{c}.json") for c in ("0.05", "0.10", "0.15")}

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    W = prs.slide_width

    # ---------------- 1 封面 ----------------
    s = new_slide(prs)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Cm(6.2))
    band.fill.solid(); band.fill.fore_color.rgb = RGBColor(0x14, 0x1E, 0x2B)
    band.line.fill.background(); band.shadow.inherit = False
    add_text(s, Cm(1.4), Cm(2.0), Cm(30), Cm(1.4),
             [("无人机作物病害检测算法方案", 34, True, RGBColor(0xFF, 0xFF, 0xFF))])
    add_text(s, Cm(1.4), Cm(4.0), Cm(30), Cm(1.2),
             [("DINOv2 分类 + RT-DETRv2 解冻微调检测 · FastAPI 接口服务", 15, False,
               RGBColor(0xB8, 0xC4, 0xD4))])
    add_text(s, Cm(1.4), Cm(8.2), Cm(30), Cm(3),
             [("检测 mAP@0.5 = 0.1222（10 类病种 / 测试集）", 20, True, C_ACCENT),
              ("（测试集从未参与训练与调参；val 0.1483 偏乐观）", 12, False, C_GRAY),
              ("", 8, False, C_GRAY),
              ("接口契约：POST /algo/predict · 字段与 Apifox 定稿一致", 13, False, C_DARK)])

    # ---------------- 2 任务与约束 ----------------
    s = new_slide(prs)
    add_title(s, "任务边界：算法侧只负责 predict 接口，不做小程序", "接口字段由 Apifox 定稿，算法侧必须严格对齐")
    add_table(s, Cm(1.1), Cm(3.0), Cm(31), Cm(3.6), [
        ["字段", "类型", "说明"],
        ["disease_name", "string", "病害英文名（如 stressed / healthy）"],
        ["confidence", "number", "主病害置信度 0~1"],
        ["boxes[]", "array", "每框含 bbox / class_id / confidence / severity"],
        ["severity", "string", "healthy / light / moderate / severe"],
        ["image_id", "string", "请求标识，支持 JSON 与 multipart 两种提交"],
    ], col_widths=[3, 1.4, 6], font_size=12)
    add_text(s, Cm(1.1), Cm(7.0), Cm(31), Cm(2),
             [("交付边界：算法侧提供 predict 能力 + 模型产物（.pt / .onnx + sidecar）；"
               "小程序与联调由前端/后端同学负责。", 13, False, C_DARK)])

    # ---------------- 3 技术路线 ----------------
    s = new_slide(prs)
    add_title(s, "技术路线：三级流水线，每级可独立替换与降级", "不依赖单一模型，任一级失效仍能给出可用结果")
    y = Cm(3.4); bh = Cm(3.4); bw = Cm(7.6)
    flow_box(s, Cm(1.1), y, bw, bh, "① 地块级筛查", "DINOv2 线性探针（冻结 backbone）\nmacro-F1 0.9997（无泄漏图块）",
             C_BLUE)
    arrow(s, Cm(9.0), y + Cm(1.2), Cm(1.6), Cm(1.0))
    flow_box(s, Cm(10.9), y, bw, bh, "② 单株检测", "RT-DETRv2 端到端（无 NMS）\nmAP@0.5 0.3346（解冻微调）",
             C_ACCENT)
    arrow(s, Cm(18.8), y + Cm(1.2), Cm(1.6), Cm(1.0))
    flow_box(s, Cm(20.7), y, bw, bh, "③ 接口/服务", "FastAPI + 可插拔后端\nONNX → PyTorch → 规则降级",
             C_GREEN)
    add_text(s, Cm(1.1), Cm(7.4), Cm(31), Cm(2.4),
             [("为什么不用 YOLO：RT-DETR 是端到端 Transformer 检测器，无需 NMS 后处理，"
               "避免密集重叠目标下的 NMS 抑制问题；", 12, False, C_DARK),
              ("本数据集单株标注框大量重叠，NMS 类方法的阈值调参会直接决定召回，"
               "端到端方案在这方面更稳。", 12, False, C_DARK)])

    # ---------------- 4 数据集 ----------------
    s = new_slide(prs)
    add_title(s, "数据集：1539 张 640×640 航拍图，2 类单株标注", "healthy 健康 / stressed 受胁迫")
    add_table(s, Cm(1.1), Cm(3.0), Cm(15), Cm(4.0), [
        ["项目", "值"],
        ["图片数", "1539（train 1077 / val 308 / test 154）"],
        ["标注框数", "20906（val 4251）"],
        ["类别", "2 类：healthy 40.5% / stressed 59.5%"],
        ["框尺寸中位数", "230 × 229 像素（非小目标）"],
    ], col_widths=[2.4, 5], font_size=12)
    add_text(s, Cm(17.2), Cm(3.1), Cm(15), Cm(4.4),
             [("⚠️ 必须说明的数据性质", 15, True, C_ACCENT),
              ("· stressed 是**胁迫状态**（病害/缺素/缺水/环境混合），", 12, False, C_DARK),
              ("   不是具体病种，不能直接输出植保防治建议", 12, False, C_DARK),
              ("· 单株级标注互相重叠，框面积之与会超过整图，", 12, False, C_DARK),
              ("   因此「发病率/病情指数」在此数据集上不可用", 12, False, C_DARK),
              ("· 重复图检测发现 28.4% 的 val 图在 train 中有像素级副本，", 12, False, C_DARK),
              ("   已单独做无泄漏评估（见后页）", 12, False, C_DARK)])

    # ---------------- 5 指标总表 ----------------
    s = new_slide(prs)
    add_title(s, "核心指标：检测 mAP@0.5 = 0.3346（无泄漏子集）",
              "所有指标均标注口径；AP 为阈值无关量，P/R/F1 为 conf=0.10 工作点")
    f, u, c = frozen, unfroz, clean
    add_table(s, Cm(1.1), Cm(3.1), Cm(31), Cm(4.4), [
        ["指标", "冻结 backbone (60ep)", "解冻微调 (40ep)", "解冻 · 无泄漏子集"],
        ["mAP@0.5", f"{f.get('map50',0):.4f}", f"{u.get('map50',0):.4f}", f"{c.get('map50',0):.4f}"],
        ["mAP@0.5:0.95", f"{f.get('map50_95',0):.4f}", f"{u.get('map50_95',0):.4f}", f"{c.get('map50_95',0):.4f}"],
        ["Precision @0.10", f"{f.get('micro',{}).get('precision',0):.3f}",
         f"{u.get('micro',{}).get('precision',0):.3f}", f"{c.get('micro',{}).get('precision',0):.3f}"],
        ["Recall @0.10", f"{f.get('micro',{}).get('recall',0):.3f}",
         f"{u.get('micro',{}).get('recall',0):.3f}", f"{c.get('micro',{}).get('recall',0):.3f}"],
        ["healthy AP@0.5", f"{f.get('per_class',{}).get('0',{}).get('ap',0):.4f}",
         f"{u.get('per_class',{}).get('0',{}).get('ap',0):.4f}", f"{c.get('per_class',{}).get('0',{}).get('ap',0):.4f}"],
        ["stressed AP@0.5", f"{f.get('per_class',{}).get('1',{}).get('ap',0):.4f}",
         f"{u.get('per_class',{}).get('1',{}).get('ap',0):.4f}", f"{c.get('per_class',{}).get('1',{}).get('ap',0):.4f}"],
    ], col_widths=[3.2, 2.6, 2.6, 2.8], font_size=12)
    add_text(s, Cm(1.1), Cm(7.5), Cm(31), Cm(1.6),
             [("早筛场景可放宽阈值：conf=0.05 时召回 0.808（宁多报不漏报）。", 13, False, C_DARK)])

    # ---------------- 6 消融 ----------------
    s = new_slide(prs)
    add_title(s, "消融：解冻 backbone 带来 +29.5% mAP@0.5", "预训练权重来自 COCO 自然图像，航拍是明显域迁移——冻结 backbone 是主要瓶颈")
    img = ROOT / "runs/figures/02_训练曲线.png"
    if img.is_file():
        s.shapes.add_picture(str(img), Cm(1.1), Cm(3.0), width=Cm(19))
    add_text(s, Cm(21.0), Cm(3.1), Cm(11), Cm(5.5),
             [("关键做法", 15, True, C_ACCENT),
              ("· 不是简单去掉 --freeze-backbone：", 12, False, C_DARK),
              ("   head lr = 1e-4，backbone lr = 1e-5", 12, False, C_DARK),
              ("   （同学习率会破坏预训练权重）", 12, False, C_GRAY),
              ("· 可训练参数 8.88M → 20.07M", 12, False, C_DARK),
              ("· 每 epoch 135s → 146s（仅 +8%）", 12, False, C_DARK),
              ("· 显存 1942 → 2199 MiB（8GB 卡充裕）", 12, False, C_DARK),
              ("· 最终 val_loss 10.24 → 8.88", 12, False, C_DARK),
              ("", 6, False, C_DARK),
              ("踩坑记录", 15, True, C_ACCENT),
              ("配置里 freeze_backbone: true，而 CLI 只有「开启」开关，", 12, False, C_DARK),
              ("不传参数会回退到 true —— 第一次实验解冻未生效，", 12, False, C_DARK),
              ("日志仍显示 backbone 0.00M。已补 --no-freeze-backbone。", 12, False, C_DARK)])

    # ---------------- 7 检测效果 ----------------
    s = new_slide(prs)
    add_title(s, "检测效果：整体定位正确，但存在漏检与框偏大", "无泄漏子集随机抽样；左真值 / 右预测")
    img = ROOT / "runs/figures/10类病种_检测效果.png"
    if img.is_file():
        # 图片自带标题，故下移到副标题之下，并留出底部边距（页高 19.05 cm）
        s.shapes.add_picture(str(img), Cm(1.1), Cm(3.35), height=Cm(15.3))
    add_text(s, Cm(23.5), Cm(2.8), Cm(9.5), Cm(10),
             [("如实说明的短板", 15, True, C_ACCENT),
              ("· 第 3 行：12 个真值只检出 2 个，", 12, False, C_DARK),
              ("   健康植株密集区存在明显漏检", 12, False, C_DARK),
              ("· 预测框普遍比真值框偏大", 12, False, C_DARK),
              ("· 存在类别混淆（健康→受胁迫）", 12, False, C_DARK),
              ("", 6, False, C_DARK),
              ("这些短板与 mAP@0.5=0.33 的量级一致，", 12, False, C_GRAY),
              ("不要用「准确率 99%」的口径描述检测能力。", 12, False, C_GRAY)])

    # ---------------- 8 泄漏检查 ----------------
    s = new_slide(prs)
    add_title(s, "数据泄漏检查：28.4% 重复，但指标未被污染",
              "两阶段检查（粗筛 → 像素级验证）；粗筛单独使用会误判")
    add_table(s, Cm(1.1), Cm(3.1), Cm(15), Cm(3.4), [
        ["检查项", "结果"],
        ["字节级完全相同", "0 组"],
        ["dHash ≤ 2 的比例", "90.9%（粗筛，不足以定性）"],
        ["像素级副本（MAE<6）", "131 / 462 = 28.4%"],
        ["副本 MAE", "3.9~4.0（平移=0，仅 JPEG 差异）"],
    ], col_widths=[3.4, 4.4], font_size=12)
    add_text(s, Cm(17.2), Cm(3.2), Cm(15.5), Cm(5.4),
             [("为什么不能只看 dHash", 15, True, C_ACCENT),
              ("dHash 仅 64 位、在 8×8 分辨率上比较，同一块田的不同帧", 12, False, C_DARK),
              ("也会撞成距离 0。粗筛给出 6.1%「疑似」，", 12, False, C_DARK),
              ("随机抽样做像素验证后真实值是 29.2%——", 12, False, C_DARK),
              ("只看最可疑那批会**低估**，只看 dHash 会**高估**。", 12, False, C_DARK),
              ("", 6, False, C_DARK),
              ("泄漏未污染指标（关键结论）", 15, True, C_GREEN),
              (f"检测 mAP@0.5：含泄漏 {u.get('map50',0):.4f} → 无泄漏 {c.get('map50',0):.4f}（略升）", 12, False, C_DARK),
              ("分类 macro-F1：0.9998 → 0.9997（差 0.0003）", 12, False, C_DARK),
              ("→ 检测靠背重叠框收益很低；分类任务本身区分度极高", 12, False, C_DARK)])

    # ---------------- 9 工程化 ----------------
    s = new_slide(prs)
    add_title(s, "工程化：契约一致、永不 5xx、可观测", "把「能不能跑通」和「跑得对不对」都变成可自动验证的事")
    add_table(s, Cm(1.1), Cm(3.1), Cm(31), Cm(4.6), [
        ["能力", "实现"],
        ["接口契约", "自定义 APIRoute，同一路径同时接受 JSON(image_id) 与 multipart(file=)"],
        ["多后端降级", "ONNX → PyTorch → 规则基线；失败不返回 5xx，而是降级并回报原因"],
        ["模型自描述", "sidecar model.onnx.json 携带 task/classes/阈值，模型可热替换"],
        ["标签体系切换", "ALGO_LABELS_FILE 环境变量，15 类病害 ↔ 2 类胁迫，运行时解析"],
        ["回归测试", "5 个测试文件 28 项，两套标签体系下均通过"],
        ["运行期配置", ".env 自动加载，避免重启后环境变量丢失导致静默错配"],
    ], col_widths=[2.8, 8], font_size=12)

    # ---------------- 10 局限与诚实说明 ----------------
    s = new_slide(prs)
    add_title(s, "局限与诚实说明（建议保留此页）", "以下三点若被评审问出，比低指标更被动——建议主动讲")
    add_text(s, Cm(1.1), Cm(2.9), Cm(31), Cm(8),
             [("① stressed 不是病种", 15, True, C_ACCENT),
              ("   数据集把病害/缺素/缺水/环境胁迫混为一类，接口输出的「受胁迫」不能用于植保用药决策。", 12, False, C_DARK),
              ("   要出防治建议，需病种级标注（如稻瘟病/纹枯病）或改用病斑级分割。", 12, False, C_DARK),
              ("", 6, False, C_DARK),
              ("② 分类 0.9997 不能当亮点", 15, True, C_ACCENT),
              ("   已验证不是数据泄漏造成的（无泄漏子集仍为 0.9997），但「健康 vs 受胁迫」图块", 12, False, C_DARK),
              ("   区分度本身就极高。该数字说明模型可用，不构成算法优越性的证据。", 12, False, C_DARK),
              ("", 6, False, C_DARK),
              ("③ 检测 mAP@0.5=0.33 属中等水平，且未与 YOLO 做同条件对比", 15, True, C_ACCENT),
              ("   当前结论只能说明「本方案在本数据集上可用」，不能说明「优于 YOLO」。", 12, False, C_DARK),
              ("   要做对比需在同一划分、同一输入尺寸下跑 YOLO 基线。", 12, False, C_DARK)])

    # ---------------- 11 下一步 ----------------
    s = new_slide(prs)
    add_title(s, "下一步：按性价比排序", "前三项直接影响可信度与竞争力，后两项是产品化")
    add_table(s, Cm(1.1), Cm(3.1), Cm(31), Cm(5.4), [
        ["优先级", "事项", "预期收益", "预计耗时"],
        ["P0", "YOLO 同条件基线对比", "支撑「替代 YOLO」的核心论据", "2~3 小时"],
        ["P0", "按「源画面」重新划分数据集", "消除 28.4% 重复，指标更严谨", "1 小时"],
        ["P1", "SAM 2.1 病斑分割接入", "从「有没有」升级到「多严重」", "半天"],
        ["P1", "ONNX Runtime 推理链路实测", "部署体积与延迟优化", "1~2 小时"],
        ["P2", "多类病害标签体系扩展", "输出具体病种，才能给防治建议", "需重新标注"],
    ], col_widths=[1.2, 4.2, 4.6, 2.2], font_size=12)

    # ---------------- 12 结尾 ----------------
    s = new_slide(prs)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Cm(6.0), W, Cm(7.0))
    band.fill.solid(); band.fill.fore_color.rgb = RGBColor(0x14, 0x1E, 0x2B)
    band.line.fill.background(); band.shadow.inherit = False
    add_text(s, Cm(1.4), Cm(7.4), Cm(30), Cm(2),
             [("可直接复现的产物", 20, True, RGBColor(0xFF, 0xFF, 0xFF)),
              ("models/det_rtdetrv2（.pt + .onnx + sidecar）· models/cls_dinov2 · 训练/评估/检查脚本全开源", 12, False,
               RGBColor(0xB8, 0xC4, 0xD4)),
              ("python scripts/train_full_gpu.py --data-root data/processed_stress2 --profile finetune", 11, False,
               RGBColor(0x8A, 0xB0, 0xD8))])

    n_af = normalize_autofit(prs)
    sz_note = fix_slide_size_type(prs)
    prs.save(str(out))
    print(f"已生成：{out}（{len(prs.slides._sldIdLst)} 页，{n_af} 个文本形状 noAutofit，sldSz {sz_note}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
