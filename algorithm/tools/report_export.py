#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估/统计结果导出工具（JSON → 多 sheet Excel + 可直接发队伍的中文 Markdown 简报）。

【它解决什么】
算法跑完一轮实验，指标散落在各种日志里，队伍看不到、答辩也没证据。
本工具把「总体指标 / 每类指标 / 地块统计 / 延迟预算 / 多实验对比」一次性整理成
一个 .xlsx（多 sheet）+ 一份 .md 简报，直接发群里或贴进答辩 PPT。

【输入 JSON 格式（字段名自定，但推荐用下面这套；缺失字段会自动跳过而不报错）】
单实验：
{
  "experiment": "rtdetrv2-r18-640",        # 实验名（可选）
  "model_version": "rtdetrv2-r18",         # 模型标识（可选）
  "dataset": "drone_crop_disease_v1/val",  # 数据集（可选）
  "eval_date": "2025-10-09",               # 评估日期（可选）
  "overall": {                             # 「总体指标」sheet，键名即指标名，值需为数值
     "mAP@0.5": 0.7123, "mAP@0.5:0.95": 0.4831,
     "precision": 0.7451, "recall": 0.6802, "f1": 0.7111,
     "false_alarm_rate": 0.0523,           # 误报率：农户最关心
     "disease_index_mae": 4.12             # 病情指数误差（可选）
  },
  "per_class": [                           # 「每类指标」sheet
     {"disease_name": "rice_blast", "disease_name_cn": "稻瘟病",
      "ap@0.5": 0.751, "precision": 0.78, "recall": 0.71, "f1": 0.743,
      "num_gt": 320, "num_pred": 300}
  ],
  "fields": [                              # 「地块统计」sheet
     {"field_id": "F001", "crop": "rice", "area_mu": 12.5, "total_boxes": 148,
      "disease_index": 23.4, "diseased_ratio": 0.0612,
      "top_disease": "rice_blast", "severity": "moderate"}
  ],
  "latency": {"cpu_ms_640": 1420.5, "gpu_ms_640": 178.2, "tile_4096_s": 26.4},
  "notes": ["随便写的备注，会进「实验信息」sheet 和简报"]
}
多实验对比（生成额外的「多实验对比」sheet，行=指标，列=实验）：
[ {上述对象1}, {上述对象2}, ... ]

【输出】
  <out>/report.xlsx   多 sheet 工作簿（总体指标 / 每类指标 / 地块统计 / 延迟预算 / 实验信息 / 多实验对比）
  <out>/report.md     中文简报

【关于 Excel 依赖（重要）】
优先用 openpyxl 写 xlsx（带列宽、加粗表头、冻结首行）。
**本机实测 openpyxl 未安装**，此时自动降级为内置的纯标准库 xlsx 写出器
（zipfile + XML，生成的是合法 .xlsx，Excel/WPS 可正常打开，只是没有单元格样式）。
若想启用 openpyxl 的更丰富样式，执行：pip install openpyxl
两级都不可用时才会退回 CSV，并打印中文提示。

【用法示例】
  python tools/report_export.py --help
  python tools/report_export.py --input runs/eval/eval_report.json --out runs/report
  python tools/report_export.py --input a.json b.json --out runs/report --title "第一轮消融"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

try:
    import openpyxl  # 可选：装了就用它的样式能力
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    HAS_OPENPYXL = True
except Exception:
    HAS_OPENPYXL = False


# ==========================================================================
# 一、纯标准库 XLSX 写出器（openpyxl 缺失时的降级方案）
# ==========================================================================
def _xml_escape(s: Any) -> str:
    """XML 文本转义。"""
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&apos;")
    )


def _col_letter(idx: int) -> str:
    """1 → A, 27 → AA。"""
    s = ""
    while idx > 0:
        idx, r = divmod(idx - 1, 26)
        s = chr(65 + r) + s
    return s


class MiniXlsxWriter:
    """最小可用的 xlsx 写出器（不依赖任何第三方库）。

    支持：多 sheet、表头加粗、列宽、冻结首行、数字与文本混排。
    不支持：公式求值、图表、条件格式（这些属于 openpyxl 的活）。
    """

    def __init__(self) -> None:
        self.sheets: List[Tuple[str, List[List[Any]], List[int]]] = []

    def add_sheet(self, name: str, rows: List[List[Any]], widths: Optional[List[int]] = None) -> None:
        # Excel 的 sheet 名不能含 : \ / ? * [ ]，且不超过 31 字符
        safe = str(name)
        for ch in ':\\/?*[]':
            safe = safe.replace(ch, "_")
        safe = safe[:31] or "Sheet"
        self.sheets.append((safe, rows, widths or []))

    # ---- 各部件 XML ----
    def _sheet_xml(self, rows: List[List[Any]], widths: List[int], header_bold: bool = True) -> str:
        parts: List[str] = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
            '<sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '</sheetView></sheetViews>',
        ]
        if widths:
            parts.append("<cols>")
            for i, w in enumerate(widths, 1):
                parts.append(f'<col min="{i}" max="{i}" width="{max(8, int(w))}" customWidth="1"/>')
            parts.append("</cols>")
        parts.append("<sheetData>")
        for r_i, row in enumerate(rows, 1):
            cells: List[str] = []
            for c_i, val in enumerate(row, 1):
                ref = f"{_col_letter(c_i)}{r_i}"
                if val is None or (isinstance(val, str) and val == ""):
                    continue
                if isinstance(val, bool):
                    val = str(val)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    cells.append(f'<c r="{ref}"><v>{val}</v></c>')
                else:
                    style = ' s="1"' if (header_bold and r_i == 1) else ""
                    cells.append(
                        f'<c r="{ref}" t="inlineStr"{style}><is><t xml:space="preserve">'
                        f"{_xml_escape(val)}</t></is></c>"
                    )
            parts.append(f'<row r="{r_i}">' + "".join(cells) + "</row>")
        parts.append("</sheetData></worksheet>")
        return "".join(parts)

    def save(self, path: str) -> None:
        n = len(self.sheets)
        if n == 0:
            raise ValueError("MiniXlsxWriter: 至少要有一个 sheet")
        ct = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '<Default Extension="xml" ContentType="application/xml"/>',
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        ]
        for i in range(1, n + 1):
            ct.append(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        ct.append(
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        )
        ct.append("</Types>")

        wb_sheets = "".join(
            f'<sheet name="{_xml_escape(nm)}" sheetId="{i}" r:id="rId{i}"/>'
            for i, (nm, _, _) in enumerate(self.sheets, 1)
        )
        workbook = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{wb_sheets}</sheets></workbook>"
        )

        rels = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        ]
        for i in range(1, n + 1):
            rels.append(
                f'<Relationship Id="rId{i}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{i}.xml"/>'
            )
        rels.append(
            f'<Relationship Id="rId{n + 1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
        )
        rels.append("</Relationships>")

        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>'
        )

        # 表头加粗斜体走 s="1"：style xf 索引 1
        styles = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="2">'
            '<font><sz val="11"/><name val="Calibri"/></font>'
            '<font><b/><sz val="11"/><name val="Calibri"/></font>'
            '</fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="2">'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
            '</cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>'
        )

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", "".join(ct))
            z.writestr("_rels/.rels", root_rels)
            z.writestr("xl/workbook.xml", workbook)
            z.writestr("xl/_rels/workbook.xml.rels", "".join(rels))
            z.writestr("xl/styles.xml", styles)
            for i, (_, rows, widths) in enumerate(self.sheets, 1):
                z.writestr(f"xl/worksheets/sheet{i}.xml", self._sheet_xml(rows, widths))


# ==========================================================================
# 二、写 xlsx：优先 openpyxl，降级 MiniXlsxWriter
# ==========================================================================
def write_xlsx(path: str, sheets: List[Tuple[str, List[List[Any]], List[int]]]) -> str:
    """写多 sheet xlsx，返回实际使用的后端名（openpyxl / stdlib-fallback）。"""
    if HAS_OPENPYXL:
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill("solid", fgColor="2F5B8F")
        for name, rows, widths in sheets:
            ws = wb.create_sheet(title=str(name)[:31])
            for r in rows:
                ws.append(r)
            for i, w in enumerate(widths, 1):
                ws.column_dimensions[get_column_letter(i)].width = max(10, int(w))
            if ws.max_row >= 1:
                for c in ws[1]:
                    c.font = header_font
                    c.fill = header_fill
                    c.alignment = Alignment(horizontal="center")
                ws.freeze_panes = "A2"
            ws.row_dimensions[1].height = 20
        wb.save(path)
        return "openpyxl"

    mx = MiniXlsxWriter()
    for name, rows, widths in sheets:
        mx.add_sheet(name, rows, widths)
    mx.save(path)
    return "stdlib-fallback"


# ==========================================================================
# 三、指标整理
# ==========================================================================
_OVERALL_CN = {
    "mAP@0.5": "mAP@0.5",
    "mAP@0.5:0.95": "mAP@0.5:0.95",
    "map50": "mAP@0.5",
    "map": "mAP@0.5:0.95",
    "precision": "精确率 Precision",
    "recall": "召回率 Recall",
    "f1": "F1 分数",
    "false_alarm_rate": "误报率（农户最关心）",
    "far": "误报率（农户最关心）",
    "disease_index_mae": "病情指数 MAE",
    "severity_acc": "严重度分级准确率",
    "diseased_ratio_mae": "发病率 MAE",
}


def _pretty_metric(k: str) -> str:
    return _OVERALL_CN.get(str(k), str(k))


def _num(v: Any) -> Any:
    """能转数字就转，转不了原样返回。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def build_sheets(experiments: List[dict], title: str) -> List[Tuple[str, List[List[Any]], List[int]]]:
    """把实验数据整理成 [(sheet 名, 行, 列宽), ...]。"""
    sheets: List[Tuple[str, List[List[Any]], List[int]]] = []
    e0 = experiments[0] if experiments else {}

    # ---- sheet 1：总体指标 ----
    rows: List[List[Any]] = [["指标", "数值", "说明"]]
    overall = e0.get("overall") or {}
    for k, v in overall.items():
        rows.append([_pretty_metric(k), _num(v), ""])
    if len(experiments) > 1:  # 多实验时并排对比
        rows = [["指标"] + [str(e.get("experiment") or e.get("model_version") or f"exp{i+1}")
                            for i, e in enumerate(experiments)]]
        keys: List[str] = []
        for e in experiments:
            for k in (e.get("overall") or {}):
                if k not in keys:
                    keys.append(k)
        for k in keys:
            rows.append([_pretty_metric(k)] + [_num((e.get("overall") or {}).get(k, "")) for e in experiments])
    if len(rows) == 1:
        rows.append(["（输入 JSON 中没有 overall 字段）", "", ""])
    sheets.append(("总体指标", rows, [30, 16, 40]))

    # ---- sheet 2：每类指标 ----
    per_class = e0.get("per_class") or []
    cols = ["disease_name", "disease_name_cn", "ap@0.5", "precision", "recall", "f1",
            "num_gt", "num_pred"]
    extra = sorted({k for pc in per_class for k in pc.keys()} - set(cols))
    cols = cols + extra
    header_cn = {"disease_name": "病害英文标识", "disease_name_cn": "病害中文名",
                 "ap@0.5": "AP@0.5", "precision": "精确率", "recall": "召回率",
                 "f1": "F1", "num_gt": "GT 框数", "num_pred": "预测框数"}
    rows = [[header_cn.get(c, c) for c in cols]]
    for pc in per_class:
        rows.append([_num(pc.get(c, "")) for c in cols])
    if not per_class:
        rows.append(["（输入 JSON 中没有 per_class 字段）"] + [""] * (len(cols) - 1))
    sheets.append(("每类指标", rows, [22, 18, 12, 12, 12, 12, 12, 12] + [14] * len(extra)))

    # ---- sheet 3：地块统计 ----
    fields = e0.get("fields") or []
    fcols = ["field_id", "crop", "area_mu", "total_boxes", "disease_index",
             "diseased_ratio", "top_disease", "severity"]
    fextra = sorted({k for f in fields for k in f.keys()} - set(fcols))
    fcols = fcols + fextra
    fheader_cn = {"field_id": "地块编号", "crop": "作物", "area_mu": "面积(亩)",
                  "total_boxes": "识别框数", "disease_index": "病情指数",
                  "diseased_ratio": "发病率", "top_disease": "主病害", "severity": "严重度"}
    rows = [[fheader_cn.get(c, c) for c in fcols]]
    for f in fields:
        rows.append([_num(f.get(c, "")) for c in fcols])
    if not fields:
        rows.append(["（输入 JSON 中没有 fields 字段）"] + [""] * (len(fcols) - 1))
    sheets.append(("地块统计", rows, [14, 12, 12, 14, 14, 14, 22, 14] + [14] * len(fextra)))

    # ---- sheet 4：延迟预算 ----
    lat = e0.get("latency") or {}
    if lat:
        rows = [["指标", "数值", "目标", "是否达标"]]
        targets = {"cpu_ms_640": 1500.0, "gpu_ms_640": 200.0, "tile_4096_s": 30.0}
        unit = {"cpu_ms_640": "ms", "gpu_ms_640": "ms", "tile_4096_s": "s"}
        for k, v in lat.items():
            tgt = targets.get(k)
            vv = _num(v)
            ok = ""
            if tgt is not None and isinstance(vv, (int, float)):
                ok = "达标" if vv <= tgt else "未达标"
            rows.append([k, vv, tgt if tgt is not None else "", ok])
        sheets.append(("延迟预算", rows, [24, 16, 14, 12]))

    # ---- sheet 5：实验信息 ----
    rows = [["字段", "内容"]]
    rows.append(["报告标题", title])
    for key, label in (("experiment", "实验名"), ("model_version", "模型版本"),
                       ("dataset", "数据集"), ("eval_date", "评估日期")):
        if e0.get(key):
            rows.append([label, str(e0[key])])
    rows.append(["实验数量", len(experiments)])
    for i, e in enumerate(experiments):
        rows.append([f"实验{i+1} 名称", str(e.get("experiment") or e.get("model_version") or "")])
    for n in (e0.get("notes") or []):
        rows.append(["备注", str(n)])
    sheets.append(("实验信息", rows, [22, 70]))

    # ---- sheet 6：多实验对比（仅在多实验时）----
    if len(experiments) > 1:
        keys: List[str] = []
        for e in experiments:
            for k in (e.get("overall") or {}):
                if k not in keys:
                    keys.append(k)
        rows = [["指标"] + [str(e.get("experiment") or f"exp{i+1}") for i, e in enumerate(experiments)]]
        for k in keys:
            rows.append([_pretty_metric(k)] + [_num((e.get("overall") or {}).get(k, "")) for e in experiments])
        rows.append([])
        rows.append(["逐类 AP@0.5 对比"] + [""] * len(experiments))
        cls_keys: List[str] = []
        for e in experiments:
            for pc in (e.get("per_class") or []):
                nm = str(pc.get("disease_name", ""))
                if nm and nm not in cls_keys:
                    cls_keys.append(nm)
        for cls in cls_keys:
            row = [cls]
            for e in experiments:
                hit = next((pc for pc in (e.get("per_class") or [])
                            if str(pc.get("disease_name")) == cls), None)
                row.append(_num(hit.get("ap@0.5", "")) if hit else "")
            rows.append(row)
        sheets.append(("多实验对比", rows, [26] + [18] * len(experiments)))
    return sheets


# ==========================================================================
# 四、Markdown 简报
# ==========================================================================
def build_markdown(experiments: List[dict], title: str, exporter: str) -> str:
    """生成中文 Markdown 简报。"""
    e0 = experiments[0] if experiments else {}
    L: List[str] = []
    L.append(f"# {title}")
    L.append("")
    meta_bits = []
    for k, lab in (("experiment", "实验名"), ("model_version", "模型版本"),
                   ("dataset", "数据集"), ("eval_date", "评估日期")):
        if e0.get(k):
            meta_bits.append(f"**{lab}**：{e0[k]}")
    if meta_bits:
        L.append("- " + "　|　".join(meta_bits))
        L.append("")

    L.append("## 一、总体指标")
    L.append("")
    overall = e0.get("overall") or {}
    if overall:
        L.append("| 指标 | 数值 |")
        L.append("| --- | ---: |")
        for k, v in overall.items():
            vv = _num(v)
            shown = f"{vv:.4f}" if isinstance(vv, float) else vv
            L.append(f"| {_pretty_metric(k)} | {shown} |")
    else:
        L.append("_（无 overall 数据）_")
    L.append("")

    if len(experiments) > 1:
        L.append("## 二、多实验对比（总体指标）")
        L.append("")
        keys: List[str] = []
        for e in experiments:
            for k in (e.get("overall") or {}):
                if k not in keys:
                    keys.append(k)
        L.append("| 指标 | " + " | ".join(str(e.get("experiment") or f"exp{i+1}")
                                          for i, e in enumerate(experiments)) + " |")
        L.append("| --- | " + " | ".join("---:" for _ in experiments) + " |")
        for k in keys:
            vals = []
            for e in experiments:
                vv = _num((e.get("overall") or {}).get(k, ""))
                vals.append(f"{vv:.4f}" if isinstance(vv, float) else str(vv))
            L.append(f"| {_pretty_metric(k)} | " + " | ".join(vals) + " |")
        L.append("")

    per_class = e0.get("per_class") or []
    L.append("## 三、每类指标")
    L.append("")
    if per_class:
        L.append("| 病害 | AP@0.5 | 精确率 | 召回率 | F1 | GT 框数 | 预测框数 |")
        L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for pc in per_class:
            def _f(key: str) -> str:
                v = _num(pc.get(key, ""))
                return f"{v:.4f}" if isinstance(v, float) else str(v)
            L.append(f"| {pc.get('disease_name_cn') or pc.get('disease_name','')} "
                     f"(`{pc.get('disease_name','')}`) | {_f('ap@0.5')} | {_f('precision')} | "
                     f"{_f('recall')} | {_f('f1')} | {pc.get('num_gt','')} | {pc.get('num_pred','')} |")
        # 找最差类别，给整改建议
        try:
            worst = min(per_class, key=lambda p: float(p.get("ap@0.5", 1.0)))
            L.append("")
            L.append(f"> ⚠️ 最弱类别：**{worst.get('disease_name_cn') or worst.get('disease_name')}**"
                     f"（AP@0.5 = {worst.get('ap@0.5')}）。建议检查该类样本量是否不足、"
                     f"是否与其他类别视觉相似，必要时补充样本或加类别权重。")
        except (TypeError, ValueError):
            pass
    else:
        L.append("_（无 per_class 数据）_")
    L.append("")

    fields = e0.get("fields") or []
    if fields:
        L.append("## 四、地块统计")
        L.append("")
        L.append("| 地块 | 作物 | 面积(亩) | 识别框数 | 病情指数 | 发病率 | 主病害 | 严重度 |")
        L.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
        for f in fields:
            dr = _num(f.get("diseased_ratio", ""))
            dr_s = f"{dr:.2%}" if isinstance(dr, float) else str(dr)
            L.append(f"| {f.get('field_id','')} | {f.get('crop','')} | {f.get('area_mu','')} | "
                     f"{f.get('total_boxes','')} | {f.get('disease_index','')} | {dr_s} | "
                     f"{f.get('top_disease','')} | {f.get('severity','')} |")
        L.append("")

    lat = e0.get("latency") or {}
    if lat:
        L.append("## 五、推理延迟预算")
        L.append("")
        L.append("| 指标 | 实测 | 目标 | 结论 |")
        L.append("| --- | ---: | ---: | --- |")
        targets = {"cpu_ms_640": (1500.0, "ms"), "gpu_ms_640": (200.0, "ms"),
                   "tile_4096_s": (30.0, "s")}
        for k, v in lat.items():
            tgt, unit = targets.get(k, (None, ""))
            vv = _num(v)
            if tgt is not None and isinstance(vv, (int, float)):
                L.append(f"| {k} | {vv} {unit} | {tgt} {unit} | {'✅ 达标' if vv <= tgt else '❌ 未达标'} |")
            else:
                L.append(f"| {k} | {vv} {unit} | - | - |")
        L.append("")

    notes = e0.get("notes") or []
    if notes:
        L.append("## 六、备注")
        L.append("")
        for n in notes:
            L.append(f"- {n}")
        L.append("")

    L.append("---")
    L.append(f"详细数据见同目录 `report.xlsx`（Excel 写出后端：{exporter}）。")
    return "\n".join(L)


# ==========================================================================
# 五、CLI
# ==========================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="report_export.py",
        description="把评估/统计 JSON 导出成多 sheet Excel + 中文 Markdown 简报。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python tools/report_export.py --input eval.json --out runs/report\n"
               "  python tools/report_export.py --input exp1.json exp2.json --title \"消融对比\"\n",
    )
    p.add_argument("--input", nargs="+", required=True, help="一个或多个评估结果 JSON 路径")
    p.add_argument("--out", default="runs/report", help="输出目录（默认 runs/report）")
    p.add_argument("--title", default="农作物病害识别算法 · 评估报告",
                   help="报告标题")
    p.add_argument("--basename", default="report", help="输出文件主名（默认 report）")
    p.add_argument("--quiet", action="store_true", help="安静模式，减少提示输出")
    args = p.parse_args(argv)

    experiments: List[dict] = []
    for path in args.input:
        if not os.path.isfile(path):
            print(f"[错误] 输入文件不存在：{path}", file=sys.stderr)
            return 2
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            experiments.extend([d for d in data if isinstance(d, dict)])
        elif isinstance(data, dict):
            if "experiments" in data and isinstance(data["experiments"], list):
                experiments.extend([d for d in data["experiments"] if isinstance(d, dict)])
            else:
                experiments.append(data)
        else:
            print(f"[警告] 跳过无法识别的 JSON 结构：{path}", file=sys.stderr)

    if not experiments:
        print("[错误] 没有解析出任何实验数据。请检查输入 JSON 是否为对象或对象列表。", file=sys.stderr)
        return 2

    if not HAS_OPENPYXL and not args.quiet:
        print("[提示] 未检测到 openpyxl，已自动降级为内置纯标准库 xlsx 写出器"
              "（生成合法 .xlsx，但没有单元格配色/加粗等样式）。"
              "如需完整样式请执行：pip install openpyxl", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    sheets = build_sheets(experiments, args.title)
    xlsx_path = os.path.join(args.out, f"{args.basename}.xlsx")
    try:
        exporter = write_xlsx(xlsx_path, sheets)
        xlsx_ok = True
    except Exception as e:
        print(f"[警告] 写 xlsx 失败（{type(e).__name__}: {e}），退回 CSV。", file=sys.stderr)
        exporter, xlsx_ok = "csv-fallback", False
        for name, rows, _ in sheets:
            safe = str(name).replace("/", "_")
            with open(os.path.join(args.out, f"{args.basename}_{safe}.csv"),
                      "w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerows(rows)

    md_path = os.path.join(args.out, f"{args.basename}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(experiments, args.title, exporter))

    print("=" * 68)
    if xlsx_ok:
        print(f"[完成] Excel（{len(sheets)} 个 sheet，后端 {exporter}）→ {xlsx_path}")
        for name, rows, _ in sheets:
            print(f"       - sheet「{name}」{len(rows)} 行")
    else:
        print(f"[完成] CSV 已写出到 {args.out}")
    print(f"       简报 → {md_path}")
    e0 = experiments[0]
    if e0.get("overall"):
        top = list(e0["overall"].items())[:3]
        print("       关键指标：" + "，".join(f"{_pretty_metric(k)}={v}" for k, v in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
