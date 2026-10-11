"""生成接口联调交接文档：真实请求/响应样例 + 启动方式 + 待确认清单。

为什么用脚本生成而不是手写：文档里的响应样例必须与**当前真实服务**一致。
手写的样例过两天就会和代码脱节（字段改名、阈值改动），联调时双方各拿一份
"看起来对"的样例，排查成本极高。这里直接读 `runs/api_examples.json`
（由真实 uvicorn 服务抓取），保证文档 = 事实。

用法：
    python scripts/make_api_handover.py --out docs/接口联调说明.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="runs/api_examples.json")
    ap.add_argument("--out", default="docs/接口联调说明.md")
    args = ap.parse_args(argv)

    ex_path = ROOT / args.examples
    if not ex_path.is_file():
        print(f"[错误] 找不到样例 {ex_path}；先跑抓取脚本（见本文件 docstring 的说明）")
        return 2
    d = json.loads(ex_path.read_text(encoding="utf-8"))

    pj = d.get("predict_json") or {}
    box0 = (pj.get("boxes") or [{}])[0]
    adv = pj.get("advice") or {}
    stats = pj.get("statistics") or {}

    def j(obj, n=1400):
        return json.dumps(obj, ensure_ascii=False, indent=2)[:n]

    # 在 f-string 之外先算好，避免 `{{...}}` 在表达式里被当成集合字面量（会报 unhashable）
    summary_keys = ("code", "message", "image_id", "disease_name",
                    "disease_name_cn", "confidence", "severity")
    summary = {k: pj.get(k) for k in summary_keys}
    label_names = ", ".join(
        x.get("disease_name", "") for x in (d.get("labels") or {}).get("classes", [])
        if isinstance(x, dict)
    ) or "healthy, stressed"

    doc = f"""# 算法侧接口联调说明

> 本文档中的请求/响应样例**由真实运行的服务抓取**（`runs/api_examples.json`），
> 不是手写示例。重新生成：`python scripts/make_api_handover.py`。
> 抓取时间与服务状态见文末。

## 一、算法侧交付边界

| 项 | 归属 |
|---|---|
| `POST /algo/predict` 及其字段语义 | **算法侧负责**（已完成并 HTTP 实测） |
| `GET /health`、`GET /algo/labels`、`GET /algo/result/{{image_id}}` | 算法侧附带提供 |
| Apifox 文档、小程序页面、用户体系 | 后端/前端负责 |
| 图片存储与 `image_id` 分发 | **需与后端确认**（见第四节问题 2） |

## 二、启动方式

```bash
cd drone-crop-disease-algo
.venv/Scripts/python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

启动后自检：

```bash
curl http://127.0.0.1:8000/health
```

当前 `/health` 返回：

```json
{j(d.get("health"), 400)}
```

## 三、两种调用方式与真实响应

### 方式 A：JSON + `image_id`（Apifox 定义的形态）

```bash
curl -X POST http://127.0.0.1:8000/algo/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"image_id": "{pj.get('image_id', 'Image_001.jpg')}"}}'
```

**HTTP {d.get('predict_json_status')}**，响应（截断）：

```json
{j(summary, 400)}
```

关键字段：

| 字段 | 类型 | 示例值 | 说明 |
|---|---|---|---|
| `code` | int | `{pj.get('code')}` | 200 表示成功 |
| `message` | string | `{pj.get('message')!r}` | 固定 `success` |
| `disease_name` | string | `{pj.get('disease_name')!r}` | **英文标识**，前端做映射用这个 |
| `disease_name_cn` | string | `{pj.get('disease_name_cn')!r}` | 中文展示名 |
| `confidence` | number | `{pj.get('confidence')}` | 主病害置信度 |
| `severity` | string | `{pj.get('severity')!r}` | `healthy` / `light` / `moderate` / `severe` |
| `boxes` | array | 共 {len(pj.get('boxes') or [])} 个 | 见下 |
| `processing_time` | number | `{pj.get('processing_time')}` | 秒 |
| `model_version` | string | `{pj.get('model_version')!r}` | 便于定位线上是哪个模型 |
| `backend` | string | `{pj.get('backend')!r}` | 实际生效的后端 |
| `within_latency_budget` | bool | `{pj.get('within_latency_budget')}` | 是否在延迟预算内 |

`boxes[]` 单个元素（**同时给了归一化与像素两套坐标**，前端按需取）：

```json
{j(box0, 800)}
```

- `x, y, w, h`：**归一化**（0~1），相对图片宽高 → 适合前端按容器比例画框
- `x1, y1, x2, y2`：**像素**坐标 → 适合后端裁剪/存档
- 兼容别名：`label` / `score` / `bbox`

`statistics`：

```json
{j(stats, 500)}
```

`advice`：

```json
{j(adv, 900)}
```

> ⚠️ `advice.chemical_plan` **恒为空数组**。这是**合规设计**：在未接入当地植保部门
> 用药目录前，算法侧不生成任何农药名称与剂量，只输出"是否需要专家复核"。
> 证书/答辩场景下请勿要求算法侧编造用药建议。

### 方式 B：multipart 上传文件

```bash
curl -X POST http://127.0.0.1:8000/algo/predict -F "file=@test.jpg"
```

**HTTP {d.get('predict_multipart_status')}**，响应结构与方式 A 完全一致。

### 错误路径

缺 `image_id` 时返回 **HTTP {d.get('missing_field_status')}**（FastAPI 校验错误），
**不会返回 5xx**：

```json
{j(d.get('missing_field_body'), 500)}
```

算法侧的降级链为 `ONNX → PyTorch → 规则基线`，任一级失败都会**降级并回报原因**，
接口始终返回可用结构而不是抛 500。

## 四、需要与后端确认的 4 个问题

### 问题 1（最重要）：`disease_name` 的取值集合约定

当前模型是**二分类**：`{label_names}`。

如果 Apifox 里 `disease_name` 的枚举是按**具体病种**（如 `rice_blast`）设计的，
则需要：
- **方案 1**：数据集扩到病种级标注（工作量大，需重新标注）
- **方案 2**：Apifox 增加一个"胁迫状态"枚举值，前端按二分类展示（**建议**）

**请确认 Apifox 中的枚举定义。**

### 问题 2：`image_id` 对应的图片存在哪里

方式 A（JSON）**从服务端 `data/uploads/` 目录读图**，所以调用前图片必须已在服务端。
需要确认：

- 是小程序先上传（走方式 B），再由后端把图片落到 `data/uploads/` 并返回 `image_id`？
- 还是无人机/后端已经存好图，算法服务直接挂载该目录？

**当前约定**：算法服务读 `data/uploads/`，支持子目录递归按文件名匹配
（例如放在 `data/uploads/stress2_test/` 下也能找到）。

### 问题 3：延迟预算与部署形态

实测延迟（ONNX CPU）：**冷路径中位 65 ms**（推理+解码+NMS），
缓存命中约 1 ms。响应里的 `within_latency_budget` 会标注是否超预算。

请确认：
- 部署在哪台机器（是否有 GPU）？
- 预算上限是多少毫秒？
- 是否要求批量接口？

### 问题 4：`severity` 的分级口径

当前按**病斑面积占整图比例**分档：`<2%` 轻度、`2%~10%` 中度、`>10%` 重度
（阈值在 `configs/labels.yaml`，可调）。

**注意**：本数据集的标注框互相重叠，框面积之和会超过整图，因此
「发病率 / 病情指数」在该数据集上不可用（服务会自动加 `area_metric_note` 说明）。
若后端需要严格口径，请确认是否改用框面积并集。

## 五、可复现的自检命令

```bash
# 契约与解码回归测试（29 项）
.venv/Scripts/python.exe tests/run_all.py

# 真实 HTTP 冒烟（起 uvicorn，测 JSON/multipart/错误路径/延迟）
.venv/Scripts/python.exe scripts/smoke_http.py --port 8077

# ONNX 与 PyTorch 两个后端对拍
.venv/Scripts/python.exe scripts/bench_onnx_vs_torch.py
```

## 六、样例来源

- 抓取方式：启动真实 uvicorn（`--host 127.0.0.1`），用 `httpx` 发真实 HTTP 请求
- 样例文件：`runs/api_examples.json`
- 模型：`models/det_rtdetrv2`（解冻微调，val mAP@0.5 = 0.3346，无泄漏子集）
- 标签体系：`configs/labels_stress2.yaml`（2 类）
"""
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"已生成：{out}")
    print(f"  含真实响应样例：predict JSON HTTP {d.get('predict_json_status')}，"
          f"{len(pj.get('boxes') or [])} 个框")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
