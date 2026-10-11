# 启动参数与算法侧约定（v1.0）

这份文档给三种人看：**跑服务的运维同学**、**队友（Java / 小程序）**、**接手的算法同学**。

---

## 一、启动

### 1.1 服务启动命令

```powershell
# 开发（自动重载，改代码即生效）
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

# 生产（局域网给队友联调，多进程）
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 2
```

**为什么默认可以用 2 个 worker**：推理后端是无状态的纯前向计算，进程间不共享状态；
结果缓存（`/algo/result/{image_id}`）是进程内的，多 worker 下可能查不到另一次请求的缓存，
需要强一致就设 `--workers 1`。

### 1.2 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `ALGO_BACKEND` | `auto` | 强制推理后端：`onnx` / `torch` / `heuristic` / `auto` |
| `ALGO_MAX_UPLOAD_MB` | `200` | 单张图上传上限（正射图很大，别设太小） |
| `ALGO_SLICE_THRESHOLD` | `1600` | 长边超过它才做滑窗切片；设 `0` 关闭切片 |
| `ALGO_TILE` | `640` | 切片边长 |
| `ALGO_OVERLAP` | `128` | 切片重叠像素（必须小于 `ALGO_TILE`） |
| `PORT` | `8000` | `python -m src.api.app` 方式启动时用的端口 |

切片参数的取值依据：无人机正射图常见 4000×4000 以上，直接缩到 640 会把小病斑缩成 2~6px
（`tools/vis_labels.py` 可量化这一点），所以服务端默认对长边 >1600 的图自动切片，
推理后再做 class-wise NMS 合并回原图坐标，合并时按"置信度 × 框面积"选主病害。

### 1.3 后端自动降级顺序

```
1. ONNX      models/**/*.onnx（取最近修改的）  ← 生产用，需 pip install onnxruntime
2. PyTorch   src/train/predictors.py::build_predictor  ← 本地调试用
3. 规则基线   src/infer/heuristics.py                   ← 永不失败，保证接口可用
```

启动日志会打印实际后端；`GET /health` 的 `backend` / `model_loaded` 字段也能看到。
**只有第 3 种情况下 `model_loaded=false`**，代表还没接真实模型，此时 `disease_name` 只能区分
"黄化型 / 褐变型"两大症状族，不能当病种结论用。

---

## 二、算法侧对外约定（队友必须知道的 5 件事）

### 2.1 一个地址两种请求方式

`POST /algo/predict` 既吃 JSON 也吃 multipart，服务端按 `Content-Type` 自动分流：

```powershell
# JSON：适合队友后端已经把图存好了，只传 image_id
curl.exe -X POST http://127.0.0.1:8000/algo/predict `
  -H "Content-Type: application/json" `
  -d "{\"image_id\":\"abc123.jpg\",\"crop\":\"wheat\"}"

# multipart：适合小程序直接上传
curl.exe -X POST http://127.0.0.1:8000/algo/predict -F "file=@C:\path\to\field.jpg"
```

- `image_id` 会先在 `data/uploads/` 里按文件名查（自动尝试补 `.jpg/.png/.bmp/.webp` 扩展名）；
- 同一个 `image_id` 第二次请求会直接返回缓存结果，`message` 里带 `(cached)`；
- 找不到图返回 **404**，请求体不合法返回 **400**，缺字段返回 **422**，服务端异常返回 **500**。

### 2.2 坐标是相对原图的

大图（长边 > 1600px）在服务端走滑窗切片（640 切片 + 128 重叠）+ class-wise NMS 合并，
返回的 `x1/y1/x2/y2` 已经是**原图像素坐标**，`x/y/w/h` 是**原图归一化坐标**。
队友不需要关心切片逻辑，直接按原图尺寸画框即可。

### 2.3 主病害怎么选

- 有检测框：取**置信度最高**的框作为 `disease_name` / `confidence`；
- 整图 `severity` 取所有框里**最严重**的（保守原则：宁可高估不漏报）；
- 没有框：`disease_name="healthy"`、`severity="healthy"`、`confidence` 为模型对"健康"的把握。

### 2.4 复核闭环怎么接

`advice.need_expert_review = true` 表示置信度 < 0.6，应该：
1. 小程序把这条结果标成"待复核"，推给农技员端；
2. 农技员修正后，把修正结果按 `data/pseudo/` 的格式落盘；
3. 算法侧用 `scripts/autolabel.py` + 半监督流程回流训练集。

这是"AI 初筛 + 人工复核"的落地方式，也是评审看重的"人机协同"证据。

### 2.5 合规红线

`advice.chemical_plan` **永远是空数组**。具体农药品种与剂量必须来自当地植保站推荐目录
（项目原文要求"不要乱推荐"）。算法侧只给防治优先级和农事操作建议。
等植保站给出 `configs/drug_catalog.json` 后，在 `src/api/advice.py` 里按病害+生育期查表填充。

---

## 三、常见参数含义（训练侧）

| 参数 | 含义与建议值 |
|---|---|
| `--imgsz` | 训练/推理输入边长。小病斑多就上 800 或 1024，但 8GB 显存下 640→800 显存约涨 1.5 倍 |
| `--batch-size` | 8GB 显存建议 4（RT-DETR-l @640 + AMP）；OOM 就降到 2 并提高 `--grad-accum` |
| `--freeze-backbone` | 数据 < 2000 张时打开，省显存、防过拟合 |
| `--slice / --overlap` | 推理切片参数（`tools/tile_infer.py`），默认 640/128 |
| `--score-threshold` | 检测置信度阈值，默认 0.25；误报多就提到 0.4（但农业场景宁松勿严，漏报代价更大） |
| `--iou-threshold` | NMS 阈值，默认 0.45；病斑密集时可提到 0.6 避免相邻病斑被合并 |

---

## 四、性能预算（答辩时的量化目标）

| 场景 | 目标 | 现状（规则基线，仅供参考） |
|---|---|---|
| 单张 640×640 CPU | ≤ 1.5 s | ~0.03 s |
| 单张 640×640 GPU | ≤ 200 ms | 待测（需 GPU 版 torch） |
| 4096×4096 正射图（切片） | ≤ 30 s | ~1.5 s（规则基线，深度学习会慢 10~50 倍） |
| 接口 QPS（2 worker，CPU） | ≥ 5 | 用 `tools/bench_latency.py` 实测 |

压测命令（服务起来之后跑）：

```powershell
python tools/bench_latency.py --url http://127.0.0.1:8000/algo/predict --image test_field.jpg --n 50
```

输出 JSON + 中文报告，直接可以贴进答辩 PPT 的"性能证据"页。
