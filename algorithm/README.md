# 无人机农作物病害识别 —— 算法侧交付包

> 无人机航拍 → 识别农作物病害。本目录是**算法同学负责的全部内容**：
> 最新论文方法的训练流水线 + 一个严格符合 Apifox 契约的 HTTP 接口。
>
> 一句话卖点：**用 RT-DETRv3 + DINOv3 + SAM 2.1 的三级流水线替代 YOLO**，
> 端到端检测（无 NMS 后处理）、开放词汇支持新类别、像素级病斑面积量化。

---

## 1. 当前状态（先看这段，别被"跑起来了"骗了）

| 能力 | 状态 | 说明 |
|---|---|---|
| `POST /algo/predict` 接口 | ✅ 现在就能用 | 字段与 Apifox 契约完全一致，已通过 24 项离线测试 + 真实 HTTP 调用 |
| 规则基线后端（颜色/植被指数） | ✅ 可用 | **仅供前后端联调与对照实验**，不会识别具体病种，不要拿它当算法成果 |
| PyTorch 推理后端（训练产物） | ✅ 已跑通 | 自动发现 `models/<name>/best.pt` + sidecar；已用真实训练产物端到端验证过（该自测产物已清理，仓库内 `models/` 目前为空） |
| DINOv2/v3 分类训练脚本 | ✅ 已跑通（小样例） | `scripts/train_cls.py` 产出 `best.pt` + `model.onnx.json`，推理侧已能消费 |
| RT-DETRv2/v3 检测训练脚本 | ⚠️ 代码就绪、需装依赖 | 需 GPU 版 torch；`--backend transformers-rtdetrv2 / ultralytics-rtdetr / paddle` |
| 开放词汇 + 自动标注 | ⚠️ 代码就绪、需装依赖 | `scripts/train_openvocab.py`、`scripts/autolabel.py`（需 ultralytics / Grounding DINO） |
| SAM 2.1 病斑分割量化 | ⚠️ 方案就绪 | 见 `docs/algorithm-design.md` 第三级流水线 |
| ONNX 部署 | ⚠️ 需装 `onnx` / `onnxruntime` | 导出脚本已就绪；当前机器未装这两个包，所以只写出 sidecar |

**当前机器实测**：Python 3.10.11、torch 2.10.0+**cpu**、torchvision 0.25.0+**cpu**、RTX 5070 Laptop（8GB，sm_120）但 torch 是 CPU 版。
要真训练，先按 [REQUIREMENTS.md](REQUIREMENTS.md) 换 cu128 版 torch（**torch 与 torchvision 必须一起换**）。

自检一条命令：

```powershell
python scripts/verify_all.py
```

### 一个先用数据、再换模型的判断（实测证据）

用 `tools/vis_labels.py` 检查标注尺度，是本项目**性价比最高的一步**：

```powershell
python tools/vis_labels.py --coco data/samples/annotations.json --images data/samples/images --out runs/label_qc
```

实测效果：构造一张 2048×2048、含 24 个 8~20px 小病斑的图，工具 24/24 全部命中，
结论是"640 输入下短边 <16px 占比 100%"——**这类样本即使换成 RT-DETR 也学不到**。
这正是本项目必须做切片推理（`tools/tile_infer.py`）和高分辨率/切片训练的原因，
也是答辩时"我们先解决尺度问题、再换模型"这条叙事的实证支撑。

---

## 2. 架构：三级流水线（为什么不是"又一个 YOLO"）

```
┌─────────────────────────────────────────────────────────────────────────┐
│  无人机正射影像 / 手机拍摄图  (4000×4000 甚至更大)                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  滑窗切片 640×640, overlap 128
                                │  (大图直接缩放会把小病斑缩没)
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  第一级  地块级初筛：DINOv3 / DINOv2 自监督特征 + 线性探测                │
│  作用：整块地/整张图"有没有病、大概哪类病"。                              │
│  为什么：航拍图里病斑像素占比常 <1%，直接检测召回率低；先做图级判別能把   │
│          无病地块整片滤掉（省算力），也是小样本冷启动的最优解。           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  有病 → 进入定位
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  第二级  病斑级定位：RT-DETRv2 / RT-DETRv3 (WACV 2025)                    │
│  作用：端到端集合预测，直接输出病斑框，**无 anchor、无 NMS 后处理**。      │
│  为什么替代 YOLO：① 无 NMS 超参与手写后处理；② 分层密集正监督改善小目标   │
│          召回；③ 输出与 DINOv3 特征同源（都在 transformer 体系内）。      │
│  工程细节：切片推理后做 class-wise NMS 合并（见 src/infer/postprocess.py）│
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  候选框
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  第三级  像素级量化：SAM 2.1 分割病斑 → 真实病斑面积                     │
│  作用：把"框"变成"面积"，才能算发病率、病情指数（对应核心功能 4 统计报表）│
│  为什么必须要：框的面积包含大量健康组织，直接用框面积会高估病情 2~5 倍。  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
        发病率 / 病情指数 / 严重度分级 / 病害分布热力图 / 精准施药处方图
```

**开放词汇支线（加分项）**：YOLO-World / Grounding DINO 支持用文本提示直接检测新病种，
配合 `scripts/autolabel.py` 给无标注航拍图生成伪标签 → 农技员在小程序复核修正 →
回流训练集。这就是项目要求的"AI 初筛 + 人工复核"闭环。

详细选型对比、训练策略、评估方案、风险与参考文献见 **[docs/algorithm-design.md](docs/algorithm-design.md)**。

---

## 3. 目录结构

```
drone-crop-disease-algo/
├─ configs/
│  ├─ labels.yaml              ★ 标签体系 + 接口契约（15 类病害，唯一真源）
│  ├─ labels_stress2.yaml      外部数据集的 2 类体系（healthy/stressed）
│  ├─ train_det.yaml           检测训练配置
│  ├─ train_cls.yaml           分类训练配置
│  └─ train_openvocab.yaml     开放词汇配置
├─ src/
│  ├─ labels.py                标签体系加载（严重度换算在这里）
│  ├─ api/
│  │  ├─ schema.py             ★ 契约模型（字段名不许改）
│  │  ├─ service.py            业务编排：模型输出 → 契约响应
│  │  ├─ advice.py             防治建议 + 复核队列（不含具体农药，合规）
│  │  └─ app.py                FastAPI 应用（一个地址支持 JSON / 上传两种方式）
│  ├─ infer/
│  │  ├─ predictor.py          统一预测入口 + 大图切片
│  │  ├─ onnx_backend.py       ONNX 推理（生产部署用）
│  │  ├─ heuristics.py         规则基线（联调用 / 对照实验）
│  │  └─ postprocess.py        class-wise NMS、切片合并
│  ├─ runtime/registry.py      后端选择与降级（模型坏了也不返回 5xx）
│  ├─ data/                    数据转换与统计
│  └─ train/                   训练核心 + 统一推理入口
├─ scripts/
│  ├─ setup_gpu_env.py         ★ 一键装 cu128 版 torch + 环境体检（RTX 50 系 sm_120 必需）
│  ├─ train_full_gpu.py        ★ 一键全量训练：分类 + 检测 + ONNX + 评估
│  ├─ train_det.py             RT-DETR 检测训练
│  ├─ train_cls.py             DINOv3 分类 / 小样本
│  ├─ train_openvocab.py       YOLO-World / Grounding DINO
│  ├─ autolabel.py             伪标签自动标注
│  ├─ export_onnx.py           导出 ONNX + sidecar 元数据
│  ├─ eval.py                  检测/分级评估（纯 numpy，无需 pycocotools）
│  └─ verify_all.py            ★ 一键环境自检
├─ tools/                      切片推理、热力图、报表导出、压测（见 tools/README）
├─ tests/
│  ├─ test_contract.py         ★ 契约测试（改字段会红）
│  ├─ test_onnx_decode.py      ★ ONNX 坐标还原测试（不需装 onnxruntime）
│  └─ run_all.py               一键跑全部离线测试
├─ docs/
│  ├─ algorithm-design.md      ★ 答辩用技术方案
│  ├─ dataset-drone-plant-stress2.md ★ 外部数据集接入记录（实测数字 + 踩坑清单）
│  └─ quickstart.md            启动参数与算法侧约定
├─ REQUIREMENTS.md             依赖安装与显存规划
└─ pyproject.toml              依赖声明（可选 pip install -e .）
```

---

## 4. 三步跑起来

### 4.1 起服务（现在就能跑，CPU 足够）

```powershell
cd drone-crop-disease-algo
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 4.2 用真实 HTTP 验证（这就是队友要的结果）

```powershell
# 方式一：上传图片直接识别
curl.exe -X POST http://127.0.0.1:8000/algo/predict -F "file=@test_field.jpg"

# 方式二：先用 image_id（队友 Java 侧存图后传 id）
curl.exe -X POST http://127.0.0.1:8000/algo/predict -H "Content-Type: application/json" -d "{\"image_id\":\"abc123.jpg\"}"

# 健康检查 / 标签字典
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/algo/labels
```

交互式文档：<http://127.0.0.1:8000/docs>（可直接在浏览器里上传图片试接口，Apifox 联调前先在这里过一遍）

### 4.3 跑测试

```powershell
python tests/run_all.py           # 全部离线测试（24 项，不需要 pytest）
python tests/test_contract.py     # 只跑接口契约测试
python scripts/verify_all.py      # 环境 + 端到端自检
```

测试覆盖的关键点：契约字段名与坐标范围、健康图不误报、大图切片合并不越界、
病情指数计算、NMS 去重、ONNX 的 letterbox 坐标还原、训练侧 sidecar 与推理侧契约一致、
predictor 工厂按 source 正确分派、模型目录发现不会挑到临时半成品。

---

## 5. 接口契约（与 Apifox 逐字一致）

`POST /algo/predict`，两种请求方式共用一个地址：

| 方式 | 请求 |
|---|---|
| JSON | `{"image_id": "abc123.jpg", "crop": "wheat"}`（crop 可选，能提升类别先验） |
| 文件 | `multipart/form-data`，字段名 `file`（也兼容 `image` / `img`） |

响应（**主字段一个都不能改名**）：

```json
{
  "code": 200,
  "message": "success",
  "image_id": "abc123.jpg",
  "disease_name": "wheat_stripe_rust",
  "disease_name_cn": "小麦条锈病",
  "confidence": 0.87,
  "severity": "moderate",
  "boxes": [
    {
      "disease_name": "wheat_stripe_rust",
      "disease_name_cn": "小麦条锈病",
      "confidence": 0.87,
      "severity": "moderate",
      "class_id": 5,
      "x": 0.25, "y": 0.25, "w": 0.25, "h": 0.25,
      "x1": 240.0, "y1": 180.0, "x2": 480.0, "y2": 360.0,
      "label": "wheat_stripe_rust", "score": 0.87, "bbox": [240.0, 180.0, 480.0, 360.0]
    }
  ],
  "processing_time": 0.0442,
  "statistics": {
    "total_boxes": 1,
    "diseased_ratio": 0.0625,
    "disease_index": 6.25,
    "severity_ratio": {"moderate": 0.0625},
    "class_ratio": {"wheat_stripe_rust": 0.0625}
  },
  "model_version": "heuristic-v1",
  "backend": "heuristic-cv",
  "advice": {
    "priority": "within_3_days",
    "priority_text": "建议 3 天内防治，优先处理发病中心",
    "practices": ["及时清沟排水，降低田间湿度", "..."],
    "variable_rate_hint": "可按本结果生成处方图……",
    "chemical_plan": [],
    "chemical_plan_source": "待接入当地植保站推荐目录",
    "need_expert_review": false
  },
  "within_latency_budget": true
}
```

字段说明与约定：

- `disease_name` 取值见 `configs/labels.yaml` 的 `classes[].disease_name`；`disease_name_cn` 只用于展示。
- `severity` 只能取 `healthy | light | moderate | severe`。
- `boxes[].x/y/w/h` 是**归一化**坐标（小程序直接按百分比画框）；`x1/y1/x2/y2` 是像素坐标（后端画热力图、做二次分割）。
- `x/y/w/h` 与 `x1…y2` 都相对**原图**，不是切片坐标——大图切片在服务端已合并回原图。
- `label` / `score` / `bbox` 是给早期用了这些字段名的前端留的兼容别名，值分别等于 `disease_name` / `confidence` / `[x1,y1,x2,y2]`。
- `advice.chemical_plan` 恒为空数组：**具体农药必须由当地植保站推荐目录提供**，算法侧不生成用药方案。
- `need_expert_review=true` 表示置信度低于 0.6，应进农技员复核队列。

其它接口：`GET /health`、`POST /algo/upload`（拿 image_id）、`GET /algo/result/{image_id}`、`GET /algo/labels`。

**换数据集 / 换类别体系不用改代码**，设一个环境变量即可（训练、导出、评估、推理同步切换）：

```powershell
$env:ALGO_LABELS_FILE = "configs/labels_stress2.yaml"   # 切到 2 类胁迫体系
$env:ALGO_LABELS_FILE = ""                              # 清空即回到 15 类病害体系
```

推理时**类别解释以模型自带的 sidecar 为准**，所以 2 类模型不会把 `class_id=1`
错解释成 15 类体系里的"稻瘟病"。

---

## 6. 外部数据集接入（已完成，含实测数字）

`datasets/` 下已接入「无人机植物病害检测数据集」：**1539 张 640×640、20906 个框、2 类**
（`healthy`/`stressed`）。实测结论、训练数字、以及接入过程中修掉的 6 个真实缺陷，
全部记录在 **[docs/dataset-drone-plant-stress2.md](docs/dataset-drone-plant-stress2.md)**。

三条必须知道的（否则一定卡住）：

1. **HuggingFace 被墙**，必须走镜像：`$env:HF_ENDPOINT = "https://hf-mirror.com"`；
   同时 `$env:HF_HOME` 必须指向工作区内（默认 `C:\Users\jibei\.cache` 本机不可写）。
2. **DataLoader 多进程在受限环境会被拒**：`--workers 0`（`train_full_gpu.py` 已内置自动降级）。
3. 该数据集的 `stressed` 是**胁迫标签而非病种标签**，且框是**单株级互相重叠**，
   **不能**用来算发病率/病情指数（实测会算出 105%）。

### GPU 全量训练（两条命令）

```powershell
# ① 换 GPU 版 torch（先自动查磁盘空间；不够会中止且不动现有环境）
python scripts\setup_gpu_env.py --venv .venv --torch-mirror sjtu

# ② 全量训练：分类 + 检测 + ONNX + 评估
.\.venv\Scripts\Activate.ps1
python scripts\train_full_gpu.py --data-root data\processed_stress2 --profile lowmem
```

`setup_gpu_env.py` 会用一次真实 GPU 矩阵乘验证轮子里有没有编进 sm_120；
`train_full_gpu.py` 先做前置校验（CUDA / 框数 / 类别数与标签体系匹配），
不通过直接退出，不会白跑几小时。常用开关：`--profile lowmem|safe|balanced`、
`--only det|cls|eval|export`、`--smoke`（每阶段 30 步）、`--imgsz 512`。

> **本机实测**：torch 装在 `.venv`（D: 盘），因为 **C: 盘只剩 1.4 GB 装不下**。
> 官方 torch 源 0.14 MB/s（>5 小时），**SJTU 镜像 2.63 MB/s（18 分钟）**，所以必须
> `--torch-mirror sjtu`。**装完项目命令都要用 `.venv` 的解释器**（见 docs 第 3.4 节）。
> 显存：其它程序常占 1.5~2.5 GB，实测只有 **batch 1 + 512 输入** 稳（即 `--profile lowmem`）；
> GPU 训练比 CPU **快约 21 倍**。

---

## 7. 换成真实模型（训练完成后）

训练侧产出 ONNX + 元数据后，服务自动识别，**不用改一行接口代码**：

```
models/<name>/model.onnx
models/<name>/model.onnx.json      # sidecar：类别表、输入尺寸、归一化参数、输出解码方式
```

```powershell
python scripts/train_det.py --config configs/train_det.yaml     # 1) 训练 RT-DETR
python scripts/export_onnx.py --ckpt models/rtdetrv3/best.pt    # 2) 导出 ONNX + sidecar
python -m uvicorn src.api.app:app --port 8000                   # 3) 重启即自动加载
```

后端选择与降级顺序（`src/runtime/registry.py`）：

```
ONNX (models/**/model.onnx，生产) → PyTorch (models/**/best.pt 或 checkpoint.json，本地调试) → 规则基线 (永不失败)
```

两种模型形态都会**自动发现最近修改的那个模型目录**，不需要改代码也不用配路径：

```powershell
# 训练侧已有自测产物可用来验证这条链路
python -c "import sys;sys.path.insert(0,'.');from src.runtime.registry import build_predictor;print(build_predictor()[1])"
# -> BackendStatus(backend='DinovClassifierPredictor', loaded=True, ...)
```

可用环境变量强制指定：`ALGO_BACKEND=onnx|torch|heuristic|auto`。
**任何模型加载失败都不会让接口返回 5xx**，只降级并在 `/health` 与启动日志里说明原因——答辩现场不会白屏。

---

## 7. 常见问题

**Q：接口返回 `backend: heuristic-cv`，是坏了吗？**
不是。说明 `models/` 下还没有模型，当前用规则基线，前端可以正常联调。训练完成后会自动切到真实模型。
规则基线的**已知局限**（答辩时如实说明，别当成检测模型卖）：它每个切片只报一个"主症状"，
所以当一张图里同时有黄化与坏死病斑、且病斑尺寸与切片尺寸同量级时，
换切片参数可能导致主症状变化；真实检测模型逐病斑输出，不存在这个问题。
另外它只区分"黄化型 / 褐变型"两大症状族，不区分具体病种。

**Q：能不能直接上 YOLO 交差？**
技术上可以（`scripts/train_det.py --backend ultralytics-rtdetr` 给了退路），但项目要的是"用最新论文方法替代 YOLO"，
建议至少做到：RT-DETR 检测 + DINOv3 分类 + 与 YOLOv8 baseline 的对照实验。对照数据比模型本身更能拿分。

**Q：8GB 显存能训得动吗？**
能。RT-DETR-l 用 640 输入 + batch 4 + AMP + 冻结 backbone；分类用 DINOv2-small 冻结 backbone + 线性探测。
详细分配见 [REQUIREMENTS.md](REQUIREMENTS.md)。

**Q：中文路径读图报错？**
已处理：`src/infer/predictor.py` 用 `np.fromfile + cv2.imdecode` 而不是 `cv2.imread`。

**Q：Windows 文件权限/沙箱问题？**
本机工作区曾因权限导致命令无法执行，已修复（补上当前用户对工作区的完全控制项）。
若再次出现 `grantWrite(...) failed`，见会话记录里的恢复命令。
