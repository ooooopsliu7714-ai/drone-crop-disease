# 外部数据集接入记录：无人机植物病害检测数据集（2 类）

> 这份文档记录**真实接入过程、实测数字、踩过的坑**。
> 目的：换一台机器或换一个人接手时，不用重新踩一遍。

---

## 1. 数据集实况（实测，不是宣传文案）

| 项 | 实测值 | 备注 |
|---|---|---|
| 图像总数 | **1539 张** | train 1083 / valid 304 / test 152（按原始划分） |
| 分辨率 | **全部 640×640** | 宣传说"高分辨率"，实际已缩到 640 |
| 标注框 | **20906 个** | 平均 13.6 框/图，最多 28 框 |
| 类别 | `healthy` 40.5% / `stressed` 59.5% | `nc: 2` |
| 格式 | YOLO txt（`cls cx cy w h`，归一化） | 0 空文件、0 格式错误 |
| 目标尺度 | 面积中位数 0.042 → **640 下约 131×131 px** | **<20px 的占 0%，不是小目标** |

**结论：不需要切片推理、不需要提高输入分辨率**。我之前担心的"无人机小病斑"在这份数据上不存在。

### ⚠️ 语义边界（答辩必须如实说）

`stressed` = 受胁迫，说明书写明包含**病害感染 + 营养缺失 + 水分不足 + 环境压力**。
它是**胁迫标签，不是病种标签**。因此：

- 能用它做：胁迫/健康**二分类与检测**、作为流水线**第一级初筛**；
- **不能**用它出"稻瘟病/条锈病"这种病种结论，也出不了国标病级；
- **不能**用它的框算发病率/病情指数——实测真值发病率算出 **105%**（>100%），
  因为这是**单株级、互相重叠**的标注，框面积相加重叠区域被重复计入。
  `scripts/eval.py` 现在会自动识别并打印这条警告。

15 类病害体系（`configs/labels.yaml`）仍是二期目标。

---

## 2. 一键复现流程

### 2.1 GPU 全量训练（推荐，两条命令）

```powershell
cd D:\Users\jibei\Documents\deepseek-harness\default-workspace\drone-crop-disease-algo

# ① 换 GPU 版 torch（cu128，RTX 50 系 sm_120 必需）+ 环境体检
python scripts/setup_gpu_env.py
#   只体检不安装：python scripts/setup_gpu_env.py --check-only

# ② 全量训练：分类 + 检测 + ONNX + 评估，一键到底
python scripts/train_full_gpu.py --data-root data/processed_stress2
#   8GB 显存保守档：  ... --profile safe
#   只训检测：        ... --only det
#   快速验证链路：    ... --smoke          （每阶段 30 步，约 10 分钟）
#   只重跑评估：      ... --only eval
```

两个脚本的要点：

- `setup_gpu_env.py` 会**替换 CPU 版 torch/torchvision**，并用一次真实 GPU 矩阵乘
  验证轮子里有没有编进 sm_120（这一步能抓出"装了 CUDA 版但架构不对"）；
- `train_full_gpu.py` 先做**前置校验**（CUDA / 数据集框数 / 类别数与标签体系是否匹配），
  不通过直接退出，不会白跑几小时；每阶段日志落 `runs/*.log`，中断后可重跑单个阶段。

### 2.2 手动分步执行（想自己控参数时）

```powershell
# ① 解压（tar 自带 libarchive，支持加密 zip；密码需自行获取）
cmd /c "tar -x -f `"<zip路径>`" --passphrase <密码> -C datasets\drone-plant"

# ② 切到 2 类标签体系并转成项目格式
$env:ALGO_LABELS_FILE = "configs/labels_stress2.yaml"
python -m src.data.prepare_data --raw-dir "<解压出的数据集根>" `
  --out-dir data/processed_stress2 --labels configs/labels_stress2.yaml --copy-images
# 期望：train 1077 图/14519 框，val 308 图/4251 框，test 154 图/2136 框，均衡分 0.905

# ③ 训练
python -m scripts.train_cls --config configs/train_cls.yaml --data-root data/processed_stress2 `
  --backbone facebook/dinov2-small --freeze-backbone --class-weight --epochs 3 --workers 0
python -m scripts.train_det --config configs/train_det.yaml --data-root data/processed_stress2 `
  --freeze-backbone --amp --epochs 50 --batch-size 4 --grad-accum 2 --workers 4

# ④ 生成 sidecar（推理侧要靠它解释类别）
python -m scripts.export_onnx --ckpt-dir models/cls_dinov2 --sidecar-only --task classify

# ⑤ 评估 + 起服务
python -m scripts.eval --gt data/processed_stress2/val/annotations.json --ckpt-dir models/det_rtdetrv2
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

`ALGO_LABELS_FILE` 是**唯一开关**：设了走 2 类，清空走 15 类。
训练、导出 sidecar、评估、现场推理全部跟着切换，不用改代码。
`ALGO_MODEL_DIR` 可显式指定起服务时加载哪个模型（默认检测优先）。

---

## 3. 环境要点（不设这两条一定失败）

### 3.1 HuggingFace 被墙，必须走镜像

实测：`huggingface.co` DNS 解析到 31.13.96.192 后**连接超时**，
`hf-mirror.com` 正常（1.2 秒下完 config）。

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME     = "D:\Users\jibei\Documents\deepseek-harness\default-workspace\.hf_cache"
```

第二条同样必要：默认缓存目录 `C:\Users\jibei\.cache\huggingface` 在本机
**不可写**（`PermissionError: [WinError 5]`），必须换到工作区内。

永久生效（在**你自己的终端**里执行一次，沙箱内改注册表会被拒）：

```powershell
setx HF_ENDPOINT "https://hf-mirror.com"
setx HF_HOME "D:\Users\jibei\Documents\deepseek-harness\default-workspace\.hf_cache"
```

### 3.2 DataLoader 必须单进程

`--workers 2`（默认）在本机报 `PermissionError: [WinError 5]`——
多进程 DataLoader 用了命名管道，被沙箱挡住。**统一加 `--workers 0`**。

### 3.3 默认加载哪个模型

`models/` 下现在有两个产物（检测 + 分类），加载顺序是
**检测优先**（检测是主线产物，分类是流水线第一级）：

```powershell
# 默认：自动选中检测模型 det_rtdetrv2
# 需要分类模型做初筛时，显式指定：
$env:ALGO_MODEL_DIR = "models/cls_dinov2"
```

`GET /health` 的 `backend` 字段会告诉你当前用的是哪个
（`RTDetrTorchPredictor` 或 `DinovClassifierPredictor`），
响应里的 `model_version` 是权重目录名。

### 3.4 两个实测踩过的环境坑（重要）

**坑一（本次真踩了）：C: 盘只剩 2.72 GB，装不下 cu128 torch；而安装脚本"先卸载旧 torch"，把环境搞坏了。**

cu128 版 torch 装上要 4~5 GB，下载时还要同量级临时空间。本机实测：
C: 仅剩 2.72 GB → pip 报 `OSError(28, 'No space left on device')`。
脚本**先卸载了 CPU 版 torch 才去装**，结果装不上、旧的也没了，环境直接不可用。
更糟的是当时那句提示写成"若因网络问题可换镜像"，**把磁盘问题误导成网络问题**。

`setup_gpu_env.py` 现已修复：
- **先查磁盘空间**（安装目标盘 + pip 临时目录盘），不够就**直接中止、不卸载任何东西**；
- 报错按现象分类（`No space left on device` → 磁盘；timeout → 网络；no version → 渠道）；
- 新增 `--venv`（把 torch 装到别的盘）与 `--torch-mirror`（换下载源）。

**当前可用方案（已在本机验证）**：torch 装在项目内的 `.venv`（位于 D:），
用 `--system-site-packages` 复用系统里已装好的 opencv/transformers/fastapi，
因此只花 torch 自己的空间。**装完后项目命令都要用 venv 的解释器**：

```powershell
.\.venv\Scripts\Activate.ps1        # 激活后 python 即指向 venv
# 或写全路径：
.\.venv\Scripts\python.exe -m scripts.train_full_gpu --data-root data\processed_stress2 --profile lowmem
```

想在 C: 上腾空间（需要管理员）：Windows 设置 → 系统 → 存储 → 临时文件；
本机 `C:\Windows\SoftwareDistribution\Download` 实测占 **8.99 GB**。

**坑二：官方 torch 源在国内极慢。**

实测对比（同一文件，取前 3 MB 测速）：

| 源 | 速度 | 2.7 GB 预计耗时 |
|---|---|---|
| `download.pytorch.org` | 0.14 MB/s | **> 5 小时** |
| `mirrors.aliyun.com` | 0.10 MB/s | 更慢 |
| **`mirror.sjtu.edu.cn`** | **2.63 MB/s** | **约 18 分钟（实测 18.4 分钟）** |

所以装 torch 一定加 `--torch-mirror sjtu`。
另外 **pip 下载大文件时没有可靠进度输出**（实测 4 分钟磁盘零增长、无法判断死活）；
要观察进度就用 `scripts/dl_wheel.py`（带进度、速度、ETA 与断点续传），
它能把 2.6 GB 的 wheel 先下到本地，再用 `pip install <本地whl> --no-index --no-deps` 安装。

### 3.5 GPU 上真的能训了（含实测档位）

装好 cu128 后实测结果：

| 项 | 结果 |
|---|---|
| `torch` | **2.11.0+cu128**，CUDA 12.8，cuDNN 9.19 |
| 设备 | RTX 5070 Laptop GPU，**算力 sm_120** ✅ |
| GPU 矩阵乘 | 通过（2048×2048），显存 7.3 GB 可用 |
| **训练速度对比** | **30 步：GPU 3.88 s vs CPU 80.94 s → 快约 21 倍** |

**显存档位是实测出来的，不是猜的**（本机同时开着 Edge / DeepSeek Harness / 豆包 / ChatGPT，
它们常占 1.5~2.5 GB 显存）：

| 配置 | 结果 |
|---|---|
| batch 4 @640 + AMP | ❌ `fatal: Memory allocation failure`，进程以 0xC0000005 硬崩 |
| batch 2 @640 + AMP | ❌ `CUDA error: out of memory`（干净报错） |
| **batch 1 @512 + AMP** | ✅ **跑通**：训练完成、权重保存、推理自检通过（检出 11 框） |

所以本机推荐 **`--profile lowmem`**（= batch 1 + 512 输入 + grad-accum 8），
它就是在你这台机器上验证过能跑的那一档。

> 想用更大的 batch（更快、精度更好）：**先关掉占显存的程序**，
> 用 `nvidia-smi` 看空闲显存，再用 `--profile safe`（batch 2 @640）或 `balanced`（batch 4 @640）。

另外一个实测到的坑：不开 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 时，
显存不足会让进程**硬崩（访问违例）**而不是报 OOM，排查起来很费劲。
`train_full_gpu.py` 已默认给所有子进程设上这个变量。

### 3.6 现在跑训练的两条命令

```powershell
# 进入项目目录并激活 venv（每个新窗口都要做一次）
cd "D:\Users\jibei\Documents\deepseek-harness\default-workspace\drone-crop-disease-algo"
.\.venv\Scripts\Activate.ps1

# 环境变量（每个新窗口都要设）
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME     = "D:\Users\jibei\Documents\deepseek-harness\default-workspace\.hf_cache"
$env:ALGO_LABELS_FILE = "configs/labels_stress2.yaml"

# 全量训练（本机推荐 lowmem 档）
python scripts\train_full_gpu.py --data-root data\processed_stress2 --profile lowmem
```

预计耗时：分类约 3~5 分钟（GPU），检测 60 epoch 约 30~60 分钟（GPU，batch1@512）。

**坑一：C: 盘只有 2.7 GB，装不下 cu128 torch。**

cu128 版 torch 装上要 4~5 GB，下载时还要同量级临时空间。
本机实测：C: 仅剩 2.72 GB → pip 报
`OSError(28, 'No space left on device')`。
更糟的是**早期脚本先卸载旧 torch 再安装**，失败后环境变成"没有 torch"。

现在的正确做法（脚本已内置）：

```powershell
# 把 torch 装到 D: 的 venv（复用系统里已装好的 opencv/transformers/fastapi，只花 torch 的空间）
python scripts\setup_gpu_env.py --venv .venv --torch-mirror sjtu
```

装完后**项目命令都要用 venv 的解释器**（每个新开窗口激活一次）：

```powershell
.\.venv\Scripts\Activate.ps1      # 激活后 python 就指向 venv
# 或者不改激活状态，直接写全路径：
.\.venv\Scripts\python.exe -m scripts.train_full_gpu --data-root data/processed_stress2
```

想在 C: 上腾空间（需要管理员）：Windows 设置 → 系统 → 存储 → 临时文件；
本机 `C:\Windows\SoftwareDistribution\Download` 实测占 **8.99 GB**。

**坑二：官方 torch 源在国内极慢。**

实测对比（同一文件，取前 3 MB 测速）：

| 源 | 速度 | 2.7 GB 预计耗时 |
|---|---|---|
| `download.pytorch.org` | 0.14 MB/s | **> 5 小时** |
| `mirrors.aliyun.com` | 0.10 MB/s | 更慢 |
| **`mirror.sjtu.edu.cn`** | **2.63 MB/s** | **约 18 分钟** |

所以装 torch 一定要加 `--torch-mirror sjtu`。
pip 下载大文件时没有可靠进度输出（实测 4 分钟磁盘零增长、无法判断死活），
需要观察进度就用 `scripts/dl_wheel.py`（带进度条 + 断点续传）。

---

## 4. 实测结果

### 4.1 分类（DINOv2-small 冻结 backbone + 线性头）

1 个 epoch、908 步、约 13 分钟（CPU）：

| 指标 | 值 |
|---|---|
| val_accuracy | **0.9972** |
| val_macro_f1 | **0.9971** |
| 混淆矩阵 | `[[1694, 2], [10, 2545]]` |
| 各类 F1 | healthy 0.9965 / stressed 0.9976 |

> ⚠️ **这个数字高得可疑，先别写进 PPT**。两类可能靠颜色/光照等强视觉线索就分开了
> （框尺寸分布也提示数据像是合成的或经过统一处理）。要拿去答辩，先做：
> ① 用 test 集（154 图）而不是 val 评估；② 检查是否有重复图像/数据泄漏；
> ③ 看一下错分样本长什么样（`tools/vis_labels.py` 可视化）。

> 备注：第 2 个 epoch 因沙箱可用内存不足 OOM（`DefaultCPUAllocator: not enough memory`），
> 但 `best.pt` 已在 epoch 1 保存，产物完整。

### 4.2 检测（RT-DETRv2-r18vd，80 步冒烟训练）

2 个 epoch、每 epoch 40 步（CPU 上约 83 秒/40 步）：

| 指标 | 值 |
|---|---|
| train_loss | 37.72 → 19.41 |
| val_loss | 18.63 → **15.43** |
| mAP@0.5（308 张 val） | **0.0000** |

**mAP=0 是预期结果，不是 bug**：全量 1077 张图跑 1 个 epoch 是 539 步，
这里只跑了 40 步（约 7%），模型远未收敛。这条训练的意义是**打通管线**：
数据 → 训练 → `best.pt` + sidecar → 推理 → HTTP 接口。

要拿到可用精度：装 cu128 torch 用 GPU 跑几百 epoch，或用
`--max-steps-per-epoch 0`（不限步数）全量训练。

### 4.3 端到端接口（HTTP，真实 RT-DETR 权重）

```
GET  /health  -> {"backend":"RTDetrTorchPredictor","model_loaded":true,"device":"cpu"}
POST /algo/predict  {"image_id":"Image_010_....jpg"}
  -> stressed / 受胁迫  conf=0.388  severity=severe  框=1  服务端=347ms  端到端=382ms
  -> stressed / 受胁迫  conf=0.307  severity=severe  框=1  服务端=277ms
  -> stressed / 受胁迫  conf=0.251  severity=light   框=1  服务端=322ms
  -> stressed / 受胁迫  conf=0.314  severity=moderate 框=1 服务端=219ms
```

**类别解释以模型自带的 sidecar 为准**，所以 2 类模型不会把 `class_id=1`
错解释成 15 类体系里的"稻瘟病"。

---

## 5. 本次接入修掉的 13 个真实缺陷
### 5.1 最严重的一个：训练目标编码错误，导致 60 epoch 训练全废

**症状**：GPU 上训了 60 个 epoch（2.5 小时），`val_loss` 从 22.56 降到 7.01（看似正常收敛），
但评估 **mAP@0.5 = 0.0000**，且日志里写着"预测框（过滤前/后）：0 / 0"。

**根因**：`collate_detection` 为了让 transformers 5.x 接受标注，改用了嵌套 COCO 格式，
但把 **归一化 cxcywh** 的框直接填进了 COCO 的 `bbox` 字段——而 COCO 的 `bbox` 是
**`[x, y, w, h]`、左上角基准、像素单位**。processor 于是把 `0.0867` 当像素、
再按图宽归一化一次，真值 107×132 像素的框变成 **0.2×0.2 像素**：

| | 值 |
|---|---|
| 真值（像素 xywh） | `[2.0, 0.0, 107.0, 132.0]` |
| 传进去的（被当像素的归一化值） | `[0.0867, 0.1031, 0.1672, 0.2062]` |
| processor 解码出的目标 | `[0.0003, 0.0003, 0.0003, 0.0003]` → 0.2×0.2 像素 |
| 模型最终预测 | `[0.9, 0.76, 1.08, 1.25]` → 忠实复现了错误的训练目标 |

**模型没坏，是喂给它的答案错了。**

**为什么之前没被发现**：当时的测试只验证"训练能跑起来"，没验证"喂进去的目标对不对"。

**修复**：
- `src/train/detectors.py::collate_detection` 显式做 `cxcywh(归一化) → xywh(像素)` 换算；
- 新增 `tests/test_det_targets.py`：把编码后的目标交给真实 processor、**还原成像素与真值逐框比对**，
  偏差超 2 像素即失败。修复后实测**偏差 0.00 像素**；
- 同时修掉一个掩盖病根的问题：`_clip_boxes_to_image` 原来用 `x2-x1 < 1.0` 过滤小框，
  模型输出退化框时它会**把所有框静默删光**，让"预测 0 个框"看起来像"模型没学到东西"。
  现在只丢零面积框——**静默丢数据比报错更难查**。

**验证**：修复后短训 2 个 epoch，预测框即变为 `[20.4, 0.3, 389.3, 80.8]`（369×80 像素，正常尺寸），
`AP@0.5` 从 `0.0000` 变为**非 0**，`TP` 从 0 变为 3。

### 5.2 重训后 mAP 仍然只有 0.0039 —— 评估与服务的三层"阈值陷阱"

目标编码修好后重训 60 epoch，`mAP@0.5` 只到 **0.0039**。继续排查发现**三处独立的问题叠加**，
每一处都会让结果看起来像"模型没学会"，实际模型是好的：

| # | 问题 | 表现 | 修复 |
|---|---|---|---|
| A | **预测器内部先按 0.25 过滤**，评估的 `--conf` 在那之后才生效 | 不管 `--conf` 设 0.01 还是 0.25，都只有 51 个框 | `eval.py::predict_with_ckpt` 把预测器内部阈值一并放低到 0.05 |
| B | **AP 被人为截断**：先按阈值砍预测再算 AP，而 AP 是阈值无关指标 | mAP 随阈值变化（0.05→0.2521，0.25→0.0039） | `eval.py::evaluate` 分成 `all_preds`（算 AP）与 `preds`（算 P/R/F1） |
| C | **torch 后端的框缺 `disease_name` 字段**，接口按 `d.get("disease_name","healthy")` 取值 | 模型检出 18 个"受胁迫"框，接口输出的 20 个框全是"健康" | 训练侧 `predict()` 增加 `_enrich()` 补全字段，与 ONNX 后端形状对齐 |

**修完之后的真实指标（val 308 图 / 4251 框）：**

| 指标 | 值 |
|---|---|
| **mAP@0.5** | **0.2521** |
| mAP@0.5:0.95 | 0.0882 |
| P / R / F1（conf=0.10 工作点） | 0.318 / 0.429 / **0.365** |
| healthy AP@0.5 | 0.2029 |
| stressed AP@0.5 | **0.3012** |
| 无阈值时的召回（conf=0.05） | **0.766** |

**关键教训**：`loss 下降`、`mAP 为 0`、`接口不返回框` 这三件事**都不能直接推断模型好坏**，
必须先确认"评估与服务的口径是否正确"。

### 5.3 置信度阈值必须按模型标定（写进 sidecar）

该模型分数整体偏低（每图最高分只有 0.13~0.26，而默认阈值是 0.25）。不同阈值下的表现：

| 阈值 | mAP@0.5（受截断影响） | 精度 | 召回 | F1 | 框数/图 |
|---|---|---|---|---|---|
| 0.05 | 0.2521 | 0.202 | 0.766 | 0.320 | 52 |
| **0.10** | — | **0.318** | 0.429 | **0.365** | **18.6** |
| 0.15 | — | 0.401 | 0.172 | 0.241 | 5.9 |
| 0.25 | — | 0.588 | 0.007 | 0.014 | 0.17 |

按最佳 F1 取 **conf=0.10** 并写入 `model.onnx.json` 的 `meta.conf_threshold`；
可用环境变量 `ALGO_CONF_THRESHOLD` 临时覆盖。

### 5.5 数据泄漏检查：28.4% 的 val 图在 train 里有像素级副本

**起因**：分类模型拿到 macro-F1 = 0.9998，高得可疑。文件名带 `.rf.<hash>` 后缀
（Roboflow 导出），且 `Image_001 / 004 / 005` 连号疑似同一段航拍的相邻帧。

**两阶段检查**（缺一不可）：

1. `scripts/check_leakage.py`：MD5 字节级查重 + dHash + 颜色直方图粗筛。
   结果：字节级重复 0 组；但 **dHash 汉明距离中位数为 0，90.9% 的 holdout 图
   在 train 里能找到"距离 <= 2"的图**。
2. `scripts/check_leakage_pixel.py`：对候选做**像素级验证**（允许 ±8 像素平移后比 MAE）。

> ⚠️ 只做第一步会**误判**：dHash 在 8×8 分辨率上比较，同一块田的不同帧也会撞成距离 0。
> 本机实测：粗筛 + 颜色约束得到 6.1% 的"疑似"，但**随机抽样 120 张**做像素验证后，
> 真实泄漏率是 **29.2%**；全量 462 张 holdout 的结果是 **28.4%（131/462）**，
> 这些图对的 MAE 只有 3.9~4.0（平移 = 0，差异仅相当于 JPEG 重压缩）。

**泄漏对指标的真实影响**：

| 模型 | 含泄漏（全部 val） | **无泄漏子集** | 结论 |
|---|---|---|---|
| 检测 mAP@0.5 | 0.3264 | **0.3346** | **未被高估**（反而略好） |
| 检测 mAP@0.5:0.95 | 0.1258 | **0.1274** | 同上 |
| 分类 macro-F1 | 0.9998 | **0.9997** | **未被高估**（差距 0.0003） |
| 分类 accuracy | 0.9998 | **0.9997** | 同上 |

**怎么解读**：
- **检测不受影响**：靠背下 13.8 个高度重叠的框位置很难，重复图帮不上忙。
- **分类也不受影响**：说明 0.9997 不是"刷"出来的，而是
  **"健康 vs 受胁迫"图块二分类本身区分度极高**（两类颜色/纹理差异明显）。
- **该指标不构成算法优势的证据**，报告里应如实说明任务难度，不要拿它当亮点。
- 数据本身确有 28.4% 的重复，**做严谨对比时应使用 `runs/clean_val.txt` 的干净子集**
  （`scripts/eval.py --only-images runs/clean_val.txt`）。

### 5.6 解冻 backbone 微调：mAP@0.5 从 0.2521 提升到 0.3264（+29.5%）

把上述口径问题全部修好后的基线是 **mAP@0.5 = 0.2521（冻结 backbone，60 epoch）**。
瓶颈判断：预训练权重来自 COCO 自然图像，航拍是明显的域迁移，而冻结 backbone 时
分类头只能拿到未适配的特征——这正是置信度被压缩在 0.05~0.26 窄区间的原因。

**做法**：不止是"去掉 `--freeze-backbone`"，而是**分组学习率**——
head 用 1e-4、backbone 用 1e-5（小一个数量级），否则预训练权重会被大学习率破坏。

| 配置 | 冻结（60 epoch） | **解冻微调（40 epoch）** | 变化 |
|---|---|---|---|
| **mAP@0.5** | 0.2521 | **0.3264** | **+29.5%** |
| mAP@0.5:0.95 | 0.0882 | **0.1258** | **+42.6%** |
| P / R / F1（conf=0.10） | 0.318 / 0.429 / 0.365 | **0.386 / 0.473 / 0.425** | F1 +16% |
| healthy AP@0.5 | 0.2029 | **0.2796** | +38% |
| stressed AP@0.5 | 0.3012 | **0.3732** | +24% |
| 可训练参数 | 8.88 M | 20.07 M | — |
| 每 epoch | ~135 s | 146 s | 只慢 8% |
| 显存占用 | 1942 MiB | 2199 MiB | 8 GB 卡充裕 |
| 最终 val_loss | 10.24 | **8.88** | — |

**两个坑必须先避开**：
1. `--freeze-backbone` 是 `store_true` 且 `default=None`，**不传它会回退到配置文件的
   `freeze_backbone: true`**——想去掉它来微调，日志仍显示"已冻结 backbone"。
   已补 `--no-freeze-backbone`（`store_false`）才能真正解冻。
2. 解冻时若 backbone 与 head 用同一学习率，预训练权重会被破坏，效果可能还不如冻结。

**解冻后的服务端阈值（val 集标定）**：

| conf | 精度 | 召回 | F1 | 框数/图 | 场景 |
|---|---|---|---|---|---|
| 0.05 | 0.215 | **0.808** | 0.339 | 51.9 | 早筛（宁多报不漏报） |
| **0.10** | 0.386 | 0.473 | **0.425** | **16.9** | 默认（F1 最佳） |
| 0.15 | 0.521 | 0.163 | 0.248 | 4.3 | 高精度复核 |

### 5.7 其余 10 个缺陷

| # | 缺陷 | 后果 | 修复位置 |
|---|---|---|---|
| 1 | YOLO 标注只找同级 `labels/`，漏了上一级 | **1539 张图解析出 0 个框**，训练集为空 | `src/data/prepare_data.py::load_yolo_source` |
| 2 | `--copy-images` 后 `file_name` 未重写 | 训练报"1077 张图未找到并剔除"，训练集清空 | `prepare_data.py::build_processed` |
| 3 | 源图定位依赖 `coco["_source_dir"]`（会被覆盖） | 1539 张源图只索引到 9 个，一张也复制不出来 | `prepare_data.py::_find_src_image`（改为按文件名建索引 + 由调用方传 raw 根） |
| 4 | transformers 5.x 要求标注是**嵌套 COCO 格式** | `ValueError: Invalid COCO detection annotations`，检测训练进不去 | `src/train/detectors.py::collate_detection` |
| 5 | `from_pretrained` 收到 config 对象当 repo id | `Repo id must use alphanumeric chars` | `src/train/predictors.py`（改用 `config=` 关键字 + `num_labels` 走 config） |
| 6 | 中文名/是否病害查**全局** 15 类表 | 2 类模型输出"未知"，类别解释错误 | `src/infer/predictor.py`、`src/api/service.py`（改为以模型类别表优先） |
| 7 | 推理侧 `LABELS` 是 import 时定死的单例 | 改 `ALGO_LABELS_FILE` 对推理不生效，两侧口径漂移 | `src/labels.py`（改为运行时解析 + 代理） |
| 8 | 规则基线的"症状→类别"写死 15 类病名 | 2 类体系下指向不存在的类，接口返回 `unknown` | `src/infer/heuristics.py::resolve_symptom_class` |
| 9 | 训练侧预测器用全局标签表解释产物 | 2 类模型的 class_id 被读成 15 类的病种 | `src/train/predictors.py::_build_label_book`（产物自带类别优先） |
| 10 | 模型选择不区分任务，且 `registry.py` 漏了 `import json` | 分类器顶掉检测模型；task 判定静默失效 | `src/runtime/registry.py`（检测优先 + `ALGO_MODEL_DIR` 可指定） |

另外给 `scripts/eval.py` 加了"发病率 >100% 自动诊断"，避免出报告时被当成硬伤。

---

## 6. 下一步（按优先级）

1. **装 GPU 版 torch**：`pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision`
   （RTX 5070 是 sm_120，只有 cu128 及以上带该架构支持）。
2. **全量训练检测模型**：`--max-steps-per-epoch 0 --epochs 50~100 --batch-size 4 --amp`，
   然后 `python -m scripts.eval --gt data/processed_stress2/val/annotations.json --ckpt-dir models/det_rtdetrv2` 拿真实 mAP。
3. **换 test 集评估分类器**，并排查是否有数据泄漏（0.997 的准确率需要解释）。
4. **判断这套数据在方案里的定位**：建议作为"第一级胁迫初筛 + YOLO 对照基线"，
   病种细分仍用 15 类体系另行标注。
5. **ONNX 部署**：`pip install onnx onnxruntime` 后完整导出，`/health` 会切到 `onnxruntime`。
