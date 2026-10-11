# REQUIREMENTS.md — 依赖说明与环境搭建

> 适用项目：无人机航拍农作物病害识别算法（`drone-crop-disease-algo`）
> 本文档按「**装什么 / 为什么 / 怎么装 / 占多少**」分组，分三档安装强度，最后附中文故障排查。
> 所有版本号均为 **2026-10-09 在本机实测** 的结果，不是抄文档。
>
> ⚠️ **本文档不改动 `src/`、`scripts/`、`configs/`、`tests/`、`README.md`、`data/`、`models/`**，
> 只描述依赖。`configs/labels.yaml` 仍是标签与接口契约的唯一真源。

---

## 0. 本机实测基线（唯一权威快照）

- 解释器：`C:\Users\jibei\AppData\Local\Programs\Python\Python310\python.exe` → **Python 3.10.11**（MSC v.1929, 64 bit）
- GPU：**NVIDIA GeForce RTX 5070 Laptop GPU**，显存 **8151 MiB**，算力 **compute_cap = 12.0（sm_120，Blackwell）**
- 驱动：**596.13**，驱动侧 CUDA 版本 **13.2**，`nvidia-smi` 可用
- 当前 torch 是 **CPU 版**（`2.10.0+cpu`）：**GPU 现在完全没被用上**

### 0.1 已装（实测 `import` 成功）

| 包 | 实测版本 | 在本项目里的角色 |
| --- | --- | --- |
| `torch` | **2.10.0+cpu** ⚠️ CPU 版 | 训练/推理框架，**未启用 GPU** |
| `torchvision` | **0.25.0+cpu** | 与 torch 配套的视觉库（预训练权重、NMS、变换） |
| `fastapi` | 0.136.3 | 提供 `POST /algo/predict` 的 Web 框架 |
| `uvicorn` | 0.49.0 | 跑 FastAPI 的 ASGI 服务器 |
| `cv2`（opencv-python） | 4.13.0 | 读图/缩放/连通域，`tools/` 依赖它 |
| `transformers` | 5.6.2 | 加载 DINOv2/DINOv3、RT-DETR 等 HF 权重 |
| `numpy` | 2.2.6 | 全部数值计算，`tools/` 的核心依赖 |
| `pillow`（PIL） | 12.1.1 | 读写图片、画框、画热力图、中文字体渲染 |
| `scipy` | 1.15.3 | 科学计算（匈牙利匹配、滤波等可选用） |
| `matplotlib` | 3.10.8 | 画曲线/混淆矩阵（`tools/make_heatmap.py` 不依赖它，自带 colormap） |
| `pyyaml` | 6.0.3 | 解析 `configs/labels.yaml`（`tools/` 优先用它，缺失有兜底） |
| `psutil` | 7.2.2 | 压测时采集内存/CPU（`tools/bench_latency.py`） |
| `requests` | 2.32.5 | 压测 HTTP 客户端（`tools/bench_latency.py`，缺失降级 urllib） |

### 0.2 未装（实测 `ModuleNotFoundError`）

| 包 | 状态 | 影响 | 是否影响提交 |
| --- | --- | --- | --- |
| `timm` | ❌ 未装 | 拿不到 ViT/DINOv2 的 timm 版权重 | ⚠️ 训练阶段需要 |
| `albumentations` | ❌ 未装 | 没有现成的强增强流水线 | ⚠️ 训练阶段推荐 |
| `ultralytics` | ❌ 未装 | 跑不了 YOLO 系列（**本项目恰恰要替代 YOLO，可长期不装**） | ✅ 可选，仅对照实验用 |
| `onnxruntime` | ❌ 未装 | 不能导出/加载 ONNX 做端侧部署 | ⚠️ 部署阶段需要 |
| `pycocotools` | ❌ 未装 | 不能用官方 COCO 评测器算 mAP | ⚠️ 评估阶段需要（**严重：没有它 mAP 口径不权威**） |
| `pandas` | ❌ 未装 | 统计分析不便 | ✅ 可跳过，`tools/` 用 csv 替代 |
| **`openpyxl`** | ❌ **未装** | 写 `.xlsx` 报表 | ✅ **已处理**，见下方「与任务书不一致的两处」 |

### 0.3 ⚠️ 与任务书描述不一致的两处（请务必确认）

任务书说「openpyxl 已装」「torchvision 未装」，但**本机实测结果相反**：

1. **`openpyxl` 实际未安装**（`ModuleNotFoundError: No module named 'openpyxl'`，已用子进程二次确认）。
   → 已处理：`tools/report_export.py` 内置了一个**纯标准库 xlsx 写出器**（`zipfile` + XML 手写 OOXML），
   `try import openpyxl` 失败时自动降级，**依然产出合法的 `.xlsx`**（5～6 个 sheet，Excel/WPS 可正常打开），
   并打印中文提示。已用官方 OOXML 校验器验证 `verdict = pass`。
   想要带配色/加粗表头的样式，再装 `openpyxl` 即可，代码无需改动。
2. **`torchvision` 实际已安装**（`0.25.0+cpu`，与 torch 2.10.0+cpu 匹配）。
   → 这意味着**换成 cu128 版时必须 torch 和 torchvision 一起换**，只换 torch 会导致
   `torchvision` 与 `torch` 版本不匹配而报错。命令见 §2.2。

---

## 1. 第一档：只跑服务（CPU 就够）

**用途**：把 `POST /algo/predict` 跑起来、联调、答辩演示。不做训练。

### 装什么

```bash
pip install fastapi uvicorn "numpy>=2.0" pillow opencv-python pyyaml psutil requests
```

### 为什么

| 包 | 为什么必须 |
| --- | --- |
| `fastapi` + `uvicorn` | 契约接口 `POST /algo/predict` 的载体，队友 Apifox 联调靠它 |
| `numpy` | 所有张量/数组运算的地基 |
| `pillow` | 读图、画框、中文字体渲染（`tools/` 的可视化全靠它） |
| `opencv-python` | 快速读图/缩放/连通域；`tile_infer.py` 的切片与假 predictor 用它 |
| `pyyaml` | 读 `configs/labels.yaml`（唯一真源）；缺失时工具会退回内置标签表并告警 |
| `psutil` + `requests` | `bench_latency.py` 采资源/发请求；两者缺失都会自动降级，不会崩 |

### 占多少（下载体积量级估计，非精确值）

| 包 | 量级 |
| --- | --- |
| `numpy` | 约 15 MB |
| `opencv-python` | 约 40 MB |
| `pillow` | 约 10 MB |
| `fastapi` + `uvicorn` + `pydantic` 等 | 约 15 MB |
| 合计 | **约 80–100 MB**（不含已装的 torch） |

> 说明：体积为经验量级估计，若要精确值可用
> `pip download --no-deps --dest tmp <包名>` 后看文件大小。

### 验证

```bash
python -c "import fastapi, uvicorn, cv2, numpy, PIL, yaml; print('CPU 服务档依赖 OK')"
```

---

## 2. 第二档：本地训练（GPU，本机必做）

**用途**：在这台 RTX 5070 Laptop 上真正训练模型。

### 2.1 现状问题（**必须先解决**）

当前 `torch 2.10.0+cpu` **不带任何 CUDA 支持**，`torch.cuda.is_available()` 会是 `False`，
RTX 5070（sm_120）完全闲置。你现在的训练要么跑不起来，要么在用 CPU 慢几十倍。

**关键事实**：RTX 50 系是 **Blackwell 架构，算力 sm_120**。
**只有 CUDA 12.8（cu128）及以上的 PyTorch 轮子才编译了 sm_120 支持**；
cu121 / cu124 的轮子在 50 系卡上会直接报
`no kernel image is available for execution on the device` 或干脆 `is_available() == False`。

### 2.2 怎么装（**一行命令，torch 与 torchvision 必须同时装**）

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

如需固定版本（避免自动升级带来意外）：

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.10.0" "torchvision==0.25.0"
```

> **注意**：`--index-url` 会**替换**默认 PyPI 源，所以这条命令只能装 PyTorch 官方源里有的包。
> 装完 torch 后，其余包请用**不带** `--index-url` 的普通 `pip install` 装。

### 2.3 训练档追加依赖

```bash
pip install timm albumentations pycocotools "transformers>=5.0" einops
```

| 包 | 为什么 | 占多少（量级） |
| --- | --- | --- |
| `torch` + `torchvision`（cu128） | GPU 训练的地基；cu128 才支持 sm_120 | **约 2.5–3.5 GB**（含 CUDA 运行时，最大的一笔） |
| `timm` | 一行拿到 DINOv2/v3、ViT、ConvNeXt 权重，做线性探测 | 约 2 MB |
| `albumentations` | 颜色/几何/天气增强，域适应实验的核心工具 | 约 5 MB |
| `pycocotools` | **官方 COCO 评测器**，mAP 口径才被评审认可 | 约 1 MB（Windows 上可能要编译） |
| `transformers` | 加载 HF 上的 RT-DETR / DINOv3 权重 | 已装，升级约 30 MB |
| `einops` | 注意力模块里 reshape 更清晰 | 约 0.1 MB |

### 2.4 8GB 显存下的推荐配置（**关键，照抄即可**）

8 GB 是「能训、但必须省着用」的档位。以下配置已按 8GB 留出安全余量：

**检测（RT-DETR-l / RT-DETRv2-l）**

| 项 | 推荐值 | 理由 |
| --- | --- | --- |
| 输入分辨率 | **640×640** | 上 1024 显存翻 2.5 倍，8GB 会 OOM |
| batch size | **4** | 配 AMP 时 8GB 的安全点；OOM 就降到 2 |
| 混合精度 | **AMP（`torch.amp.autocast('cuda')` + `GradScaler`）** | 显存约省 40%，速度提升明显 |
| backbone | **冻结（freeze）前若干 stage** | 先把检测头训起来，收敛快且省显存 |
| 梯度累积 | 4 步（等效 batch 16） | 小 batch 下稳住 BN/优化器统计量 |
| 优化器 | AdamW，lr 1e-4（head）/ 1e-5（backbone） | backbone 冻结时可只留 head 的 lr |
| 输入尺寸换小 | 若仍 OOM，用 RT-DETR-s / ResNet18 backbone | 模型小一档比降分辨率更划算 |

**分类 / 初筛（DINOv2、DINOv3）**

| 项 | 推荐值 | 理由 |
| --- | --- | --- |
| 模型 | **DINOv2-small 或 ViT-B**；DINOv3 同理选小档 | ViT-L/7B 在 8GB 上不可训 |
| 策略 | **冻结 backbone + 线性探测（linear probing）** | 只训一个线性层，显存占用极低、小样本就有效 |
| 进阶 | 线性探测收敛后再上 **LoRA** 微调 | 可训练参数 <1%，显存增量极小 |
| batch size | 32～64（冻结 backbone 时可行） | 只有线性层要存梯度 |
| 输入分辨率 | 224 或 336 | DINOv3 支持分辨率外推，336 通常够用 |

> **一句话策略**：**先冻结 backbone 把两个头训起来，再解冻微调**。
> 8GB 显存下"全量微调大 backbone"是必 OOM 的路线，不要走。

### 2.5 占多少（第二档合计）

| 项 | 量级 |
| --- | --- |
| torch+torchvision cu128 | 约 2.5–3.5 GB（**覆盖原 CPU 版后净增约 2–3 GB**） |
| 训练档其余包 | 约 50 MB |
| **模型权重**（DINOv2-small + RT-DETR-l + SAM 2.1） | 约 2–4 GB |
| **数据集**（建议预留） | 按 5 万张 1080p 估算约 50–100 GB |
| **训练产物**（checkpoint/日志） | 每轮 checkpoint 约 100–500 MB |

### 2.6 验证 GPU 是否真的通了

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"
```

期望输出（关键看三点）：

```
torch 2.10.0+cu128          ← 必须带 +cu128，不能是 +cpu
cuda available: True        ← 必须是 True
device: NVIDIA GeForce RTX 5070 Laptop GPU
capability: (12, 0)         ← 必须是 (12, 0)，即 sm_120
```

再跑一次真实矩阵乘法，确认 kernel 真的能执行（这一步能抓出"编译时没带 sm_120"的问题）：

```bash
python -c "import torch; a=torch.randn(2048,2048,device='cuda'); b=torch.randn(2048,2048,device='cuda'); c=(a@b).sum().item(); print('GPU 矩阵乘法 OK:', round(c,2))"
```

---

## 3. 第三档：实验性（可跳过）

**用途**：做对比实验、端侧部署、开放词汇扩展。**不装也能完成主线交付。**

```bash
pip install ultralytics onnxruntime onnx onnxruntime-gpu
```

| 包 | 为什么要装（以及为什么可以跳过） | 占多少（量级） |
| --- | --- | --- |
| `ultralytics` | 两个用途：① 跑 YOLOv8/YOLO11 **基线做对照实验**；② `scripts/train_openvocab.py` 的 **YOLO-World** 路径（开放词汇，新病种免标注）。主线 RT-DETR 不需要它 | 约 30 MB |
| `onnx` | 导出 ONNX 图（`torch.onnx.export` 的依赖） | 约 15 MB |
| `onnxruntime` | **CPU 推理**后端，端侧/无 GPU 服务器部署 | 约 200 MB |
| `onnxruntime-gpu` | GPU 推理后端，需与 CUDA/cuDNN 版本严格匹配 | 约 250 MB |
| `peft` | 只在使用 `--peft lora`（检测/分类）时需要。**不装就走「冻结 backbone + 只训线性头」**，小样本场景效果接近 | 约 10 MB |

> ⚠️ **`onnxruntime` 与 `onnxruntime-gpu` 只能装一个**，同时装会互相覆盖并导致
> `ImportError: DLL load failed`。详见 §4.3。

### 3.1 首次训练要联网下载 backbone（离线环境看这里）

`scripts/train_det.py` / `train_cls.py` 默认从 Hugging Face 拉预训练权重，**首次运行需要联网**：

| 用途 | 默认权重 | 量级 |
| --- | --- | --- |
| RT-DETRv2 检测 | `PekingU/rtdetr_v2_r18vd` | ≈ 80 MB |
| DINOv2/v3 分类 | `facebook/dinov2-base` | ≈ 350 MB |

**离线/内网环境**的三种做法（脚本都已支持）：

```bash
# ① 指向本地已下载的权重目录
python -m scripts.train_cls --backbone /path/to/dinov2-base --data-root data/processed

# ② 检测脚本用 --pretrained / --backbone-dir / --processor-dir 指定本地路径
python -m scripts.train_det --pretrained /path/to/rtdetr_v2_r18vd --data-root data/processed

# ③ 先在有网机器上 huggingface-cli download 下好整个目录，再整体拷进内网
```

> 注意：本地目录要**完整**（含 `config.json` / 权重 / `preprocessor_config.json`），
> 否则 `transformers` 会把它当成模型名去联网查，报
> `... is not a local folder and is not a valid model identifier`——本机实测踩过这个坑。

---

## 4. 环境故障排查（中文逐步排查）

### 4.1 `torch.cuda.is_available()` 返回 False

按顺序排查：

1. **先确认版本后缀**
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```
   - 输出 `2.10.0+cpu` / `None` → **装的是 CPU 版**，回到 §2.2 重装 cu128。
   - 输出 `+cu128` 但仍是 False → 继续第 2 步。

2. **确认驱动与显卡是否被系统识别**
   ```bash
   nvidia-smi
   ```
   - 命令不存在 → 装 NVIDIA 驱动。
   - 报 `No devices were found` → 驱动没装好，或笔记本双显卡被 BIOS 关了独显。
   - 本机正常应显示 `Driver Version: 596.13`、`CUDA Version: 13.2`、`RTX 5070 Laptop GPU`。

3. **确认算力是否被编译进轮子（50 系特有陷阱）**
   ```bash
   python -c "import torch; print(torch.cuda.get_arch_list())"
   ```
   - 输出里**必须包含 `sm_120`**。若只有 `sm_80/sm_86/sm_90`，
     说明装的是 cu124 等旧轮子 → 在 5070 上必然失败，重装 cu128。

4. **确认 kernel 真能跑**（`is_available()` 为 True 也可能在算的时候才炸）
   ```bash
   python -c "import torch; print((torch.randn(1024,1024,device='cuda')@torch.randn(1024,1024,device='cuda')).sum().item())"
   ```
   - 报 `no kernel image is available for execution on the device` → **就是 sm_120 没编进去**，
     换 cu128 轮子。

5. **Windows 双显卡**：确保程序用的是独显。NVIDIA 控制面板 → 管理 3D 设置 → 程序设置 →
   给 `python.exe` 指定「高性能 NVIDIA 处理器」。

6. **CUDA_VISIBLE_DEVICES 被误设**：
   ```bash
   echo $env:CUDA_VISIBLE_DEVICES   # PowerShell
   ```
   输出为空是正常的；若为 `-1` 或 `""`，会屏蔽所有 GPU，清掉该环境变量。

### 4.2 显存 OOM（`CUDA out of memory`）

**先看是谁占着显存**：

```bash
nvidia-smi
```

本机常见现象：浏览器（msedge/msedgewebview2）、聊天软件会占几百 MB。**训练前先关掉它们**，
因为 8GB 里被桌面占掉 1.7GB 是常态（实测开机后已用约 1760 MiB）。

按性价比从高到低依次尝试：

| 顺序 | 措施 | 效果 |
| --- | --- | --- |
| 1 | 开 **AMP 混合精度** | 显存降约 40%，几乎不掉点 |
| 2 | 降 **batch size**（4 → 2 → 1） | 线性省显存，配合梯度累积补回等效 batch |
| 3 | 降 **输入分辨率**（1024 → 640） | 显存约按面积平方降 |
| 4 | **冻结 backbone** | 省掉 backbone 的激活与优化器状态 |
| 5 | `torch.cuda.empty_cache()` | 释放缓存碎片（治标，缓解碎片型 OOM） |
| 6 | 换**小一档模型**（RT-DETR-l → RT-DETR-s / ResNet18） | 结构性省显存 |
| 7 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 缓解碎片化导致的假 OOM |

PowerShell 里设环境变量：

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

**注意**：OOM 有时是**显存泄漏**而非真的不够——检查是否在验证循环里忘了 `torch.no_grad()`
（这是新手最常犯的一条，会让显存随步数持续上涨直至 OOM）。

### 4.3 `onnxruntime` 与 GPU 版本冲突

**症状**：`ImportError: DLL load failed while importing onnxruntime_pybind11_state`
或 `ImportError: cannot import name 'get_available_providers'`，或 CUDA provider 加载失败。

**根因**：`onnxruntime`（CPU）与 `onnxruntime-gpu`（GPU）**装在同一个环境里会互相覆盖**，
且 `onnxruntime-gpu` 的 CUDA/cuDNN 主版本必须和本机驱动匹配。

**排查与修复步骤**：

1. **看现在到底装了哪个、什么版本**
   ```bash
   pip list | findstr onnxruntime
   ```
   （PowerShell 用 `pip list | Select-String onnxruntime`）

2. **先彻底卸干净，再只装一个**
   ```bash
   pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
   ```
   - 只要 CPU 推理（部署到无 GPU 服务器 / 端侧）：
     ```bash
     pip install onnxruntime
     ```
   - 要 GPU 推理：
     ```bash
     pip install onnxruntime-gpu
     ```

3. **确认 provider 是否真的挂上了 CUDA**
   ```bash
   python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
   ```
   - 期望看到 `'CUDAExecutionProvider'` 在列表里。
   - 只有 `'CPUExecutionProvider'` → GPU 版没装对，或 CUDA/cuDNN 缺失。

4. **版本匹配**：`onnxruntime-gpu` 每个版本都绑定特定的 CUDA/cuDNN 主版本
   （例如 1.19/1.20 绑 CUDA 12.x）。**不确定时写「需核实」并去 onnxruntime 官方
   安装矩阵核对，不要凭猜装**。若本机 CUDA 是 13.2，可能需要在环境里补装对应
   CUDA 12.x 运行时，或直接用 CUDA 13 支持的更新版本（**需核实**）。

5. **实在搞不定的降级路线**（推荐给本项目）：
   **服务端不做 ONNX，直接用 PyTorch 推理**（`torch.compile` 或原生），
   ONNX 只用在端侧/小程序侧的小模型上，且用 CPU provider 即可。
   这样能完全绕开 CUDA 依赖地狱，答辩时也更好讲。

### 4.4 其他常见问题速查

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'openpyxl'` | 本机确实没装 | 属正常降级路径（见 §0.3）；要样式就 `pip install openpyxl` |
| 中文乱码 / `UnicodeEncodeError` | Windows 控制台默认 GBK | `tools/` 已自动 `reconfigure(encoding='utf-8')`；或 `chcp 65001` |
| 图片路径含中文读不出来 | `cv2.imread` 不支持非 ASCII 路径 | `tools/` 已改用 `np.fromfile` + `cv2.imdecode` 规避 |
| `cv2` 与 `numpy` ABI 不匹配 | 版本错配 | `pip install --upgrade opencv-python numpy` |
| `pycocotools` 在 Windows 编译失败 | 缺 C++ 编译工具链 | 用 `pip install pycocotools-windows`，或改用纯 numpy 自实现 mAP（**口径需在报告里说明**） |
| 训练 loss 为 NaN | AMP 下 fp16 溢出 | 用 `GradScaler`；或降 lr；或换 bf16（`torch.autocast(dtype=torch.bfloat16)`，Blackwell 原生支持） |

---

## 5. 三档安装命令汇总（复制即用）

**第一档 · 只跑服务（CPU 就够）**

```bash
pip install fastapi uvicorn "numpy>=2.0" pillow opencv-python pyyaml psutil requests
```

**第二档 · 本地训练（GPU，本机必做）**

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision && pip install timm albumentations pycocotools "transformers>=5.0" einops
```

**第三档 · 实验性（可跳过）**

```bash
pip install ultralytics onnx onnxruntime
```

---

## 6. 与 `tools/` 的依赖对应关系（说明为什么工具在缺包时不崩）

| 工具 | 强依赖 | 可选依赖 | 缺失时的降级行为 |
| --- | --- | --- | --- |
| `tools/tile_infer.py` | numpy、pillow | cv2、pyyaml | 无 cv2 → 用 PIL 读图并提示；无 pyyaml → 内置标签兜底表 |
| `tools/make_heatmap.py` | numpy、pillow | cv2、pyyaml | 无 cv2 → 用 PIL 读图；无 pyyaml → 内置严重度表 |
| `tools/vis_labels.py` | numpy、pillow | cv2、pyyaml | 无 pyyaml → 类别中文名回退英文；无中文字体 → 标签降级英文并提示 |
| `tools/report_export.py` | 无（纯标准库） | openpyxl | **无 openpyxl → 内置纯标准库 xlsx 写出器**（仍产合法 .xlsx） |
| `tools/bench_latency.py` | 无（urllib 兜底） | requests、psutil | 无 requests → urllib；无 psutil → ctypes 读 RSS + 中文提示 |

> 以上五条降级路径均已用合成数据**实测通过**，`--help` 全部退出码 0。

---

## 7. 需要你确认的事项

1. **`openpyxl` 到底装不装？** 当前实测未装，工具已能产出合法 xlsx（无样式）。
   如果你希望报表带配色/加粗/冻结窗格，请执行 `pip install openpyxl`（本文档不代为安装）。
2. **数据集磁盘配额**：主体数据集体积我按经验给了 50–100 GB 的估算，
   请按真实影像数量与分辨率核定。
3. **`onnxruntime-gpu` 与 CUDA 13.2 的版本对应关系**：标「需核实」，
   建议以 onnxruntime 官方安装矩阵为准，不要按本文档猜版本号。
4. **是否要做 YOLO 基线对照实验**：这决定第三档 `ultralytics` 装不装。
   从答辩角度建议装（有 baseline 对比更好讲），但主线不依赖它。
