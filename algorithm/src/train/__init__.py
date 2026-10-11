"""``src.train`` 子包：训练核心 + 导出 + 统一推理入口。

对外最重要的两个稳定接口（推理侧 ``src/infer/`` 依赖它们，签名不要改）：
    - ``src.train.predictors.build_predictor(ckpt_dir, device='auto')``
      返回带 ``predict(images: list[np.ndarray]) -> list[dict]`` 的对象；
    - ``src.train.exporter``：ONNX + sidecar json（``models/<name>/model.onnx.json``）。

子模块：
    common      —— 设备选择、随机种子、日志、JSONL 记录
    predictors  —— torch 后端推理（RT-DETR / ultralytics / DINO / Grounding DINO）
    exporter    —— ONNX 导出与 sidecar 元数据
    classifiers —— DINOv2/DINOv3 线性探测 / LoRA 分类训练
    detectors   —— RT-DETR 检测训练（transformers / ultralytics / Paddle）
    openvocab   —— YOLO-World + Grounding DINO 开放词汇与伪标签

注意：本 ``__init__`` **不预导入子模块**，避免 ``scripts/train_cls.py`` 之类脚本在
缺 transformers/ultralytics 的环境里 import 阶段就失败（惰性 import 是本项目的硬要求）。
"""

__all__ = ["common", "predictors", "exporter", "classifiers", "detectors", "openvocab"]
