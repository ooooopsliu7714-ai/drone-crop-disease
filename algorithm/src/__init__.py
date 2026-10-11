"""无人机农作物病害识别 —— 算法侧源码包。

子包说明（本文件由「训练+导出」链路负责，勿与推理侧互相 import 循环）：
- ``src.core``  : 路径解析、标签体系（configs/labels.yaml）、YAML 配置合并。
- ``src.data``  : 数据准备、COCO/YOLO 读写、torch Dataset。
- ``src.train`` : 训练与导出核心（分类 / 检测 / 开放词汇）、统一推理入口
  ``src.train.predictors.build_predictor``、ONNX 导出 ``src.train.exporter``。

约定：仓库根目录 = 本文件所在目录的上一级。
"""

__all__ = ["core", "data", "train"]
