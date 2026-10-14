"""项目脚本集合（命令行入口）。

约定：**所有脚本都在仓库根目录用 ``python -m scripts.<name>`` 运行**，
例如 ``python -m scripts.eval --demo``、``python -m scripts.train_cls --help``。
这样 ``src`` 包能被正确 import，也避免 ``python scripts/xxx.py`` 时的路径歧义。

脚本清单：
    make_samples     生成 data/samples 的极小合成样例（8 张 640x640 + COCO json）
    train_det        RT-DETRv2/v3 检测训练（transformers 主力 / ultralytics / paddle）
    train_cls        DINOv2/DINOv3 病害分类与小样本冷启动
    train_openvocab  YOLO-World / Grounding DINO 开放词汇与零样本
    autolabel        用 Grounding DINO 或已训模型生成伪标签（data/pseudo/*.txt）
    export_onnx      导出 ONNX + sidecar json（训练脚本最后一步也会调用）
    eval             纯 numpy 检测评估 + 农业指标中文报告
"""

__all__ = [
    "make_samples",
    "train_det",
    "train_cls",
    "train_openvocab",
    "autolabel",
    "export_onnx",
    "eval",
]
