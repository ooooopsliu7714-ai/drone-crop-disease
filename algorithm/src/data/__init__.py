"""``src.data`` 子包：数据准备、COCO/YOLO 读写、torch Dataset。

只有「训练+导出」链路使用本包；推理侧 (``src/infer/``) 通过
``src.train.predictors.build_predictor`` 使用训练产物，不需要 import 本包。
"""

from .io import (  # noqa: F401
    coco_to_yolo_lines,
    load_coco,
    resolve_image_path,
    save_coco,
    yolo_lines_to_coco,
)

__all__ = [
    "load_coco",
    "save_coco",
    "resolve_image_path",
    "coco_to_yolo_lines",
    "yolo_lines_to_coco",
]
