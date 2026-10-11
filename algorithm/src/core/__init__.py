"""公共基础模块：路径、标签体系、配置。"""

from .paths import (  # noqa: F401
    PROJECT_ROOT,
    CONFIGS_DIR,
    DATA_DIR,
    MODELS_DIR,
    SCRIPTS_DIR,
    ensure_dir,
    model_dir,
)
from .labels import (  # noqa: F401
    LabelBook,
    load_labels,
    SEVERITIES,
)

__all__ = [
    "PROJECT_ROOT",
    "CONFIGS_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "SCRIPTS_DIR",
    "ensure_dir",
    "model_dir",
    "LabelBook",
    "load_labels",
    "SEVERITIES",
]
