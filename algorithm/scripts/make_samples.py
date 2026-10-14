"""生成极小合成样例数据到 ``data/samples/``（不下载任何东西，纯 numpy/PIL 画图）。

干什么：
    造 8 张 640x640 的合成航拍「田块」图（绿色底 + 确定性噪点 + 不同颜色的病斑色块），
    同时导出两套标注，用来把数据转换与评估脚本真正跑通：
        data/samples/images/*.jpg                 8 张合成图（jpg，RGB）
        data/samples/annotations.json             COCO 格式标注（categories 按 labels.yaml 的 class_id）
        data/samples/labels/*.txt                 YOLO 格式标注（cls cx cy w h，归一化）
        data/samples/classes.txt                  YOLO 类别顺序（= labels.yaml 顺序）
        data/samples/README.md                    样例说明（中文）
    类别覆盖：healthy / rice_blast / wheat_powdery_mildew / corn_northern_leaf_blight，
    病斑面积占比刻意做出「轻/中/重」三档，便于验证 severity 与病情指数计算。

怎么用::

    python -m scripts.make_samples                     # 生成到 data/samples
    python -m scripts.make_samples --out-dir data/samples --seed 7 --num-images 12

    # 生成后即可验证数据链路：
    python -m src.data.prepare_data --raw-dir data/samples --out-dir data/processed
    python -m scripts.eval --demo

依赖：numpy、Pillow（都是本机已装的轻量依赖，不会联网）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.labels import load_labels
from src.core.paths import DATA_DIR, ensure_dir

#: 每张图的「剧本」：(合成类别 disease_name, 病斑数量, 每个病斑边长, 噪声强度)
#: 病斑总面积占比刻意覆盖 healthy / light(<2%) / moderate(2%~10%) / severe(>=10%) 四档，
#: 便于验证 severity 分档与病情指数计算是否合理。
SCENARIOS: Tuple[Tuple[str, int, int, float], ...] = (
    ("healthy", 0, 0, 0.06),                 # 无病斑 -> healthy
    ("healthy", 0, 0, 0.05),                 # 无病斑 -> healthy
    ("rice_blast", 3, 55, 0.08),             # 3*55^2 / 640^2 = 2.2% -> moderate
    ("rice_blast", 4, 75, 0.10),             # 4*75^2 / 640^2 = 5.5% -> moderate
    ("wheat_powdery_mildew", 1, 52, 0.12),   # 0.66% -> light
    ("corn_northern_leaf_blight", 6, 60, 0.09),  # 5.3% -> moderate
    ("tomato_early_blight", 4, 165, 0.10),   # 26.6% -> severe
    ("potato_late_blight", 8, 96, 0.11),     # 18.0% -> severe
)


def _lesion_color(disease_name: str) -> Tuple[int, int, int]:
    """按病害给病斑上色（只是可视化区分，不代表真实病理颜色）。"""
    palette = {
        "rice_blast": (150, 120, 70),          # 灰褐
        "rice_bacterial_blight": (215, 205, 140),  # 黄白
        "rice_sheath_blight": (120, 150, 95),  # 灰绿
        "wheat_powdery_mildew": (235, 235, 225),  # 白粉
        "wheat_stripe_rust": (200, 150, 60),   # 锈黄
        "wheat_fusarium_head_blight": (225, 170, 180),  # 粉红
        "corn_northern_leaf_blight": (140, 120, 80),
        "corn_southern_leaf_blight": (170, 150, 100),
        "corn_rust": (185, 120, 55),
        "potato_late_blight": (95, 80, 70),    # 深褐
        "tomato_early_blight": (110, 90, 60),
        "tomato_late_blight": (70, 90, 70),
        "grape_black_rot": (60, 50, 60),
        "citrus_huanglongbing": (220, 215, 120),
        "healthy": (40, 150, 60),
    }
    return palette.get(disease_name, (120, 110, 90))


def _field_background(rng, height: int, width: int, noise: float):
    """合成绿色田块底图（确定性：用传入的 numpy Generator）。"""
    base = np_full(height, width, (46 + noise * 60, 120 + noise * 80, 52 + noise * 50))
    # 垄沟条纹（航拍常见纹理）
    base[::16, :, :] = base[::16, :, :] * 0.92
    # 噪声
    n = rng.normal(0.0, 8.0 + noise * 40.0, size=(height, width, 1))
    return clip_u8(base + n)


def np_full(h: int, w: int, rgb: Sequence[float]):
    """构造 HxWx3 的 float 数组（避免 import numpy 时写一长串代码）。"""
    import numpy as np

    arr = np.zeros((h, w, 3), dtype=np.float32)
    arr[:, :, 0] = float(rgb[0])
    arr[:, :, 1] = float(rgb[1])
    arr[:, :, 2] = float(rgb[2])
    return arr


def clip_u8(arr):
    """裁剪到 0~255 并转 uint8。"""
    import numpy as np

    return np.clip(arr, 0, 255).astype(np.uint8)


def _draw_lesion(arr, x: int, y: int, size: int, color: Sequence[int], rng):
    """在图像上画一个病斑（矩形 + 边缘羽化，模拟不规则病斑）。"""
    import numpy as np

    h, w = arr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + size), min(h, y + size)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    patch = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.float32)
    patch[:, :] = np.asarray(color, dtype=np.float32)
    # 用随机斑点让病斑边缘不规则（中心更实，边缘更稀疏）
    noise = rng.random((y2 - y1, x2 - x1))
    yy = np.linspace(-1, 1, y2 - y1)[:, None]
    xx = np.linspace(-1, 1, x2 - x1)[None, :]
    mask = (xx * xx + yy * yy) <= (0.85 + 0.15 * noise)
    alpha = mask.astype(np.float32) * 0.85
    arr[y1:y2, x1:x2] = arr[y1:y2, x1:x2] * (1 - alpha[..., None]) + patch * alpha[..., None]


def _blur(arr, radius: int = 1):
    """极简 3x3 均值模糊（模拟航拍分辨率损失，不引入 cv2 依赖）。"""
    import numpy as np

    if radius <= 0:
        return arr
    k = 3
    pad = k // 2
    a = arr.astype(np.float32)
    p = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    acc = np.zeros_like(a)
    for dy in range(k):
        for dx in range(k):
            acc += p[dy : dy + a.shape[0], dx : dx + a.shape[1]]
    return clip_u8(acc / (k * k))


def _box_positions(rng, count: int, size: int, img: int) -> List[Tuple[int, int]]:
    """确定性摆放病斑位置（尽量不重叠，避免超出边界）。"""
    import numpy as np

    positions: List[Tuple[int, int]] = []
    margin = 24
    if count <= 0:
        return positions
    cols = max(1, int(np.ceil(np.sqrt(count))))
    cell = max(size + 12, (img - 2 * margin) // cols)
    for i in range(count):
        r, c = divmod(i, cols)
        cx = margin + c * cell + int(rng.integers(0, max(1, cell - size - 4)))
        cy = margin + r * cell + int(rng.integers(0, max(1, cell - size - 4)))
        cx = int(min(max(margin, cx), img - size - margin))
        cy = int(min(max(margin, cy), img - size - margin))
        positions.append((cx, cy))
    return positions


def generate(out_dir: Path, num_images: int, size: int, seed: int) -> Dict[str, Any]:
    """生成合成样例并写盘，返回统计信息。"""
    import numpy as np
    from PIL import Image

    lb = load_labels()
    name2id = lb.name_to_id_map()
    img_dir = ensure_dir(out_dir / "images")
    lbl_dir = ensure_dir(out_dir / "labels")

    coco: Dict[str, Any] = {
        "info": {"description": "drone-crop-disease-algo 合成样例（scripts/make_samples.py）", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": c["class_id"], "name": c["disease_name"], "name_cn": c["disease_name_cn"],
             "supercategory": "crop_disease"}
            for c in lb.sidecar_classes()
        ],
    }

    ann_id = 1
    for i in range(int(num_images)):
        disease, n_lesion, lesion_size, noise = SCENARIOS[i % len(SCENARIOS)]
        rng = np.random.default_rng(int(seed) * 1000 + i)
        arr = _field_background(rng, size, size, noise)
        positions = _box_positions(rng, n_lesion, lesion_size, size)
        color = _lesion_color(disease)
        boxes: List[List[int]] = []
        for (x, y) in positions:
            _draw_lesion(arr, x, y, lesion_size, color, rng)
            boxes.append([x, y, min(size, x + lesion_size), min(size, y + lesion_size)])
        arr = _blur(arr, radius=1)

        stem = f"sample_{i:03d}_{disease}"
        fname = f"{stem}.jpg"
        Image.fromarray(arr).save(img_dir / fname, quality=92)

        coco["images"].append({"id": i + 1, "file_name": f"images/{fname}", "width": size, "height": size,
                               "_scenario": disease})
        yolo_lines: List[str] = []
        for (x1, y1, x2, y2) in boxes:
            w, h = float(x2 - x1), float(y2 - y1)
            cid = name2id[disease]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": i + 1,
                "category_id": cid,
                "bbox": [float(x1), float(y1), w, h],
                "area": round(w * h, 2),
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1
            yolo_lines.append(
                f"{cid} {(x1 + w / 2) / size:.6f} {(y1 + h / 2) / size:.6f} {w / size:.6f} {h / size:.6f}"
            )
        (lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

    coco_path = out_dir / "annotations.json"
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    classes_txt = out_dir / "classes.txt"
    classes_txt.write_text("\n".join(lb.names) + "\n", encoding="utf-8")

    names_yaml = out_dir / "data.yaml"
    names_yaml.write_text(
        "path: .\n"
        "train: images\n"
        "val: images\n"
        f"nc: {lb.num_classes}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(lb.names)),
        encoding="utf-8",
    )

    per_class: Dict[str, int] = {}
    for a in coco["annotations"]:
        nm = lb.name_of(int(a["category_id"]))
        per_class[nm] = per_class.get(nm, 0) + 1

    (out_dir / "README.md").write_text(
        "# data/samples —— 合成样例数据（不联网、可复现）\n\n"
        f"由 `python -m scripts.make_samples --num-images {num_images} --size {size} --seed {seed}` 生成。\n\n"
        "## 内容\n"
        "- `images/*.jpg`：8 张 640x640 合成「田块」图（绿色底 + 噪点 + 病斑色块）\n"
        "- `annotations.json`：COCO 格式标注（categories 严格按 configs/labels.yaml）\n"
        "- `labels/*.txt`：YOLO 格式标注（`cls cx cy w h`，已归一化）\n"
        "- `classes.txt`：YOLO 类别顺序（= labels.yaml 顺序）\n"
        "- `data.yaml`：YOLO 数据集描述（train/val 都指向 images）\n\n"
        "## 用途\n"
        "1. 验证数据转换：`python -m src.data.prepare_data --raw-dir data/samples --out-dir data/processed`\n"
        "2. 验证评估实现：`python -m scripts.eval --demo`（确定性模拟预测，无需权重）\n"
        "3. 冒烟测试训练入口（需先联网下载一次 backbone）：\n"
        "   `python -m scripts.train_cls --data-root data/processed --epochs 1 --max-steps-per-epoch 1 --batch-size 4`\n"
        "   `python -m scripts.train_det --data-root data/processed --epochs 1 --max-steps-per-epoch 1 --batch-size 1`\n\n"
        "## 注意\n"
        "样例只为跑通流程，**不能**用来衡量模型精度；真实训练请把标注数据放进 `data/raw/`。\n",
        encoding="utf-8",
    )

    return {
        "num_images": len(coco["images"]),
        "num_boxes": len(coco["annotations"]),
        "per_class": per_class,
        "coco": str(coco_path),
        "images_dir": str(img_dir),
        "labels_dir": str(lbl_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    ap = argparse.ArgumentParser(
        prog="python -m scripts.make_samples",
        description="生成 data/samples 的极小合成样例（8 张 640x640 + COCO json + YOLO txt）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n  python -m scripts.make_samples\n  python -m scripts.make_samples --num-images 12 --size 512 --seed 7\n",
    )
    ap.add_argument("--out-dir", default=str(DATA_DIR / "samples"), help="输出目录，默认 data/samples")
    ap.add_argument("--num-images", type=int, default=8, help="生成图片数量，默认 8")
    ap.add_argument("--size", type=int, default=640, help="图片边长（正方形），默认 640")
    ap.add_argument("--seed", type=int, default=2025, help="随机种子，默认 2025")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """入口：返回退出码（0 成功，3 缺依赖）。"""
    args = build_parser().parse_args(argv)
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception as exc:
        print(
            "缺少生成合成样例所需的依赖（numpy / Pillow）。\n"
            "  pip install numpy pillow\n"
            f"（原始错误：{exc}）",
            file=sys.stderr,
        )
        return 3
    try:
        info = generate(Path(args.out_dir), int(args.num_images), int(args.size), int(args.seed))
    except Exception as exc:
        print(f"[错误] 生成合成样例失败：{exc}", file=sys.stderr)
        return 1
    print("=" * 78)
    print("合成样例已生成")
    print("=" * 78)
    print(f"输出目录：{Path(args.out_dir).resolve()}")
    print(f"图片：{info['num_images']} 张（{args.size}x{args.size}）  标注框：{info['num_boxes']} 个")
    print("每类框数：" + ", ".join(f"{k}={v}" for k, v in sorted(info["per_class"].items())))
    print(f"COCO：{info['coco']}")
    print(f"YOLO：{info['labels_dir']}")
    print("\n下一步：")
    print("  python -m src.data.prepare_data --raw-dir data/samples --out-dir data/processed")
    print("  python -m scripts.eval --demo")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
