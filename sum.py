import os
from pathlib import Path

def count_dataset(split_dir):
    image_dir = Path(split_dir) / "images"
    label_dir = Path(split_dir) / "labels"

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in image_exts]
    labels = list(label_dir.glob("*.txt"))

    box_count = 0
    for txt in labels:
        with open(txt, "r", encoding="utf-8") as f:
            lines = [line for line in f.readlines() if line.strip()]
            box_count += len(lines)

    return len(images), len(labels), box_count

base_dir = r"F:\数据集\数据集11111"

for split in ["train", "val", "test"]:
    split_path = os.path.join(base_dir, split)
    img_num, label_num, box_num = count_dataset(split_path)
    print(f"{split}: 图片数={img_num}, 标签文件数={label_num}, 标注框数={box_num}")