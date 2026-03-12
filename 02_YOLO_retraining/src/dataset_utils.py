from pathlib import Path
from collections import Counter
from typing import List

def get_images(images_dir: Path, image_extensions: List[str]):
    img_exts = {ext.lower() for ext in image_extensions}
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in img_exts])
    return images


def check_missing_labels(images, labels_dir: Path):
    missing = []

    for img_path in images:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            missing.append(img_path.name)

    return missing


def read_classes_from_label(label_path: Path):
    classes = []

    if not label_path.exists():
        return classes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                classes.append(int(parts[0]))

    return classes


def build_image_class_mapping(images, labels_dir: Path):
    img_to_classes = {}
    class_presence = Counter()

    for img_path in images:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        classes = read_classes_from_label(lbl_path)
        img_to_classes[img_path.name] = classes
        class_presence.update(set(classes))

    return img_to_classes, class_presence


def count_instances(label_folder: Path):
    counts = Counter()

    for txt_path in sorted(label_folder.glob("*.txt")):
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    counts[int(parts[0])] += 1

    return counts