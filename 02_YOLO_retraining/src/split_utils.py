import random
import shutil
from pathlib import Path
from math import floor


def make_dataset_folders(dataset_root: Path):
    folders = [
        dataset_root / "images/train",
        dataset_root / "images/val",
        dataset_root / "images/test",
        dataset_root / "labels/train",
        dataset_root / "labels/val",
        dataset_root / "labels/test",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def select_train_val_test_split(
    images,
    img_to_classes,
    class_presence,
    random_seed=42,
    min_images=10,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):
    random.seed(random_seed)

    if len(images) < min_images:
        raise ValueError(
            f"Esperaba al menos {min_images} imágenes, encontré {len(images)}"
        )

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(
            f"Las proporciones deben sumar 1.0 y actualmente suman {total_ratio}"
        )

    multi_class_images = []
    single_class_images = []

    for img_path in images:
        cls_set = set(img_to_classes[img_path.name])
        if len(cls_set) >= 2:
            multi_class_images.append(img_path)
        else:
            single_class_images.append(img_path)

    safe_multi_class = []
    unsafe_multi_class = []

    for img_path in multi_class_images:
        cls_set = set(img_to_classes[img_path.name])
        has_unique_class = any(class_presence[c] == 1 for c in cls_set)

        if has_unique_class:
            unsafe_multi_class.append(img_path)
        else:
            safe_multi_class.append(img_path)

    shuffled_images = images[:]
    random.shuffle(shuffled_images)

    n_total = len(shuffled_images)
    n_train = floor(n_total * train_ratio)
    n_val = floor(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    if n_val == 0 or n_test == 0:
        raise ValueError(
            f"Con {n_total} imágenes no alcanza para generar splits 80/10/10 válidos."
        )

    preferred_pool = safe_multi_class if len(safe_multi_class) >= (n_val + n_test) else multi_class_images

    if len(preferred_pool) >= (n_val + n_test):
        holdout_images = random.sample(preferred_pool, n_val + n_test)
        remaining_images = [p for p in shuffled_images if p not in holdout_images]
        random.shuffle(holdout_images)

        val_images = holdout_images[:n_val]
        test_images = holdout_images[n_val:]
        train_images = remaining_images
    else:
        val_images = shuffled_images[n_train:n_train + n_val]
        test_images = shuffled_images[n_train + n_val:]
        train_images = shuffled_images[:n_train]

    split_info = {
        "multi_class_images": multi_class_images,
        "single_class_images": single_class_images,
        "safe_multi_class": safe_multi_class,
        "unsafe_multi_class": unsafe_multi_class,
        "n_total": n_total,
        "n_train": len(train_images),
        "n_val": len(val_images),
        "n_test": len(test_images),
    }

    return train_images, val_images, test_images, split_info


def copy_pair(img_path: Path, labels_dir: Path, dataset_root: Path, split: str):
    lbl_path = labels_dir / f"{img_path.stem}.txt"

    if not lbl_path.exists():
        raise FileNotFoundError(f"No existe el label para {img_path.name}: {lbl_path}")

    shutil.copy2(img_path, dataset_root / f"images/{split}" / img_path.name)
    shutil.copy2(lbl_path, dataset_root / f"labels/{split}" / lbl_path.name)


def copy_split(train_images, val_images, test_images, labels_dir: Path, dataset_root: Path):
    for p in train_images:
        copy_pair(p, labels_dir, dataset_root, "train")

    for p in val_images:
        copy_pair(p, labels_dir, dataset_root, "val")

    for p in test_images:
        copy_pair(p, labels_dir, dataset_root, "test")