from pathlib import Path
import shutil
import json


CONFIG_PATH = Path("config.json")


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = [
        "labels_dir",
        "output_dir",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "make_backup",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Falta la llave requerida en el config: {key}")

    config["labels_dir"] = Path(config["labels_dir"])
    config["output_dir"] = Path(config["output_dir"])

    return config


def yolo_to_xyxy(xc, yc, w, h):
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return x1, y1, x2, y2


def box_fully_inside_region(x1, y1, x2, y2, rx1, rx2, ry1, ry2):
    return x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2


def process_label_file(label_path, output_path, x_min, x_max, y_min, y_max):
    removed = 0
    kept_lines = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h)

        if box_fully_inside_region(x1, y1, x2, y2, x_min, x_max, y_min, y_max):
            removed += 1
            continue

        kept_lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)

    return len(lines), removed, len(kept_lines)


def main():
    config = load_config(CONFIG_PATH)

    labels_dir = config["labels_dir"]
    output_dir = config["output_dir"]
    x_min = config["x_min"]
    x_max = config["x_max"]
    y_min = config["y_min"]
    y_max = config["y_max"]
    make_backup = config["make_backup"]

    backup_dir = output_dir / "_backup_originals"

    output_dir.mkdir(parents=True, exist_ok=True)

    if make_backup:
        backup_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(labels_dir.glob("*.txt"))

    total_files = 0
    total_boxes = 0
    total_removed = 0
    total_kept = 0

    for txt_file in txt_files:
        total_files += 1

        if make_backup and txt_file.resolve() == (output_dir / txt_file.name).resolve():
            shutil.copy2(txt_file, backup_dir / txt_file.name)
        elif make_backup:
            shutil.copy2(txt_file, backup_dir / txt_file.name)

        out_file = output_dir / txt_file.name

        n_boxes, n_removed, n_kept = process_label_file(
            txt_file,
            out_file,
            x_min,
            x_max,
            y_min,
            y_max,
        )

        total_boxes += n_boxes
        total_removed += n_removed
        total_kept += n_kept

        print(f"{txt_file.name}: total={n_boxes}, eliminadas={n_removed}, restantes={n_kept}")

    print("\n===== RESUMEN =====")
    print(f"Archivos procesados: {total_files}")
    print(f"BB totales: {total_boxes}")
    print(f"BB eliminadas: {total_removed}")
    print(f"BB restantes: {total_kept}")


if __name__ == "__main__":
    main()