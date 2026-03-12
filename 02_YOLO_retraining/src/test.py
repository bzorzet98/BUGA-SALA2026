import json
from pathlib import Path
import cv2
import numpy as np
import csv
from ultralytics import YOLO


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_paths(source_dir, extensions):
    exts = {e.lower() for e in extensions}

    return sorted([
        p for p in Path(source_dir).iterdir()
        if p.suffix.lower() in exts
    ])


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h

    return [x1, y1, x2, y2]


def load_gt_boxes_for_class(label_path, target_class, img_w, img_h):
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls_id = int(parts[0])

            if cls_id != target_class:
                continue

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            boxes.append(
                yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
            )

    return boxes


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)

    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def match_predictions(pred_boxes, gt_boxes, iou_thr=0.5):
    pred_boxes = sorted(pred_boxes, key=lambda x: x["conf"], reverse=True)

    matched_gt = set()

    tp = 0
    fp = 0
    matched_ious = []

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt = -1

        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue

            iou = compute_iou(pred["box"], gt)

            if iou > best_iou:
                best_iou = iou
                best_gt = i

        if best_iou >= iou_thr:
            tp += 1
            matched_gt.add(best_gt)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn, matched_ious


def draw_predictions(image_bgr, pred_boxes, class_name):
    output = image_bgr.copy()

    for pred in pred_boxes:
        x1, y1, x2, y2 = map(int, pred["box"])
        conf = pred["conf"]

        label = f"{class_name}: {conf:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output


def main():

    config = load_config("../Configs/test_config.json")

    class_names = config["dataset"]["class_names"]
    image_extensions = config["dataset"]["image_extensions"]

    inference_cfg = config["inference"]

    model_path = inference_cfg["model_path"]
    source_dir = Path(inference_cfg["source_dir"])
    labels_dir = Path(inference_cfg["labels_dir"])
    output_dir = Path(inference_cfg["output_dir"])

    imgsz = inference_cfg["imgsz"]
    conf_thr = inference_cfg["conf"]
    device = inference_cfg["device"]
    target_class = inference_cfg.get("target_class", 0)
    iou_thr = inference_cfg.get("iou_threshold", 0.5)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results.csv"

    model = YOLO(model_path)

    image_paths = get_image_paths(source_dir, image_extensions)

    print("Imágenes encontradas:", len(image_paths))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_matched_ious = []

    rows = []

    for img_path in image_paths:

        image_bgr = cv2.imread(str(img_path))
        h, w = image_bgr.shape[:2]

        label_path = labels_dir / f"{img_path.stem}.txt"

        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_thr,
            device=device,
            verbose=False
        )[0]

        pred_boxes = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, classes, confs):

                if int(cls_id) != target_class:
                    continue

                pred_boxes.append({
                    "box": box[:4].tolist(),
                    "conf": float(conf)
                })

        gt_boxes = load_gt_boxes_for_class(label_path, target_class, w, h)

        tp, fp, fn, matched_ious = match_predictions(
            pred_boxes,
            gt_boxes,
            iou_thr=iou_thr
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_matched_ious.extend(matched_ious)

        img_mean_iou = float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0

        rows.append([
            img_path.name,
            len(gt_boxes),
            len(pred_boxes),
            tp,
            fp,
            fn,
            img_mean_iou
        ])

        output_img = draw_predictions(
            image_bgr,
            pred_boxes,
            class_names[target_class]
        )

        cv2.imwrite(str(output_dir / img_path.name), output_img)

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    mean_iou = float(np.mean(all_matched_ious)) if len(all_matched_ious) > 0 else 0.0

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "image",
            "gt_boxes",
            "pred_boxes",
            "TP",
            "FP",
            "FN",
            "mean_iou"
        ])

        writer.writerows(rows)

        writer.writerow([])
        writer.writerow(["TOTAL_TP", total_tp])
        writer.writerow(["TOTAL_FP", total_fp])
        writer.writerow(["TOTAL_FN", total_fn])
        writer.writerow(["Precision", precision])
        writer.writerow(["Recall", recall])
        writer.writerow(["F1", f1])
        writer.writerow(["Mean_IoU", mean_iou])

    print("\nResultados guardados en:", csv_path)


if __name__ == "__main__":
    main()