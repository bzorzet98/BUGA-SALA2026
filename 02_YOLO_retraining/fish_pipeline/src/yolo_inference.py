from collections import defaultdict
from pathlib import Path
import cv2
from ultralytics import YOLO

from src.utils import ensure_dir


class YOLOProcessor:
    def __init__(self, model_path: str, class_names: list, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.device = device

    def infer(self, frame_bgr, conf_threshold=0.25, iou_threshold=0.45):
        result = self.model.predict(
            source=frame_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        return result

    def count_instances_per_class(self, result):
        counts = defaultdict(int)

        if result.boxes is None or len(result.boxes) == 0:
            return counts

        classes = result.boxes.cls.cpu().numpy().astype(int)
        for cls_id in classes:
            counts[int(cls_id)] += 1

        return counts

    def draw_predictions(self, frame_bgr, result):
        output = frame_bgr.copy()

        if result.boxes is None or len(result.boxes) == 0:
            return output

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box[:4])

            if 0 <= cls_id < len(self.class_names):
                label_name = self.class_names[cls_id]
            else:
                label_name = f"class_{cls_id}"

            label = f"{label_name} {conf:.2f}"
            color = (0, 255, 0)
            if conf <= 0.5:
                color = (0, 0, 255)
            if  conf>0.5 and conf <= 0.75:
                color = (0, 165, 255)    
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

        return output

    def save_yolo_txt(self, txt_path: str, result, img_w: int, img_h: int):
        txt_path = Path(txt_path)
        ensure_dir(txt_path.parent)

        lines = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = box[:4]
                xc = ((x1 + x2) / 2.0) / img_w
                yc = ((y1 + y2) / 2.0) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))