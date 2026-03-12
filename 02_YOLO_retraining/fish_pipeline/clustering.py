import ast
from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def parse_bbox(bbox_value):
    if isinstance(bbox_value, str):
        return ast.literal_eval(bbox_value)
    if isinstance(bbox_value, list):
        return bbox_value
    raise ValueError(f"Unsupported bbox format: {bbox_value}")


def yolo_to_xyxy(bbox, img_w, img_h):
    x_center, y_center, bw, bh = bbox

    x_center *= img_w
    y_center *= img_h
    bw *= img_w
    bh *= img_h

    x1 = int(round(x_center - bw / 2))
    y1 = int(round(y_center - bh / 2))
    x2 = int(round(x_center + bw / 2))
    y2 = int(round(y_center + bh / 2))

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2


def get_cluster_color(cluster_label: int):
    if cluster_label == -1:
        return (128, 128, 128)  # gray for noise

    palette = [
        #(255, 0, 0),     # blue
        (0, 255, 0),     # green
        (0, 0, 255),     # red
        (255, 255, 0),   # cyan
        (255, 0, 255),   # magenta
        (0, 255, 255),   # yellow
        (0, 128, 255),
        (128, 0, 255),
        (255, 128, 0),
        (128, 255, 0),
    ]
    return palette[cluster_label % len(palette)]


def extract_frame(video_path: Path, frame_id: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Could not read frame {frame_id} from {video_path}")

    return frame


def draw_colored_boxes(frame_bgr, df_frame):
    """
    Draw only bounding box borders and labels.
    """
    output = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]

    for _, row in df_frame.iterrows():
        bbox = parse_bbox(row["bbox"])
        cluster_label = int(row["cluster_label"])
        is_noise = str(row["is_noise"]).strip().upper() == "TRUE"
        is_core_sample = str(row["is_core_sample"]).strip().upper() == "TRUE"
        distance = float(row["distance_to_cluster_mean"])

        x1, y1, x2, y2 = yolo_to_xyxy(bbox, w, h)
        color = get_cluster_color(cluster_label)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        label = f"c:{cluster_label}"
        if is_noise:
            label += " noise"
        if is_core_sample:
            label += " core"
        label += f" d:{distance:.2f}"

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA
        )

    return output


def save_comparison_figure(original_bgr, boxes_bgr, output_path: Path, title_left="Original", title_right="Bounding boxes"):
    ensure_dir(output_path.parent)

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    boxes_rgb = cv2.cvtColor(boxes_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(original_rgb)
    axes[0].set_title(title_left, fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(boxes_rgb)
    axes[1].set_title(title_right, fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    csv_path = Path("/home/jpbarrientos/code/BUGA-SALA2026/clustering_results.csv")
    videos_dir = Path("/home/jpbarrientos/code/BUGA-SALA2026/Datasets/videos/")
    output_dir = Path("/home/jpbarrientos/code/BUGA-SALA2026/zzz")
    ensure_dir(output_dir)

    target_frame = 1050
    target_video = "LGH020002"
    target_method = "dbscam_euclidean"
    video_extension = ".MP4"

    df = pd.read_csv(csv_path)

    df_frame = df[
        (df["frame_id"] == target_frame) &
        (df["video"] == target_video) &
        (df["method"] == target_method)
    ].copy()

    if len(df_frame) == 0:
        print("No rows found for the requested frame/video/method.")
        return

    video_path = videos_dir / f"{target_video}{video_extension}"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frame_bgr = extract_frame(video_path, target_frame)

    boxes_bgr = draw_colored_boxes(frame_bgr, df_frame)

    original_out = output_dir / f"{target_video}_frame_{target_frame:06d}_original.jpg"
    boxes_out = output_dir / f"{target_video}_frame_{target_frame:06d}_{target_method}_boxes.jpg"
    figure_out = output_dir / f"{target_video}_frame_{target_frame:06d}_{target_method}_comparison.png"

    cv2.imwrite(str(original_out), frame_bgr)
    cv2.imwrite(str(boxes_out), boxes_bgr)

    save_comparison_figure(
        original_bgr=frame_bgr,
        boxes_bgr=boxes_bgr,
        output_path=figure_out,
        title_left=f"Original - {target_video} frame {target_frame}",
        title_right=f"{target_method} - bounding boxes"
    )

    print(f"Saved original frame at: {original_out}")
    print(f"Saved bounding-box image at: {boxes_out}")
    print(f"Saved comparison figure at: {figure_out}")


if __name__ == "__main__":
    main()