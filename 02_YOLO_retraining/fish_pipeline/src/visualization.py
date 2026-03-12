from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from src.utils import ensure_dir


def create_max_count_figure(
    original_bgr,
    annotated_bgr,
    expected_max_count,
    detected_count,
    output_path
):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(original_rgb)
    axes[0].set_title(f"Original frame - expected max count: {expected_max_count}", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(annotated_rgb)
    axes[1].set_title(f"YOLO detections - counted BB: {detected_count}", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)