from pathlib import Path
import pandas as pd

from src.utils import ensure_dir


def save_frame_counts_csv(rows: list, output_csv: str):
    df = pd.DataFrame(rows)
    ensure_dir(Path(output_csv).parent)
    df.to_csv(output_csv, index=False)
    return df


def create_max_frame_per_class_summary(frame_counts_df: pd.DataFrame, class_names: list):
    summary_rows = []

    for class_id, class_name in enumerate(class_names):
        col = f"count_class_{class_id}"

        if col not in frame_counts_df.columns:
            summary_rows.append({
                "class_id": class_id,
                "class_name": class_name,
                "frame_with_max_count": None,
                "max_count": 0
            })
            continue

        max_count = frame_counts_df[col].max()

        if pd.isna(max_count):
            frame_with_max = None
            max_count = 0
        else:
            subset = frame_counts_df[frame_counts_df[col] == max_count]
            frame_with_max = int(subset.iloc[0]["frame"]) if len(subset) > 0 else None

        summary_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "frame_with_max_count": frame_with_max,
            "max_count": int(max_count)
        })

    return pd.DataFrame(summary_rows)


def create_class_totals_summary(frame_counts_df: pd.DataFrame, class_names: list):
    rows = []

    for class_id, class_name in enumerate(class_names):
        col = f"count_class_{class_id}"

        total_count = int(frame_counts_df[col].sum()) if col in frame_counts_df.columns else 0
        frames_with_presence = int((frame_counts_df[col] > 0).sum()) if col in frame_counts_df.columns else 0

        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "total_detections": total_count,
            "frames_with_presence": frames_with_presence
        })

    return pd.DataFrame(rows)