from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def global_time_to_local(time_mins, sub_video_name, sub_video_duration_min):
    idx = int(sub_video_name[3:5])
    local_time = time_mins - (idx - 1) * sub_video_duration_min
    return local_time


def prepare_ground_truth_for_class0(metrics_cfg: dict, fps: float) -> pd.DataFrame:
    labels_csv = Path(metrics_cfg["labels_csv"])
    if not labels_csv.exists():
        raise FileNotFoundError(f"No existe labels_csv: {labels_csv}")

    df = pd.read_csv(labels_csv)

    target_video_name = metrics_cfg["target_video_name"]
    video_filename_column = metrics_cfg["video_filename_column"]
    time_column = metrics_cfg["time_column"]
    gt_count_column = metrics_cfg["gt_count_column"]
    species_column = metrics_cfg["species_column"]
    target_class_name_in_csv = metrics_cfg["target_class_name_in_csv"]
    sub_video_duration_min = metrics_cfg["sub_video_duration_min"]

    vid_labels = df[df[video_filename_column] == target_video_name].copy()

    if species_column in vid_labels.columns:
        vid_labels = vid_labels[vid_labels[species_column] == target_class_name_in_csv].copy()

    sub_video_name = Path(target_video_name).stem
    vid_labels["local_time_min"] = vid_labels[time_column].apply(
        lambda t: global_time_to_local(t, sub_video_name, sub_video_duration_min)
    )
    vid_labels["local_frame"] = (vid_labels["local_time_min"] * 60 * fps).round().astype(int)

    gt_df = vid_labels[["local_frame", gt_count_column]].copy()
    gt_df = gt_df.rename(columns={gt_count_column: "gt_count_class0"})
    gt_df = gt_df.groupby("local_frame", as_index=False)["gt_count_class0"].max()

    return gt_df


def compute_class0_metrics(frame_counts_df: pd.DataFrame, gt_df: pd.DataFrame, positive_threshold: int = 1):
    pred_df = frame_counts_df[["frame", "count_class_0"]].copy()
    pred_df = pred_df.rename(columns={"count_class_0": "pred_count_class0"})

    merged = pred_df.merge(gt_df, how="outer", left_on="frame", right_on="local_frame")
    merged["frame"] = merged["frame"].fillna(merged["local_frame"])
    merged["pred_count_class0"] = merged["pred_count_class0"].fillna(0).astype(int)
    merged["gt_count_class0"] = merged["gt_count_class0"].fillna(0).astype(int)

    merged["y_true"] = (merged["gt_count_class0"] >= positive_threshold).astype(int)
    merged["y_pred"] = (merged["pred_count_class0"] >= positive_threshold).astype(int)

    y_true = merged["y_true"].values
    y_pred = merged["y_pred"].values

    metrics_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "evaluated_frames": len(merged)
    }

    return metrics_dict, merged