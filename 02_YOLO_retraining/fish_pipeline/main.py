from pathlib import Path
import argparse
import cv2
from tqdm import tqdm

from src.config_loader import load_config
from src.video_utils import (
    get_video_info,
    frame_generator,
    frame_list_generator,
    read_single_frame
)
from src.preprocess import apply_preprocessing
from src.yolo_inference import YOLOProcessor
from src.reports import (
    save_frame_counts_csv,
    create_max_frame_per_class_summary,
    create_class_totals_summary
)
from src.utils import ensure_dir
from src.video_writer import VideoWriter
from src.visualization import create_max_count_figure


def main(config_path="config.json"):
    config = load_config(config_path)

    video_path = config["video_path"]
    model_path = config["model_path"]
    output_dir = Path(config["output_dir"])
    class_names = config["class_names"]

    process_every_n_frames = config.get("process_every_n_frames", 1)
    max_frames = config.get("max_frames", None)

    save_annotated_frames = config.get("save_annotated_frames", True)
    save_frames_by_class = config.get("save_frames_by_class", True)
    save_detection_txt = config.get("save_detection_txt", False)

    save_annotated_video = config.get("save_annotated_video", False)
    annotated_video_name = config.get("annotated_video_name", "detections.mp4")

    conf_threshold = config.get("conf_threshold", 0.25)
    iou_threshold = config.get("iou_threshold", 0.45)
    device = config.get("device", "cpu")

    preprocess_cfg = config.get("preprocess", {})
    frame_selection_cfg = config.get("frame_selection", {})
    max_count_vis_cfg = config.get("max_count_visualization", {})

    ensure_dir(output_dir)

    annotated_dir = output_dir / "annotated_frames"
    raw_frames_dir = output_dir / "frames_by_class"
    labels_dir = output_dir / "labels"

    if save_annotated_frames:
        ensure_dir(annotated_dir)
    if save_frames_by_class:
        ensure_dir(raw_frames_dir)
    if save_detection_txt:
        ensure_dir(labels_dir)

    video_info = get_video_info(video_path)

    fps = video_info["fps"]
    total_frames = video_info["total_frames"]
    width = video_info["width"]
    height = video_info["height"]

    selection_mode = frame_selection_cfg.get("mode", "segment")

    if selection_mode == "segment":
        start_minute = frame_selection_cfg.get("start_minute", 0)
        duration_minutes = frame_selection_cfg.get("duration_minutes", None)

        start_frame = int(start_minute * 60 * fps)

        if duration_minutes is None:
            end_frame = total_frames
        else:
            end_frame = int((start_minute + duration_minutes) * 60 * fps)

        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        if end_frame <= start_frame:
            raise ValueError(
                f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}"
            )

        frames_to_process = len(range(start_frame, end_frame, process_every_n_frames))

        generator = frame_generator(
            video_path=video_path,
            step=process_every_n_frames,
            max_frames=max_frames,
            start_frame=start_frame,
            end_frame=end_frame
        )

        selection_info = {
            "selection_mode": "segment",
            "start_minute": start_minute,
            "duration_minutes": duration_minutes,
            "start_frame": start_frame,
            "end_frame": end_frame
        }

    elif selection_mode == "frame_list":
        frame_list = frame_selection_cfg.get("frame_list", [])
        if not frame_list:
            raise ValueError("frame_selection.mode is 'frame_list' but frame_list is empty")

        clean_frame_list = sorted(set(
            int(f) for f in frame_list if 0 <= int(f) < total_frames
        ))

        if max_frames is not None:
            clean_frame_list = clean_frame_list[:max_frames]

        if not clean_frame_list:
            raise ValueError("No valid frames found in frame_list")

        frames_to_process = len(clean_frame_list)

        generator = frame_list_generator(
            video_path=video_path,
            frame_list=clean_frame_list,
            max_frames=max_frames
        )

        selection_info = {
            "selection_mode": "frame_list",
            "start_minute": None,
            "duration_minutes": None,
            "start_frame": min(clean_frame_list),
            "end_frame": max(clean_frame_list),
            "num_selected_frames": len(clean_frame_list)
        }

    else:
        raise ValueError(f"Unknown frame_selection mode: {selection_mode}")

    print("\n========== VIDEO INFORMATION ==========")
    print(f"Video file: {video_info['video_name']}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Total duration (seconds): {video_info['duration_sec']:.2f}")

    print("\n========== PROCESSING SETTINGS ==========")
    print(f"Selection mode: {selection_info['selection_mode']}")
    print(f"Start frame: {selection_info['start_frame']}")
    print(f"End frame: {selection_info['end_frame']}")
    print(f"Frame sampling interval: {process_every_n_frames}")
    print(f"Frames to process: {frames_to_process}")
    print(f"Preprocessing method: {preprocess_cfg.get('method', 'none')}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Device: {device}")

    if selection_mode == "segment":
        print(f"Start minute: {selection_info['start_minute']}")
        print(f"Duration analyzed (minutes): {selection_info['duration_minutes']}")
    else:
        print(f"Selected frames count: {selection_info['num_selected_frames']}")

    processor = YOLOProcessor(
        model_path=model_path,
        class_names=class_names,
        device=device
    )

    video_writer = None
    if save_annotated_video:
        video_writer = VideoWriter(
            output_path=output_dir / annotated_video_name,
            fps=fps,
            width=width,
            height=height
        )

    frame_rows = []

    print("\nStarting video processing...\n")

    try:
        for frame_idx, frame_bgr in tqdm(
            generator,
            total=frames_to_process,
            desc="Processing frames",
            unit="frame"
        ):
            processed_bgr = apply_preprocessing(frame_bgr, preprocess_cfg)

            result = processor.infer(
                processed_bgr,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )

            counts = processor.count_instances_per_class(result)

            row = {
                "frame": frame_idx,
                "time_sec": frame_idx / fps,
                "time_min": frame_idx / fps / 60.0
            }

            for class_id in range(len(class_names)):
                row[f"count_class_{class_id}"] = counts.get(class_id, 0)

            frame_rows.append(row)

            annotated = None
            if save_annotated_frames or save_annotated_video:
                annotated = processor.draw_predictions(processed_bgr, result)

            if save_annotated_frames and annotated is not None:
                annotated_path = annotated_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(annotated_path), annotated)

            if save_annotated_video and annotated is not None:
                video_writer.write(annotated)

            if save_frames_by_class:
                present_classes = [cid for cid, c in counts.items() if c > 0]
                for class_id in present_classes:
                    class_dir = raw_frames_dir / f"class_{class_id}_{class_names[class_id]}"
                    ensure_dir(class_dir)
                    frame_path = class_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), processed_bgr)

            if save_detection_txt:
                h, w = processed_bgr.shape[:2]
                txt_path = labels_dir / f"frame_{frame_idx:06d}.txt"
                processor.save_yolo_txt(str(txt_path), result, w, h)

    finally:
        if video_writer is not None:
            video_writer.close()

    if len(frame_rows) == 0:
        print("No frames were processed.")
        return

    print("\nSaving CSV reports...")

    frame_counts_csv = output_dir / "frame_counts.csv"
    frame_counts_df = save_frame_counts_csv(frame_rows, frame_counts_csv)

    max_frame_df = create_max_frame_per_class_summary(frame_counts_df, class_names)
    max_frame_csv = output_dir / "max_frame_per_class.csv"
    max_frame_df.to_csv(max_frame_csv, index=False)

    class_totals_df = create_class_totals_summary(frame_counts_df, class_names)
    class_totals_csv = output_dir / "class_totals.csv"
    class_totals_df.to_csv(class_totals_csv, index=False)

    print(f"Saved: {frame_counts_csv}")
    print(f"Saved: {max_frame_csv}")
    print(f"Saved: {class_totals_csv}")

    if max_count_vis_cfg.get("enabled", False):
        print("\nCreating max-count visualization...")

        target_cfg = max_count_vis_cfg.get("target", {})
        target_frame = int(target_cfg["frame"])
        expected_max_count = int(target_cfg["expected_max_count"])
        output_name = max_count_vis_cfg.get("output_name", "max_count_comparison.png")

        original_bgr = read_single_frame(video_path, target_frame)
        processed_bgr = apply_preprocessing(original_bgr, preprocess_cfg)

        result = processor.infer(
            processed_bgr,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )

        counts = processor.count_instances_per_class(result)
        detected_count = int(sum(counts.values()))

        annotated_bgr = processor.draw_predictions(processed_bgr, result)

        fig_output_path = output_dir / output_name
        create_max_count_figure(
            original_bgr=original_bgr,
            annotated_bgr=annotated_bgr,
            expected_max_count=expected_max_count,
            detected_count=detected_count,
            output_path=fig_output_path
        )

        print(f"Saved: {fig_output_path}")

    print("\nProcessing completed successfully.")
    print("\nGenerated reports:")
    print("- frame_counts.csv -> counts per frame and per class")
    print("- max_frame_per_class.csv -> frame with highest count for each class")
    print("- class_totals.csv -> total detections and number of frames where each class appears")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    main(args.config)