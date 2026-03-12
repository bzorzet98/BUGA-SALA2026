from pathlib import Path
import cv2


def get_video_info(video_path: str) -> dict:
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    cap.release()

    return {
        "video_name": video_path.name,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": duration_sec
    }


def frame_generator(video_path: str, step: int = 1, max_frames=None, start_frame=0, end_frame=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    start_frame = max(0, start_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    yielded = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % step == 0:
            yield frame_idx, frame
            yielded += 1

            if max_frames is not None and yielded >= max_frames:
                break

        frame_idx += 1

    cap.release()


def frame_list_generator(video_path: str, frame_list, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clean_frames = []
    seen = set()

    for f in frame_list:
        f = int(f)
        if 0 <= f < total_frames and f not in seen:
            clean_frames.append(f)
            seen.add(f)

    clean_frames.sort()

    if max_frames is not None:
        clean_frames = clean_frames[:max_frames]

    for frame_idx in clean_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        yield frame_idx, frame

    cap.release()


def read_single_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= total_frames:
        cap.release()
        raise ValueError(f"Frame {frame_idx} is out of range [0, {total_frames - 1}]")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {frame_idx}")

    return frame