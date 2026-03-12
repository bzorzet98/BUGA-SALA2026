import cv2
from pathlib import Path
from src.utils import ensure_dir


class VideoWriter:

    def __init__(self, output_path, fps, width, height):

        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()