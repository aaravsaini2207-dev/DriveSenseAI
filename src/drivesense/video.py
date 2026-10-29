from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        """Return the estimated video duration in seconds."""
        if self.fps <= 0:
            return 0.0

        return self.frame_count / self.fps


class VideoReader:
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self.capture = None
        self.info = None

    def open(self) -> VideoInfo:
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: {self.video_path}"
            )

        capture = cv2.VideoCapture(str(self.video_path))

        if not capture.isOpened():
            capture.release()
            raise ValueError(
                f"OpenCV could not open the video: {self.video_path}"
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise ValueError(
                "The video has invalid FPS or frame dimensions."
            )

        self.capture = capture

        self.info = VideoInfo(
            path=str(self.video_path),
            fps=fps,
            width=width,
            height=height,
            frame_count=max(frame_count, 0),
        )

        return self.info

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(
                "VideoReader is not open. Call open() first."
            )

        success, frame = self.capture.read()

        if not success:
            return False, None

        return True, frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def resize_frame(
    frame: np.ndarray,
    max_width: int = 960,
) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("Cannot resize an empty frame.")

    if max_width <= 0:
        raise ValueError("max_width must be positive.")

    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / width

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )