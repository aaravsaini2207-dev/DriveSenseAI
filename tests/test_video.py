import numpy as np
import pytest

from src.drivesense.video import VideoInfo, resize_frame


def test_resize_frame_reduces_width_and_preserves_aspect_ratio():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    resized = resize_frame(frame, max_width=960)

    assert resized.shape == (540, 960, 3)


def test_resize_frame_does_not_enlarge_small_frame():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    resized = resize_frame(frame, max_width=960)

    assert resized is frame


def test_resize_frame_rejects_empty_frame():
    frame = np.empty((0, 0, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="empty frame"):
        resize_frame(frame)


@pytest.mark.parametrize("max_width", [0, -1])
def test_resize_frame_rejects_nonpositive_max_width(max_width):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="max_width must be positive"):
        resize_frame(frame, max_width=max_width)


def test_video_duration_is_calculated_correctly():
    info = VideoInfo(
        path="sample.mp4",
        fps=30.0,
        width=1920,
        height=1080,
        frame_count=300,
    )

    assert info.duration_seconds == pytest.approx(10.0)


def test_video_duration_is_zero_when_fps_is_invalid():
    info = VideoInfo(
        path="sample.mp4",
        fps=0.0,
        width=1920,
        height=1080,
        frame_count=300,
    )

    assert info.duration_seconds == 0.0
