from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


# Resolve the repository relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPTH_REPO = PROJECT_ROOT / "external" / "Depth-Anything-V2"
CHECKPOINT = (
    DEPTH_REPO / "checkpoints" / "depth_anything_v2_vits.pth"
)

# Make the cloned repository importable without installing it as a package.
if str(DEPTH_REPO) not in sys.path:
    sys.path.insert(0, str(DEPTH_REPO))

# This repository is added to sys.path above and is not installed as a package,
# so static analyzers cannot resolve it from the workspace imports.
from depth_anything_v2.dpt import DepthAnythingV2  # pyright: ignore[reportMissingImports]


class DepthEstimator:
    """Estimate relative scene depth using Depth Anything V2 Small."""

    MODEL_CONFIG = {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    }

    def __init__(self, input_size: int = 308):
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"Depth checkpoint not found: {CHECKPOINT}"
            )

        self.device = torch.device("cpu")
        self.input_size = input_size

        self.model = DepthAnythingV2(**self.MODEL_CONFIG)

        # Explicitly load onto CPU; this machine does not use CUDA.
        state_dict = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a floating-point relative-depth map.

        Args:
            frame: OpenCV BGR image.

        Returns:
            Float32 array with the same height and width as the input.
            Values are relative predictions, not calibrated metres.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        return self.model.infer_image(
            frame,
            input_size=self.input_size,
        )

    @staticmethod
    def visualize(depth: np.ndarray) -> np.ndarray:
        """Convert a relative-depth map into a colour visualization."""
        depth = np.asarray(depth, dtype=np.float32)

        if depth.size == 0 or not np.isfinite(depth).any():
            raise ValueError("Depth map contains no valid values.")

        valid = np.isfinite(depth)
        low, high = np.percentile(depth[valid], [2, 98])

        normalized = np.zeros(depth.shape, dtype=np.uint8)

        if high > low:
            scaled = np.clip(
                (depth - low) / (high - low),
                0.0,
                1.0,
            )
            normalized[valid] = (scaled[valid] * 255).astype(
                np.uint8
            )

        return cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_INFERNO,
        )


def main():
    video_path = PROJECT_ROOT / "data" / "videos" / "road.mp4"
    output_dir = PROJECT_ROOT / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    try:
        success, frame = cap.read()
    finally:
        cap.release()

    if not success:
        raise RuntimeError("Could not read the first video frame.")

    print(f"Input frame: {frame.shape[1]}x{frame.shape[0]}")
    print("Loading Depth Anything V2 Small on CPU...")

    start_load = time.perf_counter()
    estimator = DepthEstimator()
    load_seconds = time.perf_counter() - start_load

    print(f"Model load time: {load_seconds:.2f} seconds")
    print("Estimating depth for one frame...")

    start_inference = time.perf_counter()
    depth = estimator.estimate(frame)
    inference_seconds = time.perf_counter() - start_inference

    visualization = estimator.visualize(depth)

    depth_path = output_dir / "depth_map.png"

    if not cv2.imwrite(str(depth_path), visualization):
        raise RuntimeError(f"Could not save depth map: {depth_path}")

    print("\nDepth estimation successful.")
    print(f"Depth-map shape: {depth.shape}")
    print(f"Depth range: {depth.min():.4f} to {depth.max():.4f}")
    print(f"Inference time: {inference_seconds:.2f} seconds")
    print(f"Output: {depth_path}")


if __name__ == "__main__":
    main()
