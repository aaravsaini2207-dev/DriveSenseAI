import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.drivesense.object_depth import ObjectDepthEstimator
from src.drivesense.depth import DepthEstimator
from src.drivesense.tracker import ObjectTracker
from src.drivesense.video import resize_frame


VIDEO_PATH = Path("data/videos/road.mp4")
OUTPUT_PATH = Path("data/outputs/metrics/depth_metrics.json")

MAX_WIDTH = 640
DEPTH_INTERVAL = 10


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH.resolve()}")

    tracker = ObjectTracker(
        model_name="yolo11n.pt",
        confidence_threshold=0.25,
    )
    depth_estimator = DepthEstimator()
    object_depth_estimator = ObjectDepthEstimator()

    depth_history = defaultdict(list)

    frame_index = 0
    depth_inference_calls = 0
    cached_depth_map = None
    start = time.perf_counter()

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH.resolve()}")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame = resize_frame(frame, max_width=MAX_WIDTH)

            objects = tracker.update(frame)

            # Refresh depth periodically, matching the main pipeline.
            if (
                cached_depth_map is None
                or frame_index % DEPTH_INTERVAL == 0
            ):
                cached_depth_map = depth_estimator.estimate(frame)
                depth_inference_calls += 1

            for obj in objects:
                result = object_depth_estimator.estimate_object_depth(
                    cached_depth_map,
                    obj,
                )

                if result is not None:
                    depth_history[result.track_id].append({
                        "frame": frame_index,
                        "label": result.label,
                        "median_depth": result.median_depth,
                        "mean_depth": result.mean_depth,
                        "valid_pixels": result.valid_pixels,
                        "depth_map_refreshed": (
                            frame_index % DEPTH_INTERVAL == 0
                        ),
                    })

            frame_index += 1

            if frame_index % 50 == 0:
                print(f"Processed {frame_index} frames")

    finally:
        cap.release()

    elapsed = time.perf_counter() - start

    track_metrics = {}
    all_changes = []
    all_normalized_changes = []

    for track_id, records in depth_history.items():
        values = [record["median_depth"] for record in records]

        # Compare consecutive observations of the same tracker ID.
        changes = [
            abs(current - previous)
            for previous, current in zip(values, values[1:])
        ]

        # Scale changes by the track's average absolute depth.
        scale = statistics.mean(abs(value) for value in values)
        normalized_changes = [
            change / max(scale, 1e-8)
            for change in changes
        ]

        all_changes.extend(changes)
        all_normalized_changes.extend(normalized_changes)

        track_metrics[str(track_id)] = {
            "label": records[0]["label"],
            "observations": len(records),
            "first_frame": records[0]["frame"],
            "last_frame": records[-1]["frame"],
            "median_depth_mean": statistics.mean(values),
            "median_depth_std": (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            ),
            "mean_absolute_consecutive_change": (
                statistics.mean(changes) if changes else 0.0
            ),
            "mean_normalized_consecutive_change": (
                statistics.mean(normalized_changes)
                if normalized_changes else 0.0
            ),
            "depth_history": records,
        }

    sufficiently_observed = [
        track for track in track_metrics.values()
        if track["observations"] >= 3
    ]

    metrics = {
        "video": str(VIDEO_PATH),
        "frames_processed": frame_index,
        "processing_seconds": round(elapsed, 3),
        "evaluation_fps": round(frame_index / elapsed, 3) if elapsed else 0,
        "depth_interval": DEPTH_INTERVAL,
        "depth_inference_calls": depth_inference_calls,
        "tracks_with_valid_depth": len(track_metrics),
        "tracks_with_at_least_3_observations": len(sufficiently_observed),
        "total_valid_depth_observations": sum(
            len(records) for records in depth_history.values()
        ),
        "overall_mean_absolute_consecutive_change": (
            statistics.mean(all_changes) if all_changes else None
        ),
        "overall_mean_normalized_consecutive_change": (
            statistics.mean(all_normalized_changes)
            if all_normalized_changes else None
        ),
        "tracks": track_metrics,
        "limitations": [
            "Depth values are relative model outputs, not metres.",
            "Low variation does not prove depth accuracy.",
            "Object movement and camera motion can legitimately change depth.",
            "Tracker ID switches can mix different objects into one history.",
            "Depth is cached between inference frames, as in the main pipeline.",
            "This script evaluates consistency proxies, not ground-truth depth error.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n========== DEPTH CONSISTENCY EVALUATION ==========")
    print(f"Frames processed: {frame_index}")
    print(f"Depth inference calls: {depth_inference_calls}")
    print(f"Tracks with valid depth: {len(track_metrics)}")
    print(
        "Tracks with >= 3 observations: "
        f"{len(sufficiently_observed)}"
    )
    print(
        "Mean absolute consecutive depth change: "
        f"{metrics['overall_mean_absolute_consecutive_change']}"
    )
    print(
        "Mean normalized consecutive depth change: "
        f"{metrics['overall_mean_normalized_consecutive_change']}"
    )
    print(f"Evaluation FPS: {metrics['evaluation_fps']}")
    print(f"Results saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
