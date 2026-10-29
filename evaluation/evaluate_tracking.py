import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from src.drivesense.tracker import ObjectTracker
from src.drivesense.video import resize_frame


VIDEO_PATH = Path("data/videos/road.mp4")
OUTPUT_PATH = Path("data/outputs/metrics/tracking_metrics.json")
MAX_WIDTH = 640


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH.resolve()}")

    tracker = ObjectTracker(
        model_name="yolo11n.pt",
        confidence_threshold=0.25,
    )

    frame_records = []
    observations = defaultdict(list)
    class_counts = Counter()

    frame_index = 0
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

            frame_ids = []

            for obj in objects:
                frame_ids.append(obj.track_id)
                class_counts[obj.label] += 1

                observations[obj.track_id].append({
                    "frame": frame_index,
                    "class": obj.label,
                    "bbox": [obj.x1, obj.y1, obj.x2, obj.y2],
                    "confidence": round(obj.confidence, 4),
                })

            frame_records.append({
                "frame": frame_index,
                "tracked_objects": len(objects),
                "unique_ids": sorted(set(frame_ids)),
            })

            frame_index += 1

            if frame_index % 50 == 0:
                print(f"Processed {frame_index} frames")

    finally:
        cap.release()

    elapsed = time.perf_counter() - start

    track_lengths = {
        track_id: len(records)
        for track_id, records in observations.items()
    }

    frame_gaps = {}
    fragmented_tracks = 0

    for track_id, records in observations.items():
        frames = sorted(item["frame"] for item in records)
        gaps = sum(
            1 for previous, current in zip(frames, frames[1:])
            if current - previous > 1
        )
        frame_gaps[track_id] = gaps

        if gaps:
            fragmented_tracks += 1

    metrics = {
        "video": str(VIDEO_PATH),
        "frames_processed": frame_index,
        "processing_seconds": round(elapsed, 3),
        "evaluation_fps": round(frame_index / elapsed, 3) if elapsed else 0,
        "total_unique_track_ids": len(observations),
        "total_object_observations": sum(track_lengths.values()),
        "average_track_observations": round(
            statistics.mean(track_lengths.values()), 3
        ) if track_lengths else 0,
        "median_track_observations": statistics.median(
            track_lengths.values()
        ) if track_lengths else 0,
        "longest_track_observations": max(
            track_lengths.values(), default=0
        ),
        "tracks_with_gaps": fragmented_tracks,
        "total_reappearance_gaps": sum(frame_gaps.values()),
        "class_observations": dict(class_counts),
        "frame_records": frame_records,
        "tracks": {
            str(track_id): {
                "observations": track_lengths[track_id],
                "first_frame": observations[track_id][0]["frame"],
                "last_frame": observations[track_id][-1]["frame"],
                "gaps": frame_gaps[track_id],
                "class": observations[track_id][0]["class"],
                "observations_detail": observations[track_id],
            }
            for track_id in observations
        },
        "limitations": [
            "No ground-truth identity annotations were used.",
            "Track IDs are tracker outputs, not verified real-world identities.",
            "Track gaps do not necessarily indicate fragmentation errors.",
            "ID switches, IDF1, HOTA, and MOTA cannot be validated "
            "from these measurements alone.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n========== TRACKING EVALUATION ==========")
    print(f"Frames processed: {frame_index}")
    print(f"Unique track IDs: {len(observations)}")
    print(f"Object observations: {sum(track_lengths.values())}")
    print(f"Average observations per ID: {metrics['average_track_observations']}")
    print(f"Longest track: {metrics['longest_track_observations']} frames")
    print(f"Tracks with gaps: {fragmented_tracks}")
    print(f"Evaluation FPS: {metrics['evaluation_fps']}")
    print(f"Results saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
