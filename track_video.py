import time
from collections import Counter
from pathlib import Path

import cv2

from src.drivesense.tracker import ObjectTracker
from src.drivesense.video import VideoReader, resize_frame


VIDEO_PATH = Path("data/videos/road.mp4")
OUTPUT_PATH = Path("data/outputs/tracking_output.mp4")

ROAD_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Input video not found: {VIDEO_PATH.resolve()}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tracker = ObjectTracker(
        model_name="yolo11n.pt",
        confidence_threshold=0.25,
    )

    frame_count = 0
    total_tracked_objects = 0
    unique_track_ids: set[int] = set()
    class_counts: Counter[str] = Counter()
    start_time = time.perf_counter()

    with VideoReader(VIDEO_PATH) as video:
        info = video.info
        writer = None

        print(f"Video resolution: {info.width}x{info.height}")
        print(f"Source FPS: {info.fps:.2f}")
        print(f"Total source frames: {info.frame_count}")

        try:
            while True:
                success, frame = video.read()

                if not success:
                    break

                frame = resize_frame(frame, max_width=960)

                # Update tracker sequentially for each frame.
                tracked_objects = tracker.update(frame)

                visible_objects = [
                    obj
                    for obj in tracked_objects
                    if obj.label in ROAD_CLASSES
                ]

                for obj in visible_objects:
                    unique_track_ids.add(obj.track_id)
                    class_counts[obj.label] += 1

                    cv2.rectangle(
                        frame,
                        (obj.x1, obj.y1),
                        (obj.x2, obj.y2),
                        (0, 255, 0),
                        2,
                    )

                    label = (
                        f"ID {obj.track_id} | "
                        f"{obj.label} {obj.confidence:.2f}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (obj.x1, max(20, obj.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                frame_count += 1
                total_tracked_objects += len(visible_objects)

                elapsed = time.perf_counter() - start_time
                processing_fps = frame_count / max(elapsed, 1e-9)

                cv2.putText(
                    frame,
                    f"DriveSense AI | FPS: {processing_fps:.1f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Visible objects: {len(visible_objects)}",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                if writer is None:
                    height, width = frame.shape[:2]

                    writer = cv2.VideoWriter(
                        str(OUTPUT_PATH),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        info.fps if info.fps > 0 else 25.0,
                        (width, height),
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not create output: "
                            f"{OUTPUT_PATH.resolve()}"
                        )

                writer.write(frame)

                cv2.imshow(
                    "DriveSense AI - Multi-Object Tracking",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

        finally:
            if writer is not None:
                writer.release()

            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time

    print("\n========== TRACKING REPORT ==========")
    print(f"Frames processed: {frame_count}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Average processing FPS: {frame_count / max(elapsed, 1e-9):.2f}")
    print(f"Distinct track IDs observed: {len(unique_track_ids)}")
    print(f"Total tracked object observations: {total_tracked_objects}")
    print(f"Class observation counts: {dict(class_counts)}")
    print(f"Output saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
