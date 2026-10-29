import time
from pathlib import Path

import cv2

from src.drivesense.motion import MotionEstimator
from src.drivesense.tracker import ObjectTracker
from src.drivesense.video import VideoReader, resize_frame


VIDEO_PATH = Path("data/videos/road.mp4")
OUTPUT_PATH = Path("data/outputs/motion_output.mp4")

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
    motion_estimator = MotionEstimator()

    processed_frames = 0
    start_time = time.perf_counter()
    writer = None

    with VideoReader(VIDEO_PATH) as video:
        info = video.info

        source_fps = info.fps if info.fps > 0 else 25.0
        print(f"Source FPS: {source_fps:.2f}")

        try:
            while True:
                success, frame = video.read()

                if not success:
                    break

                frame = resize_frame(frame, max_width=960)

                # Keep video time independent of processing speed.
                timestamp = processed_frames / source_fps

                tracked_objects = tracker.update(frame)
                road_objects = [
                    obj
                    for obj in tracked_objects
                    if obj.label in ROAD_CLASSES
                ]

                motion_estimates = motion_estimator.update(
                    road_objects,
                    timestamp=timestamp,
                )

                for estimate in motion_estimates:
                    obj = next(
                        (
                            candidate
                            for candidate in road_objects
                            if candidate.track_id == estimate.track_id
                        ),
                        None,
                    )

                    if obj is None:
                        continue

                    cv2.rectangle(
                        frame,
                        (obj.x1, obj.y1),
                        (obj.x2, obj.y2),
                        (0, 255, 0),
                        2,
                    )

                    label = (
                        f"ID {obj.track_id} | {obj.label} | "
                        f"v=({estimate.velocity_x:.0f}, "
                        f"{estimate.velocity_y:.0f}) px/s"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (obj.x1, max(20, obj.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        2,
                    )

                    # Draw a short vector representing image-plane motion.
                    arrow_end = (
                        int(estimate.center_x + estimate.velocity_x * 0.15),
                        int(estimate.center_y + estimate.velocity_y * 0.15),
                    )

                    cv2.arrowedLine(
                        frame,
                        (
                            int(estimate.center_x),
                            int(estimate.center_y),
                        ),
                        arrow_end,
                        (255, 0, 0),
                        2,
                        tipLength=0.25,
                    )

                processed_frames += 1
                elapsed = time.perf_counter() - start_time
                processing_fps = (
                    processed_frames / max(elapsed, 1e-9)
                )

                cv2.putText(
                    frame,
                    f"DriveSense AI | FPS: {processing_fps:.1f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                if writer is None:
                    height, width = frame.shape[:2]

                    writer = cv2.VideoWriter(
                        str(OUTPUT_PATH),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (width, height),
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not create output: {OUTPUT_PATH}"
                        )

                writer.write(frame)

                cv2.imshow("DriveSense AI - Motion Estimation", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

        finally:
            if writer is not None:
                writer.release()

            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time

    print("\n========== MOTION REPORT ==========")
    print(f"Frames processed: {processed_frames}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print(
        f"Average processing FPS: "
        f"{processed_frames / max(elapsed, 1e-9):.2f}"
    )
    print(f"Output: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()