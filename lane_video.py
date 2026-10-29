import time
from pathlib import Path

import cv2

from src.drivesense.lanes import LaneDetector
from src.drivesense.video import VideoReader, resize_frame


VIDEO_PATH = Path("data/videos/road.mp4")
OUTPUT_PATH = Path("data/outputs/lane_output.mp4")


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Input video not found: {VIDEO_PATH.resolve()}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    detector = LaneDetector()
    frame_count = 0
    start_time = time.perf_counter()
    writer = None

    with VideoReader(VIDEO_PATH) as video:
        info = video.info
        source_fps = info.fps if info.fps > 0 else 25.0

        print(f"Input resolution: {info.width}x{info.height}")
        print(f"Source FPS: {source_fps:.2f}")

        try:
            while True:
                success, frame = video.read()

                if not success:
                    break

                frame = resize_frame(frame, max_width=960)
                result = detector.detect(frame)
                output = result.overlay

                frame_count += 1

                elapsed = time.perf_counter() - start_time
                processing_fps = frame_count / max(elapsed, 1e-9)

                cv2.putText(
                    output,
                    f"DriveSense AI | FPS: {processing_fps:.1f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                if writer is None:
                    height, width = output.shape[:2]

                    writer = cv2.VideoWriter(
                        str(OUTPUT_PATH),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (width, height),
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not create output video: {OUTPUT_PATH}"
                        )

                writer.write(output)
                cv2.imshow("DriveSense AI - Lane Detection", output)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

        finally:
            if writer is not None:
                writer.release()

            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time

    print("\n========== LANE DETECTION REPORT ==========")
    print(f"Frames processed: {frame_count}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print(
        f"Average processing FPS: "
        f"{frame_count / max(elapsed, 1e-9):.2f}"
    )
    print(f"Output saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
