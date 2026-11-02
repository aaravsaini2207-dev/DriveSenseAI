from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import csv
import time

import cv2

from src.drivesense.tracker import ObjectTracker
from src.drivesense.video import VideoReader
from src.drivesense.profiling import StageProfiler


TRAFFIC_CLASSES = {
    "car",
    "bicycle",
    "person",
}

CLASS_ORDER = [
    "car",
    "bicycle",
    "person",
]

CLASS_COLORS = {
    "car": (0, 200, 255),
    "bicycle": (255, 180, 0),
    "person": (80, 220, 100),
}


@dataclass
class PipelineReport:
    frames_processed: int
    processing_seconds: float
    average_fps: float
    output_path: str
    depth_inference_calls: int
    stage_timings: dict


class TrajectoryPipeline:

    def __init__(
        self,
        max_width: int = 640,
        depth_interval: int = 15,
    ):
        if depth_interval < 1:
            raise ValueError("depth_interval must be at least 1")

        # Retained for compatibility with existing scripts.
        # Original video resolution is preserved.
        self.max_width = max_width
        self.depth_interval = depth_interval

        self.tracker = ObjectTracker(
            model_name="yolo11n.pt",
            confidence_threshold=0.25,
        )

    @staticmethod
    def draw_label(frame, x, y, text, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.46
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )

        x = max(0, min(x, frame.shape[1] - tw - 10))
        y = max(th + 10, y)

        cv2.rectangle(
            frame,
            (x, y - th - 8),
            (x + tw + 8, y + baseline - 3),
            color,
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x + 4, y - 4),
            font,
            scale,
            (15, 20, 25),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def draw_hud(
        frame,
        frame_number,
        total_frames,
        video_time,
        processing_fps,
        current_counts,
        unique_by_class,
    ):
        height, width = frame.shape[:2]

        # Scale the HUD according to the video resolution.
        scale_factor = max(0.75, min(width / 1920, 1.35))
        font_scale = 0.47 * scale_factor
        line_height = int(25 * scale_factor)
        padding = int(13 * scale_factor)

        rows = [
            ("DRIVESENSEAI", (245, 248, 255), 2),
            ("TRAFFIC INTELLIGENCE", (165, 185, 205), 1),
            (f"FRAME       {frame_number:04d} / {total_frames:04d}",
             (235, 240, 245), 1),
            (f"VIDEO TIME  {video_time:06.2f}s",
             (235, 240, 245), 1),
            (f"PROCESS FPS {processing_fps:05.2f}",
             (100, 220, 255), 1),
        ]

        rows.append(("CUMULATIVE TRACKING", (165, 185, 205), 1))

        for name in CLASS_ORDER:
            rows.append((
                f"{name.title():<15} "
                f"{len(unique_by_class[name]):>3}",
                CLASS_COLORS[name],
                1,
            ))

        total_unique = sum(
            len(unique_by_class[name]) for name in CLASS_ORDER
        )

        rows.append(
    (f"TOTAL TRACK IDS       {total_unique}", (100, 220, 255), 2)
)

        panel_width = int(315 * scale_factor)
        panel_height = padding * 2 + len(rows) * line_height

        x1 = 14
        y1 = 14
        x2 = min(width - 10, x1 + panel_width)
        y2 = min(height - 10, y1 + panel_height)

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (13, 21, 33),
            -1,
        )

        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (90, 115, 140),
            1,
        )

        text_y = y1 + padding + int(14 * scale_factor)

        for text, color, thickness in rows:
            cv2.putText(
                frame,
                text,
                (x1 + padding, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            text_y += line_height

    def run(
        self,
        video_path: str | Path,
        output_path: str | Path,
        display: bool = True,
    ) -> PipelineReport:

        video_path = Path(video_path)
        output_path = Path(output_path)

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video not found: {video_path.resolve()}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = output_path.with_suffix(".csv")

        frame_count = 0
        records = []

        unique_by_class = defaultdict(set)

        # Store the last 30 positions for each tracked object.
        trajectories = defaultdict(lambda: deque(maxlen=30))

        profiler = StageProfiler()

        start_time = time.perf_counter()

        writer = None
        stopped_by_user = False

        try:
            with VideoReader(video_path) as video:

                info = video.info
                source_fps = info.fps if info.fps > 0 else 25.0
                total_frames = info.frame_count

                print(f"Input resolution: {info.width}x{info.height}")
                print(f"Source FPS: {source_fps:.2f}")
                print(f"Source frames: {total_frames}")
                print("Traffic participants: enabled")
                print("Lane detection: disabled")
                print("Depth inference: disabled")
                print("Bird's-eye view: disabled")
                print("Video resizing: disabled")

                while True:
                    success, frame = video.read()

                    if not success:
                        break

                    frame_start = time.perf_counter()

                    original_height, original_width = frame.shape[:2]

                    with profiler.measure("tracking_detection"):
                        tracked_objects = self.tracker.update(frame)

                    current_counts = defaultdict(int)

                    for obj in tracked_objects:

                        label = obj.label.lower().strip()

                        if label not in TRAFFIC_CLASSES:
                            continue

                        x1 = max(0, min(int(obj.x1), original_width - 1))
                        y1 = max(0, min(int(obj.y1), original_height - 1))
                        x2 = max(0, min(int(obj.x2), original_width - 1))
                        y2 = max(0, min(int(obj.y2), original_height - 1))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        track_id = int(obj.track_id)
                        color = CLASS_COLORS[label]

                        current_counts[label] += 1
                        unique_by_class[label].add(track_id)

                        # Calculate the bottom-center position of the object.
                        center_x = (x1 + x2) // 2
                        center_y = y2

                        # Store its recent positions.
                        trajectory_key = (label, track_id)

                        trajectories[trajectory_key].append(
                            (center_x, center_y)
                        )

                        points = list(trajectories[trajectory_key])

                        for i in range(1, len(points)):
                            thickness = max(1, int(3 * i / len(points)))

                            cv2.line(
                                frame,
                                points[i - 1],
                                points[i],
                                color,
                                thickness,
                                cv2.LINE_AA,
                            )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                        label_text = f"{label.upper()} | ID {track_id}"

                        self.draw_label(
                            frame,
                            x1,
                            y1 - 5,
                            label_text,
                            color,
                        )

                        records.append({
                            "frame": frame_count + 1,
                            "timestamp_seconds": round(
                                frame_count / source_fps, 3
                            ),
                            "track_id": track_id,
                            "class": label,
                            "confidence": round(
                                float(obj.confidence), 4
                            ),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        })

                    frame_count += 1

                    elapsed_frame = time.perf_counter() - frame_start
                    processing_fps = 1.0 / max(elapsed_frame, 1e-9)

                    video_time = (frame_count - 1) / source_fps

                    self.draw_hud(
                        frame=frame,
                        frame_number=frame_count,
                        total_frames=total_frames,
                        video_time=video_time,
                        processing_fps=processing_fps,
                        current_counts=current_counts,
                        unique_by_class=unique_by_class,
                    )

                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            source_fps,
                            (original_width, original_height),
                        )

                        if not writer.isOpened():
                            raise RuntimeError(
                                "Could not initialize output video writer."
                            )

                    writer.write(frame)

                    if display:
                        cv2.imshow(
                            "DriveSenseAI | Traffic Intelligence",
                            frame,
                        )

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stopped_by_user = True
                            break

                    if frame_count % 50 == 0:
                        print(
                            f"Processed {frame_count}/{total_frames} frames"
                        )

        finally:
            if writer is not None:
                writer.release()

            if display:
                cv2.destroyAllWindows()

        if frame_count == 0:
            raise RuntimeError(
                "Zero frames processed. Check the source video decoder."
            )

        fieldnames = [
            "frame",
            "timestamp_seconds",
            "track_id",
            "class",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]

        with csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            csv_writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            csv_writer.writeheader()
            csv_writer.writerows(records)

        elapsed = time.perf_counter() - start_time

        report = PipelineReport(
            frames_processed=frame_count,
            processing_seconds=elapsed,
            average_fps=frame_count / max(elapsed, 1e-9),
            output_path=str(output_path.resolve()),
            depth_inference_calls=0,
            stage_timings=profiler.summary(),
        )

        print("\n========== DRIVESENSEAI REPORT ==========")
        print(f"Frames processed: {frame_count}/{total_frames}")
        print(f"Processing time: {elapsed:.2f}s")
        print(f"Average processing FPS: {report.average_fps:.2f}")
        print(f"Stopped by user: {stopped_by_user}")
        print(f"Video: {output_path.resolve()}")
        print(f"CSV: {csv_path.resolve()}")

        print("\nCumulative unique track IDs:")
        for name in CLASS_ORDER:
            print(
                f"{name.title()}: "
                f"{len(unique_by_class[name])}"
            )

        profiler.report()

        return report