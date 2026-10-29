from dataclasses import dataclass
from pathlib import Path
import time

import cv2

from src.drivesense.depth import DepthEstimator
from src.drivesense.object_depth import ObjectDepthEstimator
from src.drivesense.lanes import LaneDetector
from src.drivesense.motion import MotionEstimator
from src.drivesense.tracker import ObjectTracker
from src.drivesense.video import VideoReader, resize_frame
from src.drivesense.profiling import StageProfiler


ROAD_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}


@dataclass
class PipelineReport:
    frames_processed: int
    processing_seconds: float
    average_fps: float
    output_path: str
    depth_inference_calls: int
    stage_timings: dict


class DriveSensePipeline:
    """Combine detection, tracking, motion, lanes, and relative depth."""

    def __init__(
        self,
        max_width: int = 640,
        depth_interval: int = 10,
    ) -> None:
        if depth_interval < 1:
            raise ValueError("depth_interval must be at least 1.")

        self.max_width = max_width
        self.depth_interval = depth_interval
        self._diagnostics_printed = False

        self.tracker = ObjectTracker(
            model_name="yolo11n.pt",
            confidence_threshold=0.25,
        )

        self.motion_estimator = MotionEstimator()
        self.lane_detector = LaneDetector()

        # Monocular depth estimation.
        self.depth_estimator = DepthEstimator()
        self.object_depth_estimator = ObjectDepthEstimator()

    def run(
        self,
        video_path: str | Path,
        output_path: str | Path,
        display: bool = True,
    ) -> PipelineReport:
        video_path = Path(video_path)
        output_path = Path(output_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Input video not found: {video_path.resolve()}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        depth_inference_count = 0

        cached_depth_map = None

        profiler = StageProfiler()
        start_time = time.perf_counter()
        writer = None

        with VideoReader(video_path) as video:
            info = video.info
            source_fps = info.fps if info.fps > 0 else 25.0

            print(f"Input resolution: {info.width}x{info.height}")
            print(f"Source FPS: {source_fps:.2f}")
            print(f"Source frames: {info.frame_count}")
            print(f"Depth inference interval: {self.depth_interval} frames")

            try:
                while True:
                    success, frame = video.read()

                    if not success:
                        break

                    with profiler.measure("resize"):
                        frame = resize_frame(
                            frame,
                            max_width=self.max_width,
                        )

                    timestamp = frame_count / source_fps

                    # 1. Object detection and tracking.
                    with profiler.measure("tracking_detection"):
                        tracked_objects = self.tracker.update(frame)

                    road_objects = [
                        obj
                        for obj in tracked_objects
                        if obj.label in ROAD_CLASSES
                    ]

                    # 2. Motion estimation.
                    with profiler.measure("motion"):
                        motion_estimates = self.motion_estimator.update(
                            road_objects,
                            timestamp=timestamp,
                        )

                    # 3. Lane detection.
                    with profiler.measure("lane_detection"):
                        lane_result = self.lane_detector.detect(frame)

                    # 4. Refresh the depth map periodically.
                    should_estimate_depth = (
                        cached_depth_map is None
                        or frame_count % self.depth_interval == 0
                    )

                    if should_estimate_depth:
                        with profiler.measure("depth_inference"):
                            cached_depth_map = (
                                self.depth_estimator.estimate(frame)
                            )

                        depth_inference_count += 1

                    # 5. Estimate relative depth inside each object box.
                    with profiler.measure("object_depth"):
                        object_depths = []

                        for obj in road_objects:
                            result = (
                                self.object_depth_estimator
                                .estimate_object_depth(
                                    cached_depth_map,
                                    obj,
                                )
                            )

                            if result is not None:
                                object_depths.append(result)

                    # Print diagnostics once, after processing ALL objects.
                    if not self._diagnostics_printed:
                        print(
                            f"Depth map shape: "
                            f"{cached_depth_map.shape}"
                        )
                        print(
                            f"Tracked road objects: "
                            f"{len(road_objects)}"
                        )
                        print(
                            f"Objects with valid depth: "
                            f"{len(object_depths)}"
                        )

                        for depth in object_depths:
                            print(
                                f"ID={depth.track_id} | "
                                f"class={depth.label} | "
                                f"relative_depth="
                                f"{depth.median_depth:.4f} | "
                                f"valid_pixels={depth.valid_pixels}"
                            )

                        self._diagnostics_printed = True

                    annotated = lane_result.overlay

                    motion_by_id = {
                        estimate.track_id: estimate
                        for estimate in motion_estimates
                    }

                    depth_by_id = {
                        estimate.track_id: estimate
                        for estimate in object_depths
                    }

                    # 6. Draw object detections, motion, and relative depth.
                    for obj in road_objects:
                        cv2.rectangle(
                            annotated,
                            (obj.x1, obj.y1),
                            (obj.x2, obj.y2),
                            (0, 255, 0),
                            2,
                        )

                        parts = [
                            f"ID {obj.track_id}",
                            obj.label,
                            f"{obj.confidence:.2f}",
                        ]

                        motion = motion_by_id.get(obj.track_id)

                        if motion is not None:
                            parts.append(
                                f"{motion.speed_pixels_per_second:.0f}px/s"
                            )

                        depth = depth_by_id.get(obj.track_id)

                        if depth is not None:
                            parts.append(f"rel={depth.median_depth:.3f}")

                        label = " ".join(parts)

                        label_y = max(20, obj.y1 - 8)

                        cv2.putText(
                            annotated,
                            label,
                            (obj.x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 0),
                            2,
                        )

                    # 7. Display pipeline status.
                    frame_count += 1

                    elapsed = time.perf_counter() - start_time
                    processing_fps = frame_count / max(elapsed, 1e-9)

                    cv2.putText(
                        annotated,
                        f"DriveSense AI | FPS: {processing_fps:.1f}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        annotated,
                        f"Tracked road objects: {len(road_objects)}",
                        (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        annotated,
                        f"Depth inference calls: {depth_inference_count}",
                        (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )

                    # 8. Initialize the output video writer.
                    if writer is None:
                        height, width = annotated.shape[:2]

                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            source_fps,
                            (width, height),
                        )

                        if not writer.isOpened():
                            raise RuntimeError(
                                f"Could not create output: "
                                f"{output_path.resolve()}"
                            )

                    writer.write(annotated)

                    if display:
                        cv2.imshow("DriveSense AI", annotated)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Stopped by user.")
                            break

            finally:
                if writer is not None:
                    writer.release()

                if display:
                    cv2.destroyAllWindows()

        elapsed = time.perf_counter() - start_time

        report = PipelineReport(
                    frames_processed=frame_count,
                    processing_seconds=elapsed,
                    average_fps=frame_count / max(elapsed, 1e-9),
                    output_path=str(output_path.resolve()),
                    depth_inference_calls=depth_inference_count,
                    stage_timings=profiler.summary(),
                )

        print("\n========== DRIVESENSE AI REPORT ==========")
        print(f"Frames processed: {report.frames_processed}")
        print(f"Processing time: {report.processing_seconds:.2f}s")
        print(f"Average processing FPS: {report.average_fps:.2f}")
        print(f"Depth inference calls: {depth_inference_count}")
        print(f"Output: {report.output_path}")

        profiler.report()

        return report