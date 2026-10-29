from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class TrackedObject:
    """One object associated with a tracker ID."""

    track_id: int
    class_id: int
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class ObjectTracker:
    """YOLO detection and persistent object tracking."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        tracker_config: str = "bytetrack.yaml",
        max_box_area_ratio: float = 0.50,
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.tracker_config = tracker_config
        self.max_box_area_ratio = max_box_area_ratio

    def update(self, frame: np.ndarray) -> list[TrackedObject]:
        """
        Detect and track objects in the next video frame.

        Call this method sequentially on frames from the same video.
        """

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            imgsz=640,
            verbose=False,
            iou=0.45,
            device="cpu",
        )

        tracked_objects: list[TrackedObject] = []

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height

        if frame_area <= 0:
            return tracked_objects

        for result in results:
            boxes = result.boxes

            if boxes is None or boxes.id is None:
                continue

            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            coordinates = boxes.xyxy.int().cpu().tolist()

            for track_id, class_id, confidence, coords in zip(
                track_ids,
                class_ids,
                confidences,
                coordinates,
            ):
                x1, y1, x2, y2 = coords

                # Reject invalid bounding boxes.
                box_width = x2 - x1
                box_height = y2 - y1

                if box_width <= 0 or box_height <= 0:
                    continue

                # Reject implausibly large detections.
                box_area = box_width * box_height
                area_ratio = box_area / frame_area

                if area_ratio > self.max_box_area_ratio:
                    continue

                tracked_objects.append(
                    TrackedObject(
                        track_id=track_id,
                        class_id=class_id,
                        label=result.names[class_id],
                        confidence=float(confidence),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )

        return tracked_objects
