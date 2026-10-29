from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ObjectDepth:
    """Relative-depth statistics associated with a tracked object."""

    track_id: int
    label: str
    median_depth: float
    mean_depth: float
    valid_pixels: int


class ObjectDepthEstimator:
    """
    Extract relative-depth statistics inside tracked-object boxes.

    The returned values are model predictions, not distances in metres.
    """

    def __init__(
        self,
        horizontal_margin: float = 0.20,
        vertical_top_margin: float = 0.15,
        vertical_bottom_margin: float = 0.10,
    ):
        self.horizontal_margin = horizontal_margin
        self.vertical_top_margin = vertical_top_margin
        self.vertical_bottom_margin = vertical_bottom_margin

    def estimate_object_depth(
        self,
        depth_map: np.ndarray,
        tracked_object,
    ) -> ObjectDepth | None:
        """
        Estimate the median depth inside the central region of a box.

        Excluding box edges reduces contamination from background pixels.
        """

        if depth_map is None or depth_map.size == 0:
            return None

        height, width = depth_map.shape[:2]

        x1 = max(0, int(tracked_object.x1))
        y1 = max(0, int(tracked_object.y1))
        x2 = min(width, int(tracked_object.x2))
        y2 = min(height, int(tracked_object.y2))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width < 4 or box_height < 4:
            return None

        # Reject implausibly large detections for object-level depth.
        frame_area = width * height
        box_area_ratio = (box_width * box_height) / frame_area

        if box_area_ratio > 0.50:
            return None

        # Focus on the interior instead of the bounding-box edges.
        crop_x1 = int(x1 + box_width * self.horizontal_margin)
        crop_x2 = int(x2 - box_width * self.horizontal_margin)

        crop_y1 = int(y1 + box_height * self.vertical_top_margin)
        crop_y2 = int(y2 - box_height * self.vertical_bottom_margin)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        region = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]
        valid = region[np.isfinite(region)]

        if valid.size == 0:
            return None

        return ObjectDepth(
            track_id=int(tracked_object.track_id),
            label=str(tracked_object.label),
            median_depth=float(np.median(valid)),
            mean_depth=float(np.mean(valid)),
            valid_pixels=int(valid.size),
        )

    @staticmethod
    def draw(
        frame: np.ndarray,
        tracked_objects: list,
        object_depths: list[ObjectDepth],
    ) -> np.ndarray:
        """Draw tracked-object boxes with readable, repositioned labels."""

        output = frame.copy()
        frame_height, frame_width = output.shape[:2]

        depth_by_id = {
            result.track_id: result
            for result in object_depths
        }

        # Keep track of label positions to reduce overlap.
        occupied_labels = []

        for obj in tracked_objects:
            x1 = max(0, min(int(obj.x1), frame_width - 1))
            y1 = max(0, min(int(obj.y1), frame_height - 1))
            x2 = max(0, min(int(obj.x2), frame_width - 1))
            y2 = max(0, min(int(obj.y2), frame_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            result = depth_by_id.get(int(obj.track_id))

            color = (
                (255, 220, 0)
                if result is not None
                else (150, 150, 150)
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            if result is None:
                text = f"ID {obj.track_id} | {obj.label}"
            else:
                text = (
                    f"ID {result.track_id} | {result.label} "
                    f"| rel={result.median_depth:.2f}"
                )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                font,
                font_scale,
                thickness,
            )

            # Prefer above the box, then below, then inside it.
            candidates = [
                (x1, y1 - 6),
                (x1, y2 + text_height + 6),
                (x1, y1 + text_height + 6),
            ]

            chosen = None

            for text_x, text_bottom in candidates:
                text_x = max(
                    0,
                    min(text_x, frame_width - text_width - 1),
                )
                text_bottom = max(
                    text_height + 2,
                    min(text_bottom, frame_height - baseline - 1),
                )

                label_rect = (
                    text_x,
                    text_bottom - text_height - 2,
                    text_x + text_width,
                    text_bottom + baseline,
                )

                overlaps = any(
                    not (
                        label_rect[2] < previous[0]
                        or label_rect[0] > previous[2]
                        or label_rect[3] < previous[1]
                        or label_rect[1] > previous[3]
                    )
                    for previous in occupied_labels
                )

                if not overlaps:
                    chosen = (text_x, text_bottom, label_rect)
                    break

            if chosen is None:
                # Last resort: draw above the box, even if labels overlap.
                text_x = max(
                    0,
                    min(x1, frame_width - text_width - 1),
                )
                text_bottom = max(
                    text_height + 2,
                    min(y1 - 6, frame_height - baseline - 1),
                )
                label_rect = (
                    text_x,
                    text_bottom - text_height - 2,
                    text_x + text_width,
                    text_bottom + baseline,
                )
                chosen = (text_x, text_bottom, label_rect)

            text_x, text_bottom, label_rect = chosen
            occupied_labels.append(label_rect)

            # Add a solid background for readability.
            cv2.rectangle(
                output,
                (label_rect[0], label_rect[1]),
                (label_rect[2], label_rect[3]),
                (30, 30, 30),
                -1,
            )

            cv2.putText(
                output,
                text,
                (text_x, text_bottom),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        return output