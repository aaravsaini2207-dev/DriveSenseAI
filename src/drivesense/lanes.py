from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LaneResult:
    overlay: np.ndarray
    left_line: tuple[int, int, int, int] | None
    right_line: tuple[int, int, int, int] | None
    lane_detected: bool


class LaneDetector:
    """Baseline lane detector with road-region and geometry constraints."""

    def __init__(self, roi_top_ratio: float = 0.58) -> None:
        self.roi_top_ratio = roi_top_ratio

    def detect(self, frame: np.ndarray) -> LaneResult:
        if frame is None or frame.ndim != 3:
            raise ValueError("Expected a valid color image.")

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Restrict detection to the road area, excluding most of the dashboard.
        polygon = np.array(
            [[
                (int(width * 0.10), height - 1),
                (int(width * 0.40), int(height * self.roi_top_ratio)),
                (int(width * 0.60), int(height * self.roi_top_ratio)),
                (int(width * 0.90), height - 1),
            ]],
            dtype=np.int32,
        )

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        roi_edges = cv2.bitwise_and(edges, mask)

        segments = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=35,
            maxLineGap=25,
        )

        # Keep weighted candidate points for the two sides of the road.
        left_points = []
        right_points = []

        if segments is not None:
            for segment in segments:
                x1, y1, x2, y2 = segment[0]

                dx = x2 - x1
                dy = y2 - y1

                if dx == 0:
                    continue

                slope = dy / dx
                length = float(np.hypot(dx, dy))
                midpoint_x = (x1 + x2) / 2

                # Reject horizontal edges and nearly vertical artifacts.
                if not 0.45 <= abs(slope) <= 3.5:
                    continue

                # On a forward-facing camera, expected lane lines
                # generally diverge from a point near the horizon.
                if slope < 0 and midpoint_x < width * 0.60:
                    left_points.extend([
                        (x1, y1, length),
                        (x2, y2, length),
                    ])
                elif slope > 0 and midpoint_x > width * 0.40:
                    right_points.extend([
                        (x1, y1, length),
                        (x2, y2, length),
                    ])

        def fit_lane(points, side):
            if len(points) < 4:
                return None

            data = np.asarray(points, dtype=np.float64)
            xs, ys, weights = data[:, 0], data[:, 1], data[:, 2]

            try:
                # Fit x as a function of y, which is convenient for lanes.
                coefficients = np.polyfit(ys, xs, deg=1, w=weights)
            except (ValueError, np.linalg.LinAlgError):
                return None

            x_per_y, intercept = coefficients

            y_bottom = int(height * 0.88)
            y_top = int(height * self.roi_top_ratio)

            x_bottom = int(x_per_y * y_bottom + intercept)
            x_top = int(x_per_y * y_top + intercept)

            # Require the line to occupy its expected side of the image.
            if side == "left":
                if not (
                    width * 0.05 <= x_bottom <= width * 0.55
                    and width * 0.20 <= x_top <= width * 0.65
                ):
                    return None

            else:
                if not (
                    width * 0.45 <= x_bottom <= width * 0.95
                    and width * 0.35 <= x_top <= width * 0.80
                ):
                    return None

            return (x_bottom, y_bottom, x_top, y_top)

        left_line = fit_lane(left_points, "left")
        right_line = fit_lane(right_points, "right")

        overlay = frame.copy()

        # Draw only the lane candidates that passed geometry checks.
        for line in (left_line, right_line):
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    5,
                )

        lane_detected = (
            left_line is not None and right_line is not None
        )

        status = (
            "LANE CANDIDATES DETECTED"
            if lane_detected
            else "LANE UNCERTAIN"
        )

        cv2.putText(
            overlay,
            status,
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255) if lane_detected else (0, 0, 255),
            2,
        )

        return LaneResult(
            overlay=overlay,
            left_line=left_line,
            right_line=right_line,
            lane_detected=lane_detected,
        )
