from dataclasses import dataclass
from time import perf_counter

from src.drivesense.tracker import TrackedObject


@dataclass
class MotionEstimate:
    """Image-plane motion for a tracked object."""

    track_id: int
    center_x: float
    center_y: float
    velocity_x: float
    velocity_y: float
    speed_pixels_per_second: float


class MotionEstimator:
    """Estimate image-plane velocity using tracked object centers."""

    def __init__(self, max_history_seconds: float = 2.0) -> None:
        self.max_history_seconds = max_history_seconds
        self.previous_positions: dict[
            int, tuple[float, float, float]
        ] = {}

    def update(
        self,
        tracked_objects: list[TrackedObject],
        timestamp: float | None = None,
    ) -> list[MotionEstimate]:
        """
        Update motion estimates.

        timestamp should be the video's elapsed time in seconds.
        If omitted, wall-clock time is used.
        """
        if timestamp is None:
            timestamp = perf_counter()

        estimates: list[MotionEstimate] = []
        current_positions: dict[
            int, tuple[float, float, float]
        ] = {}

        for obj in tracked_objects:
            center_x = (obj.x1 + obj.x2) / 2.0
            center_y = (obj.y1 + obj.y2) / 2.0

            previous = self.previous_positions.get(obj.track_id)

            velocity_x = 0.0
            velocity_y = 0.0

            if previous is not None:
                prev_time, prev_x, prev_y = previous
                dt = timestamp - prev_time

                if dt > 0:
                    velocity_x = (center_x - prev_x) / dt
                    velocity_y = (center_y - prev_y) / dt

            speed = (velocity_x**2 + velocity_y**2) ** 0.5

            estimates.append(
                MotionEstimate(
                    track_id=obj.track_id,
                    center_x=center_x,
                    center_y=center_y,
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    speed_pixels_per_second=speed,
                )
            )

            current_positions[obj.track_id] = (
                timestamp,
                center_x,
                center_y,
            )

        # Keep recent tracks, including temporarily missing objects.
        updated_history = {
            track_id: position
            for track_id, position in self.previous_positions.items()
            if timestamp - position[0] <= self.max_history_seconds
        }
        updated_history.update(current_positions)
        self.previous_positions = updated_history

        return estimates
