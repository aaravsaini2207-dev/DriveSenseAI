from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict, deque
import csv
import math

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = Path("data/videos/top_view.mp4")
MODEL_PATH = Path("yolo11n.pt")

OUTPUT_PATH = Path("data/outputs/top_view_tracking_test.mp4")
TRACKS_CSV_PATH = Path("data/outputs/top_view_tracks.csv")
TRACK_SUMMARY_PATH = Path("data/outputs/top_view_track_summary.csv")

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

FRAME_STRIDE = 2
MAX_FRAMES = 1200

CONFIDENCE = 0.05
IMAGE_SIZE = 1280

TILE_WIDTH = 1600
TILE_HEIGHT = 1200
TILE_OVERLAP = 0.20

NMS_THRESHOLD = 0.50

TRACK_BUFFER = 45
TRACK_HIGH_THRESHOLD = 0.20
TRACK_LOW_THRESHOLD = 0.05
NEW_TRACK_THRESHOLD = 0.20
MATCH_THRESHOLD = 0.80

VEHICLE_CLASSES = [2, 3, 5, 7]

CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Number of recent centre points retained for each trajectory.
TRAIL_LENGTH = 30

# Minimum centre movement in output pixels before drawing
# a direction arrow.
MIN_ARROW_MOVEMENT = 5


# ============================================================
# TILE UTILITIES
# ============================================================

def get_tile_positions(length, tile_size, overlap):

    if length <= tile_size:
        return [0]

    step = int(tile_size * (1 - overlap))

    positions = list(range(0, length - tile_size + 1, step))

    last_position = length - tile_size

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions


# ============================================================
# TILED DETECTION
# ============================================================

def detect_tiled(model, frame):

    height, width = frame.shape[:2]

    x_positions = get_tile_positions(
        width, TILE_WIDTH, TILE_OVERLAP
    )

    y_positions = get_tile_positions(
        height, TILE_HEIGHT, TILE_OVERLAP
    )

    detections = []

    for y1 in y_positions:
        for x1 in x_positions:

            x2 = min(x1 + TILE_WIDTH, width)
            y2 = min(y1 + TILE_HEIGHT, height)

            tile = frame[y1:y2, x1:x2]

            results = model.predict(
                source=tile,
                classes=VEHICLE_CLASSES,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False,
            )

            boxes = results[0].boxes

            if boxes is None:
                continue

            for box in boxes:

                coords = box.xyxy[0].tolist()

                coords[0] += x1
                coords[1] += y1
                coords[2] += x1
                coords[3] += y1

                detections.append([
                    coords[0],
                    coords[1],
                    coords[2],
                    coords[3],
                    float(box.conf[0].item()),
                    int(box.cls[0].item()),
                ])

    return detections


# ============================================================
# CROSS-TILE NON-MAXIMUM SUPPRESSION
# ============================================================

def remove_duplicate_detections(detections):

    filtered = []

    for class_id in VEHICLE_CLASSES:

        class_detections = [
            d for d in detections if d[5] == class_id
        ]

        if not class_detections:
            continue

        boxes = []
        scores = []

        for d in class_detections:

            x1, y1, x2, y2, confidence, _ = d

            boxes.append([
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1),
            ])

            scores.append(float(confidence))

        selected = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            CONFIDENCE,
            NMS_THRESHOLD,
        )

        if len(selected) == 0:
            continue

        for index in selected.flatten():
            filtered.append(class_detections[index])

    return filtered


# ============================================================
# BYTE TRACKER
# ============================================================

def create_tracker():

    args = SimpleNamespace(
        track_high_thresh=TRACK_HIGH_THRESHOLD,
        track_low_thresh=TRACK_LOW_THRESHOLD,
        new_track_thresh=NEW_TRACK_THRESHOLD,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESHOLD,
        fuse_score=True,
    )

    return BYTETracker(args)


def update_tracker(tracker, detections, frame):

    if detections:
        detection_array = np.asarray(
            detections,
            dtype=np.float32,
        )
    else:
        detection_array = np.empty(
            (0, 6),
            dtype=np.float32,
        )

    boxes = Boxes(
        detection_array,
        orig_shape=frame.shape[:2],
    )

    return tracker.update(boxes, frame)


# ============================================================
# TRAJECTORY AND MOVEMENT UTILITIES
# ============================================================

def get_center(x1, y1, x2, y2):

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y


def calculate_distance(point_a, point_b):

    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]

    return math.sqrt(dx ** 2 + dy ** 2)


def draw_trajectory(frame, points):

    if len(points) < 2:
        return

    # Draw the trajectory as connected line segments.
    for index in range(1, len(points)):

        previous_point = points[index - 1]
        current_point = points[index]

        # Older trail segments are thinner.
        thickness = max(
            1,
            int(3 * index / len(points)),
        )

        cv2.line(
            frame,
            previous_point,
            current_point,
            (255, 180, 0),
            thickness,
            cv2.LINE_AA,
        )

    # Highlight the latest vehicle centre.
    cv2.circle(
        frame,
        points[-1],
        4,
        (0, 255, 255),
        -1,
    )


def draw_direction_arrow(frame, points):

    if len(points) < 2:
        return

    start_point = points[-2]
    end_point = points[-1]

    movement = calculate_distance(
        start_point,
        end_point,
    )

    if movement < MIN_ARROW_MOVEMENT:
        return

    cv2.arrowedLine(
        frame,
        start_point,
        end_point,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
        tipLength=0.35,
    )


# ============================================================
# TRACK STATISTICS
# ============================================================

def update_track_statistics(
    track_statistics,
    track_id,
    center,
    frame_number,
    class_id,
):

    if track_id not in track_statistics:

        track_statistics[track_id] = {
            "class_id": class_id,
            "first_frame": frame_number,
            "last_frame": frame_number,
            "observations": 0,
            "cumulative_displacement": 0.0,
            "first_center_x": center[0],
            "first_center_y": center[1],
            "last_center_x": center[0],
            "last_center_y": center[1],
        }

    stats = track_statistics[track_id]

    previous_center = (
        stats["last_center_x"],
        stats["last_center_y"],
    )

    displacement = calculate_distance(
        previous_center,
        center,
    )

    stats["cumulative_displacement"] += displacement
    stats["last_center_x"] = center[0]
    stats["last_center_y"] = center[1]
    stats["last_frame"] = frame_number
    stats["observations"] += 1


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\nStarting overhead vehicle tracking with trajectories...\n")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH.resolve()}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH.resolve()}"
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    model = YOLO(str(MODEL_PATH))

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine source FPS.")

    output_fps = source_fps / FRAME_STRIDE

    tracker = create_tracker()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )

    if not video_writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    # --------------------------------------------------------
    # Tracking state
    # --------------------------------------------------------

    track_records = []

    track_statistics = {}

    trajectories = defaultdict(
    lambda: deque(maxlen=TRAIL_LENGTH)
)

    last_observed_frame = {}
    MAX_TRAIL_GAP = 2

    unique_track_ids = set()

    frame_index = 0
    processed_frames = 0

    total_detections = 0
    total_tracked = 0

    frames_with_detections = 0
    max_detections_in_frame = 0

    print(f"Source FPS: {source_fps:.2f}")
    print(f"Output FPS: {output_fps:.2f}")
    print(f"Frame stride: {FRAME_STRIDE}")
    print(f"Maximum source frames: {MAX_FRAMES}")
    print(f"Tile size: {TILE_WIDTH} x {TILE_HEIGHT}")
    print(f"YOLO image size: {IMAGE_SIZE}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"Trajectory history: {TRAIL_LENGTH} points")

    print("\nProcessing...\n")

    try:

        while frame_index < MAX_FRAMES:

            success, original_frame = cap.read()

            if not success:
                break

            current_frame = frame_index
            frame_index += 1

            if current_frame % FRAME_STRIDE != 0:
                continue

            # Calculate scaling using the actual source-frame dimensions.
            scale_x = FRAME_WIDTH / original_frame.shape[1]
            scale_y = FRAME_HEIGHT / original_frame.shape[0]

            # 1. Detect vehicles in original-resolution tiles.
            detections = detect_tiled(
                model,
                original_frame,
            )

            # 2. Remove duplicate detections.
            detections = remove_duplicate_detections(
                detections
            )

            current_detections = len(detections)

            # 3. Update ByteTrack.
            tracks = update_tracker(
                tracker,
                detections,
                original_frame,
            )

            # 4. Resize frame for output.
            output_frame = cv2.resize(
                original_frame,
                (FRAME_WIDTH, FRAME_HEIGHT),
            )

            current_tracked = 0

            # 5. Process tracked vehicles.
            for track in tracks:

                x1, y1, x2, y2 = track[:4]

                track_id = int(track[4])
                confidence = float(track[5])
                class_id = int(track[6])

                unique_track_ids.add(track_id)

                current_tracked += 1

                # Original-resolution centre.
                center_x, center_y = get_center(
                    x1, y1, x2, y2
                )

                # Output-resolution coordinates.
                out_x1 = int(x1 * scale_x)
                out_y1 = int(y1 * scale_y)
                out_x2 = int(x2 * scale_x)
                out_y2 = int(y2 * scale_y)

                center_output = (
                    int(center_x * scale_x),
                    int(center_y * scale_y),
                )

                # Store trajectory point.
                # Break the old trajectory if the track was not observed
                # in the immediately preceding processed frames.
                previous_frame = last_observed_frame.get(track_id)

                if (
                    previous_frame is not None
                    and processed_frames - previous_frame > MAX_TRAIL_GAP
                ):
                    trajectories[track_id].clear()

                trajectories[track_id].append(center_output)
                last_observed_frame[track_id] = processed_frames

                # Update per-track statistics.
                update_track_statistics(
                    track_statistics,
                    track_id,
                    (center_x, center_y),
                    processed_frames,
                    class_id,
                )

                # Save frame-level tracking record.
                track_records.append({
                    "frame": processed_frames,
                    "track_id": track_id,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "confidence": confidence,
                    "class_id": class_id,
                })

                # Draw trajectory.
                points = list(trajectories[track_id])

                draw_trajectory(
                    output_frame,
                    points,
                )

                draw_direction_arrow(
                    output_frame,
                    points,
                )

                # Draw bounding box.
                cv2.rectangle(
                    output_frame,
                    (out_x1, out_y1),
                    (out_x2, out_y2),
                    (0, 255, 0),
                    2,
                )

                vehicle_name = CLASS_NAMES.get(
                    class_id,
                    "Vehicle",
                )

                label = (
                    f"ID {track_id} | "
                    f"{vehicle_name} | "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    output_frame,
                    label,
                    (out_x1, max(out_y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # 6. Update global statistics.
            total_detections += current_detections
            total_tracked += current_tracked

            if current_detections > 0:
                frames_with_detections += 1

            max_detections_in_frame = max(
                max_detections_in_frame,
                current_detections,
            )

            processed_frames += 1

            # 7. Draw information overlay.
            cv2.putText(
                output_frame,
                (
                    f"Detections: {current_detections} | "
                    f"Active tracks: {current_tracked} | "
                    f"Unique IDs: {len(unique_track_ids)}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                output_frame,
                "OVERHEAD VEHICLE TRACKING",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            video_writer.write(output_frame)

            if processed_frames % 25 == 0:

                print(
                    f"Processed: {processed_frames} | "
                    f"Detections: {total_detections} | "
                    f"Tracked observations: {total_tracked} | "
                    f"Unique IDs: {len(unique_track_ids)}"
                )

    finally:

        cap.release()
        video_writer.release()

    # ========================================================
    # SAVE FRAME-LEVEL TRACK DATA
    # ========================================================

    TRACKS_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    track_fieldnames = [
        "frame",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "confidence",
        "class_id",
    ]

    with open(
        TRACKS_CSV_PATH,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=track_fieldnames,
        )

        writer.writeheader()
        writer.writerows(track_records)

    # ========================================================
    # SAVE PER-TRACK SUMMARY
    # ========================================================

    summary_fieldnames = [
        "track_id",
        "vehicle_class",
        "first_frame",
        "last_frame",
        "observations",
        "cumulative_displacement_pixels",
        "net_displacement_pixels",
    ]

    with open(
        TRACK_SUMMARY_PATH,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fieldnames,
        )

        writer.writeheader()

        for track_id, stats in sorted(
            track_statistics.items()
        ):

            first_center = (
                stats["first_center_x"],
                stats["first_center_y"],
            )

            last_center = (
                stats["last_center_x"],
                stats["last_center_y"],
            )

            net_displacement = calculate_distance(
                first_center,
                last_center,
            )

            writer.writerow({
                "track_id": track_id,
                "vehicle_class": CLASS_NAMES.get(
                    stats["class_id"],
                    "Vehicle",
                ),
                "first_frame": stats["first_frame"],
                "last_frame": stats["last_frame"],
                "observations": stats["observations"],
                "cumulative_displacement_pixels": round(
                    stats["cumulative_displacement"],
                    2,
                ),
                "net_displacement_pixels": round(
                    net_displacement,
                    2,
                ),
            })

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("OVERHEAD VEHICLE TRACKING COMPLETED")
    print("=" * 60)

    print(f"Frames processed: {processed_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Tracked observations: {total_tracked}")
    print(f"Unique track IDs: {len(unique_track_ids)}")

    print(f"Frames with detections: {frames_with_detections}")

    print(
        f"Maximum detections in one frame: "
        f"{max_detections_in_frame}"
    )

    print(
        f"Average detections per frame: "
        f"{total_detections / max(processed_frames, 1):.2f}"
    )

    print(f"\nVideo saved: {OUTPUT_PATH.resolve()}")
    print(f"Frame-level CSV: {TRACKS_CSV_PATH.resolve()}")
    print(f"Track summary CSV: {TRACK_SUMMARY_PATH.resolve()}")

    print("=" * 60)


if __name__ == "__main__":
    main()