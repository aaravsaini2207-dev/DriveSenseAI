import cv2
import math
from pathlib import Path

VIDEO_PATH = Path("data/outputs/top_view_tracking_test.mp4")
OUTPUT_PATH = Path("data/outputs/track_524_gap_collage.jpg")

FRAME_NUMBERS = [469, 475, 485, 495, 505, 514, 516]

COLS = 2
CELL_WIDTH = 960
CELL_HEIGHT = 600
HEADER_HEIGHT = 45

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

frames = []

for frame_number in FRAME_NUMBERS:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()

    if not success:
        print(f"Could not read frame {frame_number}")
        continue

    # Resize while preserving aspect ratio.
    h, w = frame.shape[:2]
    scale = min(CELL_WIDTH / w, CELL_HEIGHT / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    frame = cv2.resize(frame, (new_w, new_h))

    cell = 255 * (cv2.UMat(CELL_HEIGHT, CELL_WIDTH, cv2.CV_8UC3).get() * 0)

    x = (CELL_WIDTH - new_w) // 2
    y = (CELL_HEIGHT - new_h) // 2

    cell[y:y + new_h, x:x + new_w] = frame

    # Add frame number.
    cv2.putText(
        cell,
        f"FRAME {frame_number}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    frames.append(cell)

cap.release()

if not frames:
    raise RuntimeError("No frames were extracted.")

rows = math.ceil(len(frames) / COLS)

collage = 255 * (cv2.UMat(
    rows * CELL_HEIGHT,
    COLS * CELL_WIDTH,
    cv2.CV_8UC3
).get() * 0)

for index, frame in enumerate(frames):
    row = index // COLS
    col = index % COLS

    y = row * CELL_HEIGHT
    x = col * CELL_WIDTH

    collage[y:y + CELL_HEIGHT, x:x + CELL_WIDTH] = frame

cv2.imwrite(str(OUTPUT_PATH), collage)

print(f"\nCollage saved: {OUTPUT_PATH.resolve()}")
print(f"Frames included: {FRAME_NUMBERS}")
print(f"Collage dimensions: {collage.shape[1]} x {collage.shape[0]}")
