import cv2
from pathlib import Path

VIDEO = Path("data/videos/road.mp4")
OUTPUT = Path("data/evaluation/frames")
NUM_FRAMES = 20

OUTPUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames <= 0:
    cap.release()
    raise RuntimeError("Video contains no readable frames.")

indices = sorted({
    round(i * (total_frames - 1) / (NUM_FRAMES - 1))
    for i in range(NUM_FRAMES)
})

saved = 0

for index in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, frame = cap.read()

    if not success:
        print(f"Skipping unreadable frame {index}")
        continue

    output_path = OUTPUT / f"frame_{index:04d}.jpg"

    if cv2.imwrite(str(output_path), frame):
        saved += 1
    else:
        print(f"Could not save frame {index}")

cap.release()
print(f"Saved {saved} frames to {OUTPUT.resolve()}")