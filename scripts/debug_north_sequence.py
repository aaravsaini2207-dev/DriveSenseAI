import cv2
from pathlib import Path
from ultralytics import YOLO

VIDEO = "data/videos/top_view.mp4"
OUTPUT = Path("debug_north_sequence")
OUTPUT.mkdir(exist_ok=True)

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(VIDEO)

for frame_no in range(90, 111):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()

    if not ok:
        continue

    # Original 4K north-road crop
    crop = frame[0:900, 2400:3500]

    result = model.predict(
        crop,
        imgsz=960,
        conf=0.10,
        classes=[2, 3, 5, 7],
        device="cpu",
        verbose=False
    )[0]

    annotated = result.plot()

    print(f"Frame {frame_no}: {len(result.boxes)} detections")

    cv2.imwrite(
        str(OUTPUT / f"frame_{frame_no}.jpg"),
        annotated
    )

cap.release()

print("Finished. Inspect debug_north_sequence.")