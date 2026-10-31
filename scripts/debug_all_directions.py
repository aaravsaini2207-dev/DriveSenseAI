import cv2
from pathlib import Path
from ultralytics import YOLO

VIDEO = "data/videos/top_view.mp4"
OUTPUT = Path("debug_all_directions")
OUTPUT.mkdir(exist_ok=True)

MODEL_PATH = "yolo11n.pt"
FRAME_NUMBERS = [300, 360, 420, 480, 540, 600, 660, 720]

# Crops in ORIGINAL 3840x2160 coordinates.
# These are diagnostic starting regions, not final calibrated ROIs.
REGIONS = {
    "north": (2400, 0, 3500, 900),
    "south": (2400, 1200, 3500, 2160),
    "east":  (3000, 400, 3840, 1400),
    "west":  (0, 400, 1500, 1400),
}

CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

model = YOLO(MODEL_PATH)

summary = {}

for direction, (x1, y1, x2, y2) in REGIONS.items():

    direction_dir = OUTPUT / direction
    direction_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 15} {direction.upper()} {'=' * 15}")

    summary[direction] = []

    for frame_no in FRAME_NUMBERS:

        cap = cv2.VideoCapture(VIDEO)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            print(f"Frame {frame_no}: Could not read")
            continue

        crop = frame[y1:y2, x1:x2]

        result = model.predict(
            crop,
            imgsz=960,
            conf=0.10,
            classes=[2, 3, 5, 7],
            device="cpu",
            verbose=False
        )[0]

        annotated = crop.copy()
        detections = []

        if result.boxes is not None:
            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                bx1, by1, bx2, by2 = map(
                    int, box.xyxy[0].tolist()
                )

                # Convert crop coordinates to original 4K coordinates.
                global_box = (
                    bx1 + x1,
                    by1 + y1,
                    bx2 + x1,
                    by2 + y1
                )

                detections.append({
                    "class": CLASS_NAMES[cls],
                    "confidence": conf,
                    "box": global_box
                })

                label = f"{CLASS_NAMES[cls]} {conf:.2f}"

                cv2.rectangle(
                    annotated,
                    (bx1, by1),
                    (bx2, by2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    annotated,
                    label,
                    (bx1, max(25, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imwrite(
            str(direction_dir / f"frame_{frame_no}.jpg"),
            cv2.resize(annotated, (960, 720))
        )

        count = len(detections)
        summary[direction].append(count)

        print(f"Frame {frame_no}: {count} detections")

        for d in detections:
            print(
                f"  {d['class']:<12} "
                f"confidence={d['confidence']:.3f} "
                f"box={d['box']}"
            )

print("\n========== FINAL SUMMARY ==========")

for direction, counts in summary.items():
    avg = sum(counts) / len(counts) if counts else 0
    print(
        f"{direction.upper():<8} "
        f"Average detections/frame: {avg:.2f} "
        f"| Counts: {counts}"
    )

print(f"\nImages saved in: {OUTPUT.resolve()}")