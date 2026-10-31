import cv2
import torch
import csv
from pathlib import Path
from ultralytics import YOLO

VIDEO = "data/videos/top_view.mp4"
MODEL_PATH = "yolo11n.pt"

# Choose a frame where North-road vehicles are clearly visible.
FRAME_NUMBER = 90

VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

OUTPUT = Path("debug_north")
OUTPUT.mkdir(exist_ok=True)

CONFIDENCE = 0.15
IMAGE_SIZES = [960, 1280, 1600]

# Read original-resolution frame.
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)

success, original = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read selected video frame.")

print("Original frame:", original.shape)

# Load model independently of tracking.
model = YOLO(MODEL_PATH)

print("CUDA available:", torch.cuda.is_available())
print("Testing frame:", FRAME_NUMBER)

# Save resized reference.
resized = cv2.resize(original, (1280, 720))
cv2.imwrite(str(OUTPUT / "reference_1280.jpg"), resized)

all_results = []

for imgsz in IMAGE_SIZES:

    print(f"\nRunning YOLO: imgsz={imgsz}, conf={CONFIDENCE}")

    # Test original 4K frame, not the already resized frame.
    results = model.predict(
        source=original,
        imgsz=imgsz,
        conf=CONFIDENCE,
        classes=VEHICLE_CLASSES,
        verbose=False,
        device="cpu"
    )

    result = results[0]
    annotated = original.copy()

    detections = []

    if result.boxes is not None:
        for box in result.boxes:

            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            detections.append({
                "class": CLASS_NAMES.get(cls, str(cls)),
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

            label = (
                f"{CLASS_NAMES.get(cls, cls)} "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Resize annotated output for convenient viewing.
    preview = cv2.resize(annotated, (1280, 720))

    output_file = OUTPUT / f"raw_yolo_imgsz_{imgsz}.jpg"
    cv2.imwrite(str(output_file), preview)

    print("Detections:", len(detections))

    for d in detections:
        print(
            f"{d['class']:12s} "
            f"confidence={d['confidence']:.3f} "
            f"box=({d['x1']},{d['y1']},"
            f"{d['x2']},{d['y2']})"
        )

    all_results.append({
        "imgsz": imgsz,
        "detections": detections
    })

# Save detection comparison.
with open(
    OUTPUT / "detection_comparison.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "imgsz", "class", "confidence",
        "x1", "y1", "x2", "y2"
    ])

    for result in all_results:
        for d in result["detections"]:
            writer.writerow([
                result["imgsz"],
                d["class"],
                d["confidence"],
                d["x1"], d["y1"],
                d["x2"], d["y2"]
            ])

print("\nDiagnostic complete.")
print("Check the debug_north folder.")