import cv2
from ultralytics import YOLO

VIDEO = "data/videos/top_view.mp4"
FRAME_NUMBER = 90

# North approach crop in ORIGINAL 3840x2160 coordinates.
# Based on your north counting-line position.
X1, Y1 = 2400, 0
X2, Y2 = 3500, 900

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read frame")

north_crop = frame[Y1:Y2, X1:X2]

cv2.imwrite("north_crop_original.jpg", north_crop)

for imgsz in [960, 1280, 1600]:
    results = model.predict(
        north_crop,
        imgsz=imgsz,
        conf=0.10,
        classes=[2, 3, 5, 7],
        device="cpu",
        verbose=False
    )

    result = results[0]
    annotated = north_crop.copy()

    print(f"\nimgsz={imgsz}")

    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            print(
                f"class={model.names[cls]}, "
                f"confidence={conf:.3f}, "
                f"box={(x1, y1, x2, y2)}"
            )

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2),
                (0, 255, 0), 3
            )

            cv2.putText(
                annotated,
                f"{model.names[cls]} {conf:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

    cv2.imwrite(
        f"north_crop_{imgsz}.jpg",
        annotated
    )

print("\nCheck north_crop_960.jpg, north_crop_1280.jpg and north_crop_1600.jpg")