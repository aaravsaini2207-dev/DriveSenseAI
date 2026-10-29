import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# Project paths
FRAMES_DIR = Path("data/evaluation/frames")
ANNOTATIONS_DIR = Path("data/evaluation/annotations")
PREVIEWS_DIR = Path("data/outputs/metrics/annotation_previews")

MODEL_PATH = Path("yolo11n.pt")
CONFIDENCE = 0.25

# Keep the same object classes as DriveSense AI
TARGET_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}


def main():
    if not FRAMES_DIR.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {FRAMES_DIR}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.resolve()}"
        )

    image_paths = sorted(FRAMES_DIR.glob("*.jpg"))

    if not image_paths:
        raise RuntimeError(f"No JPG images found in {FRAMES_DIR}")

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    dataset = {
        "description": (
            "YOLO-generated preliminary annotations; "
            "not independently verified ground truth."
        ),
        "model": str(MODEL_PATH),
        "confidence_threshold": CONFIDENCE,
        "images": [],
    }

    total_detections = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        # Detect objects in this image
        result = model.predict(
            source=image,
            imgsz=640,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False,
        )[0]

        image_annotations = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                label = result.names[class_id]

                if label not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                annotation = {
                    "class": label,
                    "class_id": class_id,
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox_xyxy": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],
                }

                image_annotations.append(annotation)

        # Store predictions as JSON
        dataset["images"].append({
            "file_name": image_path.name,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "annotation_count": len(image_annotations),
            "annotations": image_annotations,
        })

        # Generate visual preview
        preview = image.copy()

        for annotation in image_annotations:
            x1, y1, x2, y2 = map(
                int, annotation["bbox_xyxy"]
            )
            label = (
                f'{annotation["class"]} '
                f'{annotation["confidence"]:.2f}'
            )

            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                preview,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        preview_path = PREVIEWS_DIR / image_path.name

        if not cv2.imwrite(str(preview_path), preview):
            print(f"Warning: couldn't save preview {preview_path.name}")

        total_detections += len(image_annotations)

        print(
            f"{image_path.name}: "
            f"{len(image_annotations)} objects detected"
        )

    # Save the complete annotation dataset
    output_json = ANNOTATIONS_DIR / "auto_annotations.json"

    output_json.write_text(
        json.dumps(dataset, indent=2),
        encoding="utf-8",
    )

    print("\n========== AUTO-ANNOTATION COMPLETE ==========")
    print(f"Images processed: {len(dataset['images'])}")
    print(f"Total detections: {total_detections}")
    print(f"Annotations: {output_json.resolve()}")
    print(f"Visual previews: {PREVIEWS_DIR.resolve()}")
    print("\nNote: These are model predictions, not ground-truth labels.")


if __name__ == "__main__":
    main()