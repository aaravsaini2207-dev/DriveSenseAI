import json
from collections import Counter
from pathlib import Path

path = Path("data/evaluation/annotations/auto_annotations.json")
data = json.loads(path.read_text(encoding="utf-8"))

class_counts = Counter()
detections_per_image = []

for image in data["images"]:
    detections = image["annotations"]
    detections_per_image.append(len(detections))

    for detection in detections:
        class_counts[detection["class"]] += 1

print("\n========== DETECTION SUMMARY ==========")
print(f"Images processed: {len(data['images'])}")
print(f"Total detections: {sum(class_counts.values())}")
print(f"Average detections/image: {sum(detections_per_image) / max(len(detections_per_image), 1):.2f}")

print("\nDetections by class:")
for label, count in class_counts.most_common():
    print(f"{label:12}: {count}")

print("\nImages with most detections:")
for image in sorted(
    data["images"],
    key=lambda item: item["annotation_count"],
    reverse=True,
)[:5]:
    print(f"{image['file_name']:25} {image['annotation_count']}")

print("\nNote: These are YOLO predictions, not ground-truth accuracy measurements.")