import math
from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = Path("data/outputs/metrics/annotation_previews")
OUTPUT_PATH = Path("data/outputs/metrics/contact_sheet.jpg")

THUMB_WIDTH = 320
THUMB_HEIGHT = 200
COLS = 4

images = []

for path in sorted(INPUT_DIR.glob("*.jpg")):
    image = cv2.imread(str(path))

    if image is None:
        continue

    image = cv2.resize(image, (THUMB_WIDTH, THUMB_HEIGHT))
    cv2.putText(
        image,
        path.name,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    images.append(image)

if not images:
    raise RuntimeError(f"No preview images found in {INPUT_DIR}")

rows = math.ceil(len(images) / COLS)
sheet = np.zeros(
    (rows * THUMB_HEIGHT, COLS * THUMB_WIDTH, 3),
    dtype=np.uint8,
)

for i, image in enumerate(images):
    row, col = divmod(i, COLS)
    y = row * THUMB_HEIGHT
    x = col * THUMB_WIDTH
    sheet[y:y + THUMB_HEIGHT, x:x + THUMB_WIDTH] = image

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_PATH), sheet)

print(f"Contact sheet saved: {OUTPUT_PATH.resolve()}")
print(f"Images included: {len(images)}")