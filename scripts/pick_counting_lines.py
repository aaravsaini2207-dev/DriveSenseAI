import cv2

VIDEO = "data/videos/top_view.mp4"
FRAME_NUMBER = 90
WIDTH, HEIGHT = 1280, 720

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)

ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read overhead video frame.")

frame = cv2.resize(frame, (WIDTH, HEIGHT))

approaches = ["north", "south", "east", "west"]
lines = {}
points = []
current = 0

def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))

cv2.namedWindow("Counting Line Coordinate Picker")
cv2.setMouseCallback("Counting Line Coordinate Picker", mouse_callback)

while current < len(approaches):
    display = frame.copy()
    name = approaches[current]

    for p in points:
        cv2.circle(display, p, 6, (0, 255, 255), -1)

    if len(points) == 2:
        cv2.line(display, points[0], points[1], (0, 255, 0), 3)

    cv2.putText(
        display,
        f"Select {name.upper()} line: click 2 endpoints",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        "ENTER = confirm | R = retry | ESC = quit",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Counting Line Coordinate Picker", display)
    key = cv2.waitKey(20) & 0xFF

    if key == 27:
        break

    elif key == ord("r"):
        points = []

    elif key == 13 and len(points) == 2:
        lines[name] = {
            "start": points[0],
            "end": points[1]
        }
        print(f"{name}: {lines[name]}")
        points = []
        current += 1

cv2.destroyAllWindows()

print("\nFINAL COUNTING LINES:")
print("COUNTING_LINES = {")
for name, coords in lines.items():
    print(f'    "{name}": {coords},')
print("}")
