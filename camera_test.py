import cv2
import time

CAMERA_INDEX = 1  #cam change

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open Logitech C270")

# resolution and fps settings
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Resolution:",
      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      "x",
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

print("Reported FPS:", cap.get(cv2.CAP_PROP_FPS))

print(
    "FourCC:",
    "".join(
        chr((int(cap.get(cv2.CAP_PROP_FOURCC)) >> (8 * i)) & 0xFF)
        for i in range(4)
    )
)

frames = 0
start = time.perf_counter()
last_report = start

print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame capture failed")
        break

    frames += 1
    now = time.perf_counter()

    if now - last_report >= 5:
        fps = frames / (now - start)
        print(f"Measured FPS: {fps:.2f}")
        last_report = now

    cv2.imshow("Logitech C270 - FPS Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()