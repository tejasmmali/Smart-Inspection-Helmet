import cv2
import numpy as np
import math
import os


IMAGE_PATH = "gauge4.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load {IMAGE_PATH}. Check the image path."
    )

height, width = image.shape[:2]

print("Image dimensions:", width, "x", height)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)


min_radius = int(min(width, height) * 0.20)
max_radius = int(min(width, height) * 0.48)

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=min(width, height) * 0.25,
    param1=100,
    param2=45,
    minRadius=min_radius,
    maxRadius=max_radius
)

if circles is None:
    raise RuntimeError(
        "Gauge circle not detected. Try adjusting param2 or radius limits."
    )

circles = np.round(circles[0]).astype(int)

# select  the circle with largest radius
cx, cy, radius = max(circles, key=lambda circle: circle[2])

print("Gauge centre:", (cx, cy))
print("Gauge radius:", radius)

# create a mask inside the gauge
mask = np.zeros((height, width), dtype=np.uint8)

cv2.circle(
    mask,
    (cx, cy),
    int(radius * 0.82),
    255,
    -1
)

# Detect dark needle features inside the gauge
blackhat_kernel_size = max(5, int(radius * 0.12))

if blackhat_kernel_size % 2 == 0:
    blackhat_kernel_size += 1

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (blackhat_kernel_size, blackhat_kernel_size)
)

blackhat = cv2.morphologyEx(
    gray,
    cv2.MORPH_BLACKHAT,
    kernel
)

# threshold the dark features and restrict it to the dial
_, dark_features = cv2.threshold(
    blackhat,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

dark_features = cv2.bitwise_and(
    dark_features,
    mask
)

# remove small isolated regions
small_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

dark_features = cv2.morphologyEx(
    dark_features,
    cv2.MORPH_OPEN,
    small_kernel
)

# detect straight line candidates
edges = cv2.Canny(blurred, 50, 150)

edges = cv2.bitwise_and(edges, mask)

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=max(15, int(radius * 0.25)),
    minLineLength=max(15, int(radius * 0.25)),
    maxLineGap=max(5, int(radius * 0.08))
)

# find lines that is close to the centre
candidates = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.flatten()

        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(dx, dy)

        if length < radius * 0.25:
            continue

        denominator = dx * dx + dy * dy

        if denominator == 0:
            continue

        t = (
            (cx - x1) * dx +
            (cy - y1) * dy
        ) / denominator

        t = np.clip(t, 0, 1)

        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy

        distance = np.hypot(
            cx - nearest_x,
            cy - nearest_y
        )

        if distance > radius * 0.08:
            continue

        # restrict both endpoints to the detected dial
        d1 = np.hypot(x1 - cx, y1 - cy)
        d2 = np.hypot(x2 - cx, y2 - cy)

        if d1 > radius * 0.85 or d2 > radius * 0.85:
            continue

        # store candidate info for score
        angle = math.degrees(math.atan2(cy - y1, x1 - cx))

        candidates.append({
            "line": (x1, y1, x2, y2),
            "length": length,
            "distance": distance,
            "angle": angle
        })

# 
best_candidate = None
best_score = -1

for candidate in candidates:
    score = (
        candidate["length"] -
        candidate["distance"] * 2
    )

    if score > best_score:
        best_score = score
        best_candidate = candidate

#draw the center n circle of gauge
result = image.copy()

cv2.circle(
    result,
    (cx, cy),
    radius,
    (0, 255, 0),
    2
)

cv2.circle(
    result,
    (cx, cy),
    4,
    (0, 0, 255),
    -1
)

#show candidate lines n select the needle 
print("Candidate lines:", len(candidates))

if best_candidate is not None:
    x1, y1, x2, y2 = best_candidate["line"]

    cv2.line(
        result,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )

    print("Selected needle candidate:", best_candidate["line"])
    print("Estimated line angle:", round(best_candidate["angle"], 2))
else:
    print("No suitable needle candidate found.")

# Display the processing results
cv2.imshow("Original Gauge", image)
cv2.imshow("Detected Gauge and Needle", result)
cv2.imshow("Gauge Edges", edges)
cv2.imshow("Dark Features", dark_features)

cv2.waitKey(0)
cv2.destroyAllWindows()