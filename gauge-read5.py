
import cv2
import numpy as np


# load the gauge image
image = cv2.imread("gauge.jpg")

if image is None:
    raise FileNotFoundError("not find gauge.jpg")

print("img dimensions:", image.shape)

# gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#reduce noise 
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# detect circle in gauge

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=100,
    param1=100,
    param2=40,
    minRadius=50,
    maxRadius=100
)


if circles is None:
    print("Gauge circle not detected")
    exit()
edges = cv2.Canny(blurred, 50, 150)
circles = np.round(circles[0]).astype(int)

cx, cy, radius = circles[0]
# detect candidate
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=25,
    minLineLength=20,
    maxLineGap=10
)

result = image.copy()

# detected lines in blue
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.flatten()

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            1
        )

# mark the center of gauge
cv2.circle(result, (cx, cy), 4, (0, 0, 255), -1)

cv2.imshow("Candidate Lines", result)

print("gauge centre:", (cx, cy))
print("gauge radius:", radius)

# draw the detected circle and centre
result = image.copy()

cv2.circle(result, (cx, cy), radius, (0, 255, 0), 2)
cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

cv2.imshow("detected gauge", result)

#edge detection
edges = cv2.Canny(blurred, 50, 150)

# highlight dark detail in gauge face
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (15, 15)
)

blackhat = cv2.morphologyEx(
    gray,
    cv2.MORPH_BLACKHAT,
    kernel
)

# separate prominent dark detail
_, dark_details = cv2.threshold(
    blackhat,
    25,
    255,
    cv2.THRESH_BINARY
)

#  the results
cv2.imshow("Blackhat", blackhat)
cv2.imshow("Dark Details", dark_details)


cv2.imshow("og gauage", image)
cv2.imshow("grayscale", gray)
cv2.imshow("gauge edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

