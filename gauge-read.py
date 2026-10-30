import cv2
import math
import numpy as np

image = cv2.imread("gauge4.jpg")

if image is None:
    raise FileNotFoundError("Could not load gauge.jpg")

# gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur
blurred = cv2.GaussianBlur(gray, (9, 9), 3)

# use separate image to draw lines on
line_image = image.copy()

# gray cale image
cv2.imshow("Gauge (Grayscale)", gray)


# edge detect
edges = cv2.Canny(gray, 50, 150)

cv2.imshow("Gauge (Edges)", edges)


# circle detection
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=100,
    param1=100,
    param2=40,
    minRadius=50,
    maxRadius=250,
)

dial_image = image.copy()

print("Original shape:", image.shape)
print("Grayscale shape:", gray.shape)

if circles is not None:
    circles = np.round(circles[0]).astype(int)

    print("Circles detected:", len(circles))

    # use the first detected circle as main dial
    x, y, radius = circles[0]
    cx, cy = x, y
    print("Centre:", (cx, cy))
    print("Radius:", radius)

    # draw detected circle and  centre
    cv2.circle(dial_image, (cx, cy), radius, (0, 255, 0), 2)

    cv2.circle(dial_image, (cx, cy), 3, (0, 0, 255), -1)

    # use separate image to draw lines on
    line_image = image.copy()

       # mask for inside dial
    mask = np.zeros(gray.shape, dtype=np.uint8)

    cv2.circle(
        mask,
        (cx, cy),
        int(radius * 0.80),
        255,
        -1
    )

    # Detect dark pixels
    dark_pixels = cv2.threshold(
        gray,
        100,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    # ignore pxl outside the dial
    dark_pixels = cv2.bitwise_and(
        dark_pixels,
        mask
    )

    cv2.imshow("dark pixels", dark_pixels)

    # Needle search settings
    start_radius = int(radius * 0.10)
    end_radius = int(radius * 0.75)

    best_angle = None
    best_score = -1

    # search diff directions
    for angle in range(360):

        radians = math.radians(angle)

        score = 0

        # scan from near the centre to the outer dial
        for r in range(start_radius, end_radius):

            px = int(cx + r * math.cos(radians))
            py = int(cy + r * math.sin(radians))

            if dark_pixels[py, px] > 0:
                score += 1

        # keep the direction with the highes score
        if score > best_score:
            best_score = score
            best_angle = angle

    print("best candidate angle:", best_angle)
    print("candidate score:", best_score)

    # draw the candidate direction
    if best_angle is not None:

        radians = math.radians(best_angle)

        tip_x = int(
            cx + radius * 0.75 * math.cos(radians)
        )

        tip_y = int(
            cy + radius * 0.75 * math.sin(radians)
        )

        cv2.line(
            dial_image,
            (cx, cy),
            (tip_x, tip_y),
            (0, 0, 255),
            2
        )

        cv2.circle(
            dial_image,
            (cx, cy),
            3,
            (255, 0, 0),
            -1
        )

else:
    print("No circles detected")

cv2.imshow("Detected Dial", dial_image)

cv2.waitKey(0)
cv2.destroyAllWindows()