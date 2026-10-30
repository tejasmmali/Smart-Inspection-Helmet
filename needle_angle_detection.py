import cv2
import numpy as np
import math
import argparse
from pathlib import Path


def detect_gauge_circle(gray):
    height, width = gray.shape
    size = min(height, width)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=size * 0.2,
        param1=100,
        param2=40,
        minRadius=int(size * 0.20),
        maxRadius=int(size * 0.49)
    )

    if circles is None:
        raise RuntimeError(
            "gauge circle not found, use clear image"
        )

    circles = np.round(circles[0]).astype(int)

    image_cx = width / 2
    image_cy = height / 2

    def circle_score(circle):
        x, y, radius = circle

        distance = math.hypot(
            x - image_cx,
            y - image_cy
        )

        return radius - distance * 0.35

    cx, cy, radius = max(
        circles,
        key=circle_score
    )

    return int(cx), int(cy), int(radius)


def detect_needle_angle(gray, cx, cy, radius):
    height, width = gray.shape

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    threshold_value = int(
        np.clip(np.mean(blurred) * 0.72, 55, 170)
    )

    dark_pixels = (
        blurred < threshold_value
    ).astype(np.uint8)

    start_radius = max(
        5,
        int(radius * 0.12)
    )

    end_radius = max(
        start_radius + 5,
        int(radius * 0.72)
    )

    sample_radii = np.arange(
        start_radius,
        end_radius
    )

    scores = []

    for angle in range(360):
        radians = math.radians(angle)

        xs = np.rint(
            cx + sample_radii * math.cos(radians)
        ).astype(int)

        ys = np.rint(
            cy - sample_radii * math.sin(radians)
        ).astype(int)

        valid = (
            (xs >= 0) &
            (xs < width) &
            (ys >= 0) &
            (ys < height)
        )

        xs = xs[valid]
        ys = ys[valid]

        if len(xs) == 0:
            scores.append(0)
            continue

        ray_pixels = dark_pixels[ys, xs]

        dark_count = int(np.sum(ray_pixels))

        longest_run = 0
        current_run = 0

        for pixel in ray_pixels:
            if pixel:
                current_run += 1
                longest_run = max(
                    longest_run,
                    current_run
                )
            else:
                current_run = 0

        score = dark_count + longest_run * 1.5

        scores.append(score)

    scores = np.asarray(scores, dtype=np.float32)

    best_angle = int(np.argmax(scores))
    best_score = float(scores[best_angle])

    if best_score <= 0:
        raise RuntimeError("needle not detected.")

    return best_angle, best_score, dark_pixels


def main():
    parser = argparse.ArgumentParser(
        description="Gauge needle angle detection"
    )

    parser.add_argument(
        "image",
        nargs="?",
        default="gauge.jpg",
        help="Path to the gauge image"
    )

    parser.add_argument(
        "--save",
        default="needle_angle_result.jpg",
        help="Output image path"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    height, width = image.shape[:2]

    print("Image dimensions:", width, "x", height)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # detect dial
    cx, cy, radius = detect_gauge_circle(gray)

    print("Gauge centre:", (cx, cy))
    print("Gauge radius:", radius)

    # detect needle direction
    angle, score, dark_pixels = detect_needle_angle(
        gray,
        cx,
        cy,
        radius
    )

    print("Estimated needle angle:", angle, "degrees")
    print("Detection score:", round(score, 2))

    print(
        "Angle convention: 0° right, 90° up, "
        "180° left, 270° down."
    )

    # draw results
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
        (255, 0, 0),
        -1
    )

    radians = math.radians(angle)

    tip_x = int(
        cx + radius * 0.72 * math.cos(radians)
    )

    tip_y = int(
        cy - radius * 0.72 * math.sin(radians)
    )

    cv2.line(
        result,
        (cx, cy),
        (tip_x, tip_y),
        (0, 0, 255),
        3
    )

    cv2.putText(
        result,
        f"Angle: {angle} deg",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    # Display the results
    cv2.imshow(
        "Needle Angle Detection",
        result
    )

    cv2.imshow(
        "Dark Pixel Map",
        dark_pixels * 255
    )

    cv2.imwrite(
        args.save,
        result
    )

    print("Saved result:", args.save)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()