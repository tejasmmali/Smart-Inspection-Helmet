
import cv2
import numpy as np
import math
import argparse
from pathlib import Path

#calibration
MIN_VALUE = 0.0
MAX_VALUE = 10.0
UNIT = "bar"

MIN_ANGLE = 225.0
MAX_ANGLE = 315.0

# Image settings
MAX_SIZE = 1280


def resize_image(image):

    height, width = image.shape[:2]

    if max(height, width) > MAX_SIZE:

        scale = MAX_SIZE / max(height, width)

        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA
        )

    return image


def detect_gauge_circle(gray):

    height, width = gray.shape
    size = min(height, width)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=size * 0.18,
        param1=100,
        param2=40,
        minRadius=int(size * 0.22),
        maxRadius=int(size * 0.48)
    )

    if circles is None:
        raise RuntimeError("Gauge circle not detected")

    circles = np.round(circles[0]).astype(int)

    target_x = width / 2
    target_y = height * 0.44
    target_radius = size * 0.36

    def circle_score(circle):

        x, y, radius = circle

        centre_error = math.hypot(
            x - target_x,
            y - target_y
        ) / size

        radius_error = abs(
            radius - target_radius
        ) / size

        return centre_error + radius_error * 0.5

    cx, cy, radius = min(
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

    # separate dark needle features
    threshold_value = int(
        np.clip(
            np.mean(blurred) * 0.72,
            55,
            170
        )
    )

    

    dark = (
        blurred < threshold_value
    ).astype(np.uint8)


    start_radius = int(radius * 0.12)
    end_radius = int(radius * 0.55)

    sample_radii = np.arange(
        start_radius,
        end_radius
    )

    half_width = max(
        2,
        int(radius * 0.035)
    )

    offsets = np.arange(
        -half_width,
        half_width + 1
    )

    scores = []

    for angle in range(360):

        radians = math.radians(angle)

        cos_a = math.cos(radians)
        sin_a = math.sin(radians)

        xs = (
            cx
            + sample_radii[:, None] * cos_a
            - offsets[None, :] * sin_a
        )

        ys = (
            cy
            - sample_radii[:, None] * sin_a
            - offsets[None, :] * cos_a
        )

        xs = np.rint(xs).astype(int)
        ys = np.rint(ys).astype(int)

        valid = (
            (xs >= 0) &
            (xs < width) &
            (ys >= 0) &
            (ys < height)
        )

        pixels = np.zeros_like(
            xs,
            dtype=np.uint8
        )

        pixels[valid] = dark[
            ys[valid],
            xs[valid]
        ]

        # thickness and continuity meter
        row_scores = np.mean(
            pixels,
            axis=1
        )

        occupied = row_scores > 0.20

        longest_run = 0
        current_run = 0

        for value in occupied:

            if value:
                current_run += 1

                longest_run = max(
                    longest_run,
                    current_run
                )

            else:
                current_run = 0

        score = (
            np.sum(row_scores)
            + longest_run * 1.5
        )

        scores.append(score)

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    # choose strongest pointer direction
    best_angle = int(np.argmax(scores))
    best_score = float(scores[best_angle])

    if best_score <= 0:
        raise RuntimeError("needle not detected")

    return best_angle, best_score, dark


def calculate_gauge_reading(angle):

    total_sweep = (
        MIN_ANGLE - MAX_ANGLE
    ) % 360

    needle_sweep = (
        MIN_ANGLE - angle
    ) % 360

    if total_sweep == 0:
        raise ValueError("Invalid calibration angles")


    tolerance = 8.0

    if needle_sweep >= 360 - tolerance:
        needle_sweep = 0

   
    elif needle_sweep > total_sweep:

        if needle_sweep <= total_sweep + tolerance:
            needle_sweep = total_sweep

        else:
            return None

    percentage = needle_sweep / total_sweep

    percentage = np.clip(
        percentage,
        0,
        1
    )

    reading = (
        MIN_VALUE
        + percentage * (MAX_VALUE - MIN_VALUE)
    )

    return reading


def main():

    parser = argparse.ArgumentParser(
        description="Gauge needle n read detection"
    )

    parser.add_argument(
        "image",
        nargs="?",
        default="gauge2.jpg"
    )

    parser.add_argument(
        "--save",
        default="needle_angle_result.jpg"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    # Resize large images
    image = resize_image(image)

    height, width = image.shape[:2]

    print("\nSMART INSPECTION HELMET")
 
    print(
        "Image dimensions:",
        width,
        "x",
        height
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # detect gauge circle
    cx, cy, radius = detect_gauge_circle(gray)

    print("gauge centre:", (cx, cy))
    print("gauge radius:", radius)

    # detect needle
    angle, score, dark = detect_needle_angle(
        gray,
        cx,
        cy,
        radius
    )

    print(
        "needle angle:",
        round(angle, 2),
        "degrees"
    )

    print(
        "Detection score:",
        round(score, 2)
    )

    # calculate pressure
    reading = calculate_gauge_reading(angle)

    if reading is not None:

        reading = round(reading, 2)

        print("\nGAUGE READING")
   

        print(
            "Pressure:",
            reading,
            UNIT
        )

        reading_text = f"{reading} {UNIT}"

    else:

        print("needle outside calibrated range")

        reading_text = "Out of range"

    # draw result
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
        cx + radius * 0.55 * math.cos(radians)
    )

    tip_y = int(
        cy - radius * 0.55 * math.sin(radians)
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
        2
    )

    cv2.putText(
        result,
        f"Reading: {reading_text}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # display result
    cv2.imshow(
        "needle detection",
        result
    )

    cv2.imshow(
        "dark pxl map",
        dark * 255
    )

    # Save output
    cv2.imwrite(
        args.save,
        result
    )

    print("\nSaved:", args.save)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
