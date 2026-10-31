import cv2
import numpy as np
import math
import argparse
import time
import sys
from collections import deque
from monitoring import publish_frame, start_monitoring_server, save_reading

# gauge settings
MIN_VALUE = 0.0
MAX_VALUE = 10.0
UNIT = "bar"

MIN_ANGLE = 225.0
MAX_ANGLE = 315.0

# camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MAX_SIZE = 1280
WARMUP_FRAMES = 10  # allow the camera exposure to settle

# detection and tracking settings
CIRCLE_INTERVAL = 15          
CIRCLE_CONFIRM_FRAMES = 3     
CIRCLE_MAX_MISSES = 6         
CIRCLE_SMOOTHING = 0.25       

SMOOTHING_FRAMES = 5
SCORE_MEMORY = 0.65           
NEEDLE_MIN_PROMINENCE = 1.35  
SWEEP_TOLERANCE = 8.0         

ANGLE_JUMP_DEG = 20.0        
ANGLE_CONFIRM_FRAMES = 4
ANGLE_ACQUIRE_FRAMES = 3
HOLD_FRAMES = 15             
READING_DEADBAND = 0.005     


def circular_diff(a, b):
    """Signed smallest difference a - b in degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


def valid_angle_mask():
    """True for needle angles inside the calibrated dial sweep (+ tolerance)."""
    total_sweep = (MIN_ANGLE - MAX_ANGLE) % 360
    sweeps = (MIN_ANGLE - np.arange(360)) % 360
    return (
        (sweeps <= total_sweep + SWEEP_TOLERANCE) |
        (sweeps >= 360 - SWEEP_TOLERANCE)
    )


VALID_ANGLES = valid_angle_mask()

def detect_gauge_circle(gray, prev_circle=None):

    height, width = gray.shape
    size = min(height, width)

    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )
    edges = cv2.Canny(blurred, 60, 140)

    min_radius = int(size * 0.22)
    max_radius = int(size * 0.48)

    # narrow the search after the gauge has been found
    if prev_circle is not None:
        prev_r = prev_circle[2]
        min_radius = max(min_radius, int(prev_r * 0.8))
        max_radius = min(max_radius, int(prev_r * 1.2))

    candidates = []

    for dp, param1, param2 in ((1.2, 100, 40), (1.2, 90, 32)):

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=size * 0.18,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is not None:
            candidates = np.round(circles[0]).astype(int)
            break

    if len(candidates) == 0:
        raise RuntimeError("Gauge not detected")

    if prev_circle is not None:
        target_x, target_y, target_radius = prev_circle
    else:
        target_x = width / 2
        target_y = height * 0.44
        target_radius = size * 0.36

    ring_angles = np.linspace(0, 2 * math.pi, 90, endpoint=False)
    ring = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    def circle_score(circle):

        x, y, radius = circle

        centre_error = math.hypot(
            x - target_x,
            y - target_y
        ) / size

        radius_error = abs(
            radius - target_radius
        ) / size

        # prefer circles that match the dial outer edge
        xs = np.rint(x + radius * np.cos(ring_angles)).astype(int)
        ys = np.rint(y + radius * np.sin(ring_angles)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

        edge_strength = 0.0
        if np.any(valid):
            edge_strength = float(np.mean(ring[ys[valid], xs[valid]])) / 255.0

        return centre_error + radius_error * 0.5 - edge_strength * 0.4

    cx, cy, radius = min(
        candidates,
        key=circle_score
    )

    return int(cx), int(cy), int(radius)


def needle_score_curve(gray, cx, cy, radius):
    """Calculate how strongly the needle is pointing in each direction."""

    height, width = gray.shape

    blurred = cv2.GaussianBlur(
        gray, (3, 3), 0
    )

    # use the dial face to set the threshold. This reduces the effect of
    # changes in the background.
    face = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(face, (cx, cy), int(radius * 0.9), 1, -1)
    face_pixels = blurred[face > 0]
    face_mean = float(np.mean(face_pixels)) if face_pixels.size else float(np.mean(blurred))

    threshold_value = int(
        np.clip(
            face_mean * 0.72,
            55,
            170
        )
    )

    dark = (
        blurred < threshold_value
    ).astype(np.float32)
    dark = cv2.medianBlur((dark * 255).astype(np.uint8), 3) / 255.0

    # skip the centre hub and the markings near the edge
    start_radius = int(radius * 0.12)
    end_radius = int(radius * 0.55)

    sample_radii = np.arange(start_radius, end_radius, dtype=np.float32)

    half_width = max(
        2,
        int(radius * 0.035)
    )

    offsets = np.arange(-half_width, half_width + 1, dtype=np.float32)

    radians = np.deg2rad(np.arange(360, dtype=np.float32))
    cos_a = np.cos(radians)[:, None, None]
    sin_a = np.sin(radians)[:, None, None]
    r = sample_radii[None, :, None]
    o = offsets[None, None, :]

    xs = np.rint(cx + r * cos_a - o * sin_a).astype(np.int32)
    ys = np.rint(cy - r * sin_a - o * cos_a).astype(np.int32)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    pixels = dark[
        np.clip(ys, 0, height - 1),
        np.clip(xs, 0, width - 1)
    ] * valid

    row_scores = pixels.mean(axis=2)          # (360, radii)
    occupied = row_scores > 0.20

    # Longest continuous dark run along each spoke
    counts = np.cumsum(occupied, axis=1)
    resets = np.maximum.accumulate(np.where(~occupied, counts, 0), axis=1)
    longest_run = (counts - resets).max(axis=1)

    # Give more weight to dark lines that continue towards the outer dial
    outer = row_scores[:, int(row_scores.shape[1] * 0.55):]
    tip_weight = outer.mean(axis=1) if outer.size else 0.0

    return (
        row_scores.sum(axis=1)
        + longest_run * 1.5
        + tip_weight * 8.0
    ).astype(np.float32)


def detect_needle_angle(score_curve):
    """Find the most likely needle angle from the score curve."""

    scores = np.where(VALID_ANGLES, score_curve, 0.0)

    # smooth the score includin across the 0 to 360-degree boundary
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.concatenate([scores[-2:], scores, scores[:2]])
    smooth = np.convolve(padded, kernel, mode="valid")

    best_angle = int(np.argmax(smooth))
    best_score = float(smooth[best_angle])
    median_score = float(np.median(smooth[VALID_ANGLES]))

    if best_score <= 0:
        raise RuntimeError("Needle not detected")

    if median_score > 0 and best_score < median_score * NEEDLE_MIN_PROMINENCE:
        raise RuntimeError("Needle not detected")

    # refine the angle for a little more precision
    y0 = smooth[(best_angle - 1) % 360]
    y1 = smooth[best_angle]
    y2 = smooth[(best_angle + 1) % 360]
    denom = y0 - 2 * y1 + y2
    offset = 0.0
    if abs(denom) > 1e-6:
        offset = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))

    confidence = 1.0 - (median_score / best_score) if best_score > 0 else 0.0

    return (best_angle + offset) % 360, confidence


def calculate_gauge_reading(angle):

    total_sweep = (
        MIN_ANGLE - MAX_ANGLE
    ) % 360

    needle_sweep = (
        MIN_ANGLE - angle
    ) % 360

    if total_sweep == 0:
        raise ValueError("Invalid calibration")

    # handle small errors around the minimum position
    tolerance = SWEEP_TOLERANCE

    if needle_sweep >= 360 - tolerance:
        needle_sweep = 0

    elif needle_sweep > total_sweep:

        if needle_sweep <= total_sweep + tolerance:
            needle_sweep = total_sweep
        else:
            return None

    percentage = needle_sweep / total_sweep

    percentage = np.clip(
        percentage, 0, 1
    )

    return (
        MIN_VALUE
        + percentage * (MAX_VALUE - MIN_VALUE)
    )


class CircleTracker:
    """Keep track of the same gauge across frames."""

    def __init__(self):
        self.circle = None         
        self.misses = 0
        self.candidate = None
        self.candidate_count = 0

    @property
    def locked(self):
        return self.circle is not None

    def _same(self, a, b):
        tol = max(a[2], b[2]) * 0.15
        return (
            math.hypot(a[0] - b[0], a[1] - b[1]) < tol and
            abs(a[2] - b[2]) < tol
        )

    def update(self, measured):
        """Update the gauge position and report when the lock changes."""

        if measured is None:
            self.candidate = None
            self.candidate_count = 0
            if self.circle is not None:
                self.misses += 1
                if self.misses > CIRCLE_MAX_MISSES:
                    self.circle = None
                    return True
            return False

        measured = tuple(float(v) for v in measured)

        if self.circle is not None and self._same(measured, self.circle):
            a = CIRCLE_SMOOTHING
            self.circle = tuple(
                (1 - a) * old + a * new
                for old, new in zip(self.circle, measured)
            )
            self.misses = 0
            self.candidate = None
            self.candidate_count = 0
            return False

       
        if self.candidate is not None and self._same(measured, self.candidate):
            self.candidate_count += 1
            self.candidate = measured
        else:
            self.candidate = measured
            self.candidate_count = 1

        if self.candidate_count >= CIRCLE_CONFIRM_FRAMES:
            self.circle = self.candidate
            self.misses = 0
            self.candidate = None
            self.candidate_count = 0
            return True

        if self.circle is not None:
            self.misses += 1
            if self.misses > CIRCLE_MAX_MISSES:
                self.circle = None
                return True

        return False

    def as_int(self):
        cx, cy, r = self.circle
        return int(round(cx)), int(round(cy)), int(round(r))


class AngleTracker:
    """Smooth the needle movement and ignore sudden false detections."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.angle = None
        self.pending = None
        self.pending_count = 0
        self.acquire = deque(maxlen=ANGLE_ACQUIRE_FRAMES)
        self.misses = 0

    def update(self, measured):

        self.misses = 0

        if self.angle is None:
            self.acquire.append(measured)
            if len(self.acquire) == ANGLE_ACQUIRE_FRAMES:
                ref = self.acquire[-1]
                if all(abs(circular_diff(a, ref)) < 10 for a in self.acquire):
                    self.angle = ref
            return self.angle

        delta = circular_diff(measured, self.angle)

        if abs(delta) <= ANGLE_JUMP_DEG:
            # smooth small changes while still following real needle movement
            alpha = 0.15 if abs(delta) < 1.5 else 0.45
            self.angle = (self.angle + alpha * delta) % 360
            self.pending = None
            self.pending_count = 0
            return self.angle

        # check large changes over several frames before accepting them
        if self.pending is not None and abs(circular_diff(measured, self.pending)) < 10:
            self.pending_count += 1
            self.pending = measured
        else:
            self.pending = measured
            self.pending_count = 1

        if self.pending_count >= ANGLE_CONFIRM_FRAMES:
            self.angle = self.pending
            self.pending = None
            self.pending_count = 0

        return self.angle

    def miss(self):
        self.misses += 1
        if self.misses > HOLD_FRAMES:
            self.reset()


def open_camera(camera_index):

    # try the camera backend that normally works best on this operating system
    if sys.platform.startswith("linux"):
        backends = [cv2.CAP_V4L2, None]
    elif sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, None]
    else:
        backends = [None]

    cap = None

    for backend in backends:

        if backend is None:
            trial = cv2.VideoCapture(camera_index)
        else:
            trial = cv2.VideoCapture(
                camera_index,
                backend
            )

        if trial.isOpened():
            cap = trial
            break

        trial.release()

    if cap is None or not cap.isOpened():

        raise RuntimeError(
            f"Cannot open camera {camera_index}"
        )

    # Request MJPG format
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30
    )

    # keep the camera buffer small so the displayed video stays current
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    return cap


def put_text(display, text, y, color, scale=0.7):
    cv2.putText(display, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(display, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)



def validate_gauge_face(gray, cx, cy, radius):
    height, width = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 120)
    angles = np.deg2rad(np.arange(360))
    rim_hits = np.zeros(360, dtype=bool)

    for offset in range(-4, 5):
        r = radius * 0.94 + offset
        xs = np.rint(cx + r * np.cos(angles)).astype(int)
        ys = np.rint(cy + r * np.sin(angles)).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        hits = np.zeros(360, dtype=bool)
        idx = np.where(valid)[0]
        hits[idx] = edges[ys[idx], xs[idx]] > 0
        rim_hits |= hits

    rim_coverage = float(np.mean(rim_hits))
    rim_sectors = sum(
        np.mean(rim_hits[i:i + 45]) > 0.18 for i in range(0, 360, 45)
    )

    face_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(face_mask, (cx, cy), int(radius * 0.88), 255, -1)
    cv2.circle(face_mask, (cx, cy), int(radius * 0.55), 0, -1)
    face_edges = cv2.bitwise_and(edges, edges, mask=face_mask)

    face_sectors = 0
    for i in range(8):
        sector = np.zeros_like(gray, dtype=np.uint8)
        cv2.ellipse(
            sector, (cx, cy),
            (int(radius * 0.88), int(radius * 0.88)),
            0, i * 45, (i + 1) * 45, 255, -1
        )
        inner = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(inner, (cx, cy), int(radius * 0.55), 255, -1)
        sector = cv2.bitwise_and(sector, cv2.bitwise_not(inner))
        count = cv2.countNonZero(cv2.bitwise_and(face_edges, face_edges, mask=sector))
        if count > radius * 0.20:
            face_sectors += 1

    return rim_coverage >= 0.30 and rim_sectors >= 6 and face_sectors >= 4


def main():

    parser = argparse.ArgumentParser(
        description="Real-time Gauge Monitoring"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Webcam index (default: 1)"
    )

    args = parser.parse_args()

    start_monitoring_server()
    cap = open_camera(args.camera)

    print("\nSMART INSPECTION HELMET")
    print("Camera:", args.camera)
    print("Resolution:", CAMERA_WIDTH, "x", CAMERA_HEIGHT)
    print("Press Q to exit")

    circle_tracker = CircleTracker()
    angle_tracker = AngleTracker()
    score_memory = None

    frame_count = 0

    readings = deque(
        maxlen=SMOOTHING_FRAMES
    )
    shown_reading = None
    value_range = MAX_VALUE - MIN_VALUE
    valid_frame_count = 0
    stable_pressures = deque(maxlen=4)

    fps = 0
    previous_time = time.perf_counter()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Camera frame unavailable")
                break

            frame_count += 1

            if frame_count <= WARMUP_FRAMES:
                continue

            # resize large frames
            height, width = frame.shape[:2]

            if max(height, width) > MAX_SIZE:

                scale = MAX_SIZE / max(height, width)

                frame = cv2.resize(
                    frame,
                    (
                        int(width * scale),
                        int(height * scale)
                    ),
                    interpolation=cv2.INTER_AREA
                )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # search every frame until a gauge is found, then check periodically
            if (
                not circle_tracker.locked or
                frame_count % CIRCLE_INTERVAL == 0
            ):

                prev = (
                    circle_tracker.as_int()
                    if circle_tracker.locked else None
                )

                try:
                    measured = detect_gauge_circle(gray, prev_circle=prev)
                except RuntimeError:
                    measured = None

                if circle_tracker.update(measured):
                    score_memory = None
                    angle_tracker.reset()
                    readings.clear()
                    shown_reading = None

            display = frame.copy()
            angle = None

            if circle_tracker.locked:

                cx, cy, radius = circle_tracker.as_int()

                # combine information from recent frames to reduce flicker
                curve = needle_score_curve(gray, cx, cy, radius)
                peak = float(curve.max())
                if peak > 0:
                    curve = curve / peak
                score_memory = (
                    curve if score_memory is None
                    else SCORE_MEMORY * score_memory + (1 - SCORE_MEMORY) * curve
                )

                status = "LOCKED"
                confidence = 0.0

                try:
                    measured_angle, confidence = detect_needle_angle(score_memory)
                    angle = angle_tracker.update(measured_angle)
                    if angle is None:
                        status = "ACQUIRING"
                    elif angle_tracker.pending is not None:
                        status = "CONFIRMING MOVE"
                except (RuntimeError, ValueError):
                    angle_tracker.miss()
                    angle = angle_tracker.angle
                    status = "HOLD" if angle is not None else "NO NEEDLE"

                # draw the detected gauge
                cv2.circle(
                    display,
                    (cx, cy),
                    radius,
                    (0, 255, 0),
                    2
                )

                # mark the centre of the gauge
                cv2.circle(
                    display,
                    (cx, cy),
                    4,
                    (255, 0, 0),
                    -1
                )

                if angle is not None:

                    reading = calculate_gauge_reading(angle)

                    if reading is not None:

                        readings.append(reading)

                        # smooth the displayed pressure
                        smooth_reading = float(
                            np.median(list(readings))
                        )

                        if (
                            shown_reading is None or
                            abs(smooth_reading - shown_reading)
                            > value_range * READING_DEADBAND
                        ):
                            shown_reading = smooth_reading

                        reading_text = f"{shown_reading:.2f} {UNIT}"

                    else:

                        readings.clear()
                        shown_reading = None
                        reading_text = "Out of range"

                    # draw the needle only after its angle has been accepted
                    radians = math.radians(angle)

                    tip_x = int(
                        cx + radius * 0.55 * math.cos(radians)
                    )

                    tip_y = int(
                        cy - radius * 0.55 * math.sin(radians)
                    )

                    cv2.line(
                        display,
                        (cx, cy),
                        (tip_x, tip_y),
                        (0, 0, 255),
                        3
                    )

                    put_text(display, f"Angle: {angle:.1f} deg", 30, (0, 0, 255))
                    put_text(display, f"Pressure: {reading_text}", 60, (0, 0, 255))

                else:
                    put_text(display, "Reading needle...", 60, (0, 165, 255))

                status_color = (0, 200, 0) if status == "LOCKED" else (0, 165, 255)
                put_text(
                    display,
                    f"{status}  conf {confidence:.2f}",
                    125,
                    status_color,
                    0.6
                )

            else:

                readings.clear()
                shown_reading = None

                searching = "Searching for gauge..."
                if circle_tracker.candidate_count:
                    searching += (
                        f" ({circle_tracker.candidate_count}/"
                        f"{CIRCLE_CONFIRM_FRAMES})"
                    )
                put_text(display, searching, 40, (0, 165, 255), 0.8)

            # update the FPS counter
            current_time = time.perf_counter()

            elapsed = current_time - previous_time

            if elapsed > 0:
                instant_fps = 1.0 / elapsed

                if fps == 0:
                    fps = instant_fps
                else:
                    fps = 0.8 * fps + 0.2 * instant_fps

            previous_time = current_time

            put_text(display, f"FPS: {fps:.1f}", 95, (255, 0, 0))

            if circle_tracker.locked and angle is not None and shown_reading is not None:
                cx, cy, radius = circle_tracker.as_int()
                face_ok = validate_gauge_face(gray, cx, cy, radius)
                stable_pressures.append(shown_reading)

                pressure_stable = (
                    len(stable_pressures) == 4 and
                    max(stable_pressures) - min(stable_pressures) <= 0.35
                )

                if face_ok and pressure_stable:
                    valid_frame_count += 1
                else:
                    valid_frame_count = 0

                if valid_frame_count >= 4:
                    save_reading(shown_reading, angle, display, UNIT)
            else:
                valid_frame_count = 0
                stable_pressures.clear()

            publish_frame(display)

            cv2.imshow(
                "Smart Inspection Helmet - Live Gauge",
                display
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()

        print("\nWebcam stopped.")
        print("Application closed.")


if __name__ == "__main__":
    main()
