

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, send_from_directory, render_template_string


# Files are stored beside this Python file.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "snapshots"
READINGS_FILE = DATA_DIR / "readings.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

if not READINGS_FILE.exists():
    READINGS_FILE.write_text("[]", encoding="utf-8")

app = Flask(__name__)

_file_lock = threading.Lock()
_save_lock = threading.Lock()
_frame_lock = threading.Lock()
_last_save_time = 0.0
_jpeg = None
_frame_time = 0.0

# Save at most one reading per second. Change this if needed.
SAVE_INTERVAL_SECONDS = 1.0


def publish_frame(frame):
    """Keep the latest detection JPEG. At most 10 per second."""
    global _jpeg, _frame_time
    if frame is None:
        return
    now = time.time()
    if now - _frame_time < 0.1:
        return
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return
    with _frame_lock:
        _jpeg = buf.tobytes()
        _frame_time = now


def save_reading(pressure, angle, annotated_frame, unit="bar"):
  
    global _last_save_time

    if annotated_frame is None:
        return None

    with _save_lock:
        now = time.time()
        if now - _last_save_time < SAVE_INTERVAL_SECONDS:
            return None

        stamp = datetime.now()
        image_name = stamp.strftime("gauge_%Y%m%d_%H%M%S_%f.jpg")
        image_path = IMAGE_DIR / image_name

        if not cv2.imwrite(str(image_path), annotated_frame):
            print("Monitoring: could not save gauge screenshot.")
            return None

        record = {
            "timestamp": stamp.isoformat(timespec="seconds"),
            "pressure": round(float(pressure), 2),
            "angle": round(float(angle), 2),
            "unit": str(unit),
            "image": image_name,
        }

        try:
            with _file_lock:
                try:
                    readings = json.loads(READINGS_FILE.read_text(encoding="utf-8"))
                    if not isinstance(readings, list):
                        readings = []
                except (json.JSONDecodeError, OSError):
                    readings = []

                readings.append(record)

                # Write to a temporary file, then replace the old file.
                temp_file = READINGS_FILE.with_suffix(".tmp")
                temp_file.write_text(
                    json.dumps(readings, indent=2),
                    encoding="utf-8",
                )
                os.replace(temp_file, READINGS_FILE)

            _last_save_time = now
            print(
                f"Monitoring: saved {record['pressure']} {record['unit']} "
                f"with image {image_name}"
            )
            return record

        except Exception:
            # Remove the image if its matching record could not be written.
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise


@app.get("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.get("/api/readings")
def api_readings():
    with _file_lock:
        try:
            readings = json.loads(READINGS_FILE.read_text(encoding="utf-8"))
            if not isinstance(readings, list):
                readings = []
        except (json.JSONDecodeError, OSError):
            readings = []

    # Send the most recent 500 readings to keep the dashboard responsive.
    return jsonify(readings[-500:])


@app.get("/api/latest")
def api_latest():
    with _file_lock:
        try:
            readings = json.loads(READINGS_FILE.read_text(encoding="utf-8"))
            if not isinstance(readings, list):
                readings = []
        except (json.JSONDecodeError, OSError):
            readings = []

    return jsonify(readings[-1] if readings else {})


@app.get("/detection")
def detection():
    def frames():
        while True:
            with _frame_lock:
                jpeg = _jpeg
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.1)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshots/<path:filename>")
def snapshot(filename):
    # send_from_directory prevents paths outside IMAGE_DIR from being served.
    return send_from_directory(str(IMAGE_DIR), filename)


def start_monitoring_server(host="0.0.0.0", port=5000):
    """Start Flask in a background thread; call once from your main program."""
    thread = threading.Thread(
        target=lambda: app.run(
            host=host,
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True,
        ),
        daemon=True,
        name="inspection-monitoring-server",
    )
    thread.start()
    print(f"Monitoring dashboard: http://127.0.0.1:{port}")
    return thread


DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Inspection Helmet</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root {
    color-scheme: dark;
    --bg: #111318;
    --panel: #1b2029;
    --border: #303744;
    --muted: #9aa4b2;
    --text: #edf1f5;
    --accent: #55aaff;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: Arial, sans-serif;
}
main { max-width: 1250px; margin: auto; }
header { margin-bottom: 22px; }
h1 { margin: 0 0 7px; font-size: 25px; }
.sub, .label { color: var(--muted); }
.sub { font-size: 14px; }
.grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
}
.card {
    min-width: 0;
    padding: 19px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
}
.label { font-size: 13px; }
.value { margin-top: 10px; font-size: 30px; font-weight: 700; }
.unit { color: var(--muted); font-size: 13px; }
.section { margin-top: 16px; }
h2 { margin: 0 0 15px; font-size: 17px; }
.chart-wrap { height: 300px; }
#latestImage img, #detection {
    display: block;
    width: 100%;
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    border: 1px solid var(--border);
}
.empty { color: var(--muted); font-size: 14px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; min-width: 600px; }
th, td {
    padding: 12px 10px;
    border-bottom: 1px solid var(--border);
    text-align: left;
    font-size: 13px;
}
th { color: var(--muted); font-weight: 600; }
a { color: var(--accent); }
#updated { color: var(--muted); font-size: 12px; margin-top: 10px; }
@media (max-width: 800px) {
    .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 480px) {
    body { padding: 14px; }
    .grid { grid-template-columns: 1fr 1fr; gap: 9px; }
    .card { padding: 14px; }
    .value { font-size: 23px; }
    .chart-wrap { height: 240px; }
}
</style>
</head>
<body>
<main>
<header>
    <h1>Smart Inspection Helmet</h1>
    <div class="sub">Local industrial gauge monitoring</div>
</header>

<section class="grid" aria-label="Pressure statistics">
    <div class="card"><div class="label">Current pressure</div>
        <div class="value" id="current">--</div><div class="unit" id="unit">bar</div></div>
    <div class="card"><div class="label">Minimum</div>
        <div class="value" id="minimum">--</div><div class="unit">Recorded</div></div>
    <div class="card"><div class="label">Maximum</div>
        <div class="value" id="maximum">--</div><div class="unit">Recorded</div></div>
    <div class="card"><div class="label">Average</div>
        <div class="value" id="average">--</div><div class="unit">Recorded</div></div>
</section>

<section class="card section">
    <h2>Pressure history</h2>
    <div class="chart-wrap"><canvas id="pressureChart"></canvas></div>
    <div id="chartMessage" class="empty"></div>
</section>

<section class="card section">
    <h2>Gauge detection</h2>
    <img id="detection" src="/detection" alt="Live webcam with gauge detection drawn">
</section>

<section class="card section">
    <h2>Latest gauge screenshot</h2>
    <div id="latestImage" class="empty">Waiting for the first saved reading...</div>
</section>

<section class="card section">
    <h2>Reading history</h2>
    <div class="table-wrap">
        <table>
            <thead><tr><th>Timestamp</th><th>Pressure</th><th>Angle</th><th>Screenshot</th></tr></thead>
            <tbody id="history"><tr><td colspan="4">No readings yet.</td></tr></tbody>
        </table>
    </div>
    <div id="updated"></div>
</section>
</main>

<script>
let pressureChart = null;

function safeText(value) {
    return String(value ?? "");
}

async function refreshDashboard() {
    try {
        const response = await fetch("/api/readings", { cache: "no-store" });
        if (!response.ok) throw new Error("Could not load readings");
        const readings = await response.json();

        if (!readings.length) {
            document.getElementById("chartMessage").textContent =
                "No readings have been saved yet.";
            return;
        }

        const values = readings.map(r => Number(r.pressure))
            .filter(Number.isFinite);
        if (!values.length) return;

        const latest = readings[readings.length - 1];
        const unit = safeText(latest.unit || "bar");

        document.getElementById("current").textContent =
            Number(latest.pressure).toFixed(2);
        document.getElementById("unit").textContent = unit;
        document.getElementById("minimum").textContent =
            Math.min(...values).toFixed(2);
        document.getElementById("maximum").textContent =
            Math.max(...values).toFixed(2);
        document.getElementById("average").textContent =
            (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(2);

        const labels = readings.map(r => safeText(r.timestamp));
        const chartData = readings.map(r => Number(r.pressure));

        if (!pressureChart) {
            pressureChart = new Chart(document.getElementById("pressureChart"), {
                type: "line",
                data: {
                    labels,
                    datasets: [{
                        label: `Pressure (${unit})`,
                        data: chartData,
                        borderColor: "#55aaff",
                        backgroundColor: "rgba(85,170,255,0.12)",
                        fill: true,
                        tension: 0.25,
                        pointRadius: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        x: { ticks: { color: "#9aa4b2", maxTicksLimit: 7 } },
                        y: { ticks: { color: "#9aa4b2" } }
                    },
                    plugins: { legend: { labels: { color: "#edf1f5" } } }
                }
            });
        } else {
            pressureChart.data.labels = labels;
            pressureChart.data.datasets[0].data = chartData;
            pressureChart.data.datasets[0].label = `Pressure (${unit})`;
            pressureChart.update("none");
        }

        const imageName = encodeURIComponent(latest.image || "");
        document.getElementById("latestImage").innerHTML = latest.image
            ? `<a href="/snapshots/${imageName}" target="_blank" rel="noopener">
                 <img src="/snapshots/${imageName}" alt="Latest annotated gauge reading">
               </a>`
            : "No screenshot attached to the latest reading.";

        const history = readings.slice().reverse();
        const tbody = document.getElementById("history");
        tbody.replaceChildren();

        for (const item of history) {
            const row = document.createElement("tr");
            const fields = [
                safeText(item.timestamp),
                `${Number(item.pressure).toFixed(2)} ${safeText(item.unit || "bar")}`,
                `${Number(item.angle).toFixed(2)}°`
            ];
            for (const text of fields) {
                const cell = document.createElement("td");
                cell.textContent = text;
                row.appendChild(cell);
            }
            const imageCell = document.createElement("td");
            if (item.image) {
                const link = document.createElement("a");
                link.href = `/snapshots/${encodeURIComponent(item.image)}`;
                link.target = "_blank";
                link.rel = "noopener";
                link.textContent = "View image";
                imageCell.appendChild(link);
            } else {
                imageCell.textContent = "Unavailable";
            }
            row.appendChild(imageCell);
            tbody.appendChild(row);
        }

        document.getElementById("chartMessage").textContent = "";
        document.getElementById("updated").textContent =
            `Last refreshed: ${new Date().toLocaleTimeString()}`;
    } catch (error) {
        document.getElementById("updated").textContent =
            `Dashboard update error: ${error.message}`;
    }
}

refreshDashboard();
setInterval(refreshDashboard, 2000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import numpy as np

    publish_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    assert _jpeg[:2] == b"\xff\xd8"
    print("detection jpeg ok", len(_jpeg))
