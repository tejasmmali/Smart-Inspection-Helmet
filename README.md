

# Smart Inspection Helmet

A smart helmet project that uses computer vision to read analog gauges and monitor industrial equipment.

## Hardware

* RDK X5
* Logitech C270 Webcam
* Laptop for development


## Progress

* [x] Set up Python environment
* [x] Capture webcam video using OpenCV
* [x] Configure webcam at 640×480 using MJPEG
* [x] Detect gauge circle and estimate its center and radius
* [x] Implement initial line detection using OpenCV
* [x] Filter lines to identify possible needle candidates
* [x] Improve needle detection accuracy
* [x] Calculate needle angle reliably
* [x] Convert angle into gauge readings
* [x] Test readings with live webcam footage
* [x] Store readings for monitoring
* [x] Deploy on RDK X5
* [ ] Integrate hardware into the helmet


## Tech Stack

* Python
* OpenCV
* NumPy
* HTML, CSS, JavaScript
## How to Run

### Requirements

- Python 3.10 or newer
- Webcam (Logitech C270 recommended)
- OpenCV
- NumPy

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Smart-Inspection-Helmet
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install opencv-python numpy
```

Install any additional dependencies required by the web dashboard if applicable.

### 4. Connect the Webcam

Connect the  webcam and make sure it is detected by your computer. (use internal webcam if you have laptop)

### 5. Run the Project

```bash
python gauge-read.py
```



## Screenshots & Project Images

### Hardware Setup

![Smart Inspection Helmet Hardware](https://stardance.hackclub.com/rails/active_storage/representations/proxy/eyJfcmFpbHMiOnsiZGF0YSI6NTM1NTU2LCJwdXIiOiJibG9iX2lkIn19--8956d20fe029cdc7ba6b596f61f15a6d9e504c77/eyJfcmFpbHMiOnsiZGF0YSI6eyJmb3JtYXQiOiJ3ZWJwIiwicmVzaXplX3RvX2xpbWl0IjpbMTYwMCw5MDBdLCJzYXZlciI6eyJzdHJpcCI6dHJ1ZSwicXVhbGl0eSI6NzV9fSwicHVyIjoidmFyaWF0aW9uIn19--5394ecd620f1b8ee9be71be3f37cd22b8a88953c/WhatsApp%20Image%202026-11-01%20at%202.18.32%20AM.jpeg)

## More Project Screenshots

### detecting the circle in the gauge (Oct 21)

<img width="1164" height="671" alt="image" src="https://github.com/user-attachments/assets/ad115612-cdce-4df6-95f9-66f3ff8eefc9" />

<img width="402" height="313" alt="image" src="https://github.com/user-attachments/assets/528669e5-bf77-43a2-a677-1b1e5cffaad5" />

### added angle detection (26 Oct)

<img width="625" height="574" alt="image" src="https://github.com/user-attachments/assets/9a940bf1-b1c9-4b6f-a9de-2b09d95716d1" />

### reading with angle detection (30 Oct)

<img width="1493" height="664" alt="image" src="https://github.com/user-attachments/assets/bf142d2f-abda-40e7-a7bb-91d048ba716d" />

### Added detection in webcam feed (Oct 1)

<img width="1182" height="900" alt="image" src="https://github.com/user-attachments/assets/a1b3556b-b718-4161-8e06-b38b579df68a" />




## Goal

Build a low-cost smart inspection helmet that automatically reads industrial gauges and helps workers monitor equipment.

## Links
- [Youtube](https://youtu.be/O2CuzHTImEc)
- [stardance](https://stardance.hackclub.com/projects/69752)
- [Phantom ](https://phantom.hackclub.com/projects/3191)

