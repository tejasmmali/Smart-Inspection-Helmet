

# Smart Inspection Helmet

A smart helmet project that uses computer vision to read analog gauges and monitor industrial equipment.

## Hardware

* RDK X5
* Logitech C270 Webcam
* Laptop for development

![Alt text for image](https://stardance.hackclub.com/rails/active_storage/representations/proxy/eyJfcmFpbHMiOnsiZGF0YSI6NTM1NTU2LCJwdXIiOiJibG9iX2lkIn19--8956d20fe029cdc7ba6b596f61f15a6d9e504c77/eyJfcmFpbHMiOnsiZGF0YSI6eyJmb3JtYXQiOiJ3ZWJwIiwicmVzaXplX3RvX2xpbWl0IjpbMTYwMCw5MDBdLCJzYXZlciI6eyJzdHJpcCI6dHJ1ZSwicXVhbGl0eSI6NzV9fSwicHVyIjoidmFyaWF0aW9uIn19--5394ecd620f1b8ee9be71be3f37cd22b8a88953c/WhatsApp%20Image%202026-11-01%20at%202.18.32%20AM.jpeg)

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

## Goal

Build a low-cost smart inspection helmet that automatically reads industrial gauges and helps workers monitor equipment.

## Links
- [Youtube](https://youtu.be/O2CuzHTImEc)
- [stardance](https://stardance.hackclub.com/projects/69752)
- [Phantom ](https://phantom.hackclub.com/projects/3191)

