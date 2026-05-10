<div align="center">

# VELOXIS
### AI-Powered Traffic Analysis Platform

*Developed for transportation engineering research on Bangladesh's mixed-traffic urban roads*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Custom%20BD%20Model-orange?logo=opencv)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)](https://microsoft.com)

**Nishan** · B.Sc. Civil & Environmental Engineering · SUST · 2026  
*A product of [NextCity Tessera](https://github.com/NextCityTessera)*

</div>

---

## Overview

VELOXIS is a desktop application for real-time and offline traffic analysis, built specifically for the vehicle mix and road conditions found in Bangladesh. It combines a custom-trained YOLOv11 object detection model with a custom IoU tracker (Hungarian assignment + velocity prediction) to deliver intersection-level traffic metrics — vehicle counts, speed distributions, queue lengths, and peak-hour factors — through a clean, research-oriented interface.

The application was developed as a final-year thesis instrument at the Department of Civil and Environmental Engineering, SUST, focusing on non-motorized transport (NMT) behavior and intersection capacity analysis on mixed-traffic urban corridors.

---

## Key Features

**Detection & Tracking**
- YOLOv11 model custom-trained on 45,862 annotated images of Bangladeshi road vehicles
- Custom `_iou_track()` — Hungarian assignment + velocity prediction, no external ReID file needed — tuned for dense, occluded BD intersections
- 8 native vehicle classes: car, rickshaw, CNG/auto, motorcycle, bicycle, bus, truck, easybike
- Bidirectional counting with forward/backward classification
- AI-assisted counting line placement from optical flow analysis

**Traffic Engineering Metrics**
- Peak Hour Factor (PHF) — 15-minute interval volume tracking per HCM 6th edition
- Average headway and saturation flow rate
- 85th percentile speed (V85) and mean speed
- Queue length detection (vehicles stopped behind line)
- Occupancy percentage and live vehicle density
- Near-miss and sudden braking safety event detection

**Analysis & Export**
- Real-time HUD overlay with all metrics
- Analytics dashboard with daily, hourly, monthly, and directional charts
- Per-vehicle CSV log including all session-level metrics at time of crossing
- Session summary CSV: single-row report with PHF, V85, saturation flow, safety events
- PTV Vissim / Aimsun compatible export (volume by vehicle type per hour)
- Homography-based speed calibration for perspective-correct measurement

**Interface**
- Live camera support: webcam, USB, DroidCam (WiFi), RTSP/IP streams
- Click-to-draw counting line with visual feedback
- Video seek and frame preview before analysis
- Lane/zone drawing for approach-wise counting
- CPU performance mode for integrated-graphics laptops
- Light/dark theme

---

## Installation

**Requirements:** Python 3.10+, Windows 10/11

```bash
# Clone the repository
git clone https://github.com/Nishan-30/veloxis.git
cd veloxis

# First-time setup (installs all dependencies)
setup_windows.bat
```

Or manually:

```bash
pip install ultralytics opencv-python customtkinter pillow matplotlib pandas flask
```

**Model file:** Place `bd_vehicles_yolo11.pt` in the project root directory. The model is not included in the repository due to file size — see [Releases](https://github.com/yourusername/veloxis/releases) or train your own using the instructions below.

---

## Quick Start

```bash
python app_windows.py
```

1. Go to **File Detection** → Browse a video file
2. Click on the video frame to set the counting line (or enable AI auto-detect)
3. Click **Analyse Video**
4. Results appear in the right panel; CSV logs are saved to `data/`

For live camera:

```bash
# Navigate to Live Detection → select source → Start Detection
python app_windows.py
```

---

## Configuration

All settings are accessible through the **Settings** page in the app, or directly in `config.py`:

```python
YOLO_MODEL           = "bd_vehicles_yolo11.pt"  # detection model
CONFIDENCE           = 0.35                      # lower for crowded roads, higher for clear daylight
FRAME_SKIP           = 1                         # 1 = full accuracy, 2 = faster on CPU
CPU_PERFORMANCE_MODE = True                      # enable for integrated-graphics machines
PIXELS_PER_METER     = 55                        # calibrate for accurate speed
USE_DUAL_LINES       = False                     # True for bidirectional roads
```

For accurate speed estimation, use the **Calibrate Speed** page to perform a 4-point homography calibration using known road markings.

---

## Custom Model Training

The included model was trained on the [BD-Vehicle-Detection](https://roboflow.com) dataset. To retrain or extend it with additional vehicle classes (battery rickshaw, human hauler, leguna, nosimon, etc.):

1. Annotate images on [Roboflow](https://roboflow.com) — export in YOLOv8 format
2. Open `train_yolo11_kaggle.py` in a Kaggle notebook with GPU T4 or T4×2
3. Fill in your Roboflow credentials and run all cells (~3–4 hours)
4. Download `bd_vehicles_yolo11.pt` from the Kaggle output and place it in the project root

```python
# train_yolo11_kaggle.py — edit only these lines
ROBOFLOW_API_KEY   = "your_api_key"
ROBOFLOW_WORKSPACE = "your_workspace"
ROBOFLOW_PROJECT   = "your_project_slug"
ROBOFLOW_VERSION   = 1
MODEL_SIZE         = "s"   # n / s / m / l
```

**Current model performance:**

| Metric | Value |
|--------|-------|
| mAP50 | 0.871 |
| Precision | 0.882 |
| Recall | 0.782 |
| Training images | 45,862 |
| Classes | 8 |

---

## Output Files

Each detection session generates files in the `data/` directory:

| File | Contents |
|------|----------|
| `log_YYYYMMDD_HHMMSS.csv` | Per-vehicle crossing log with all metrics |
| `log_YYYYMMDD_HHMMSS_summary.csv` | Single-row session summary (PHF, V85, saturation flow, etc.) |
| `log_YYYYMMDD_HHMMSS_vissim.csv` | PTV Vissim / Aimsun compatible input |
| `snapshots/` | Manual snapshots from detection |
| `homography.npy` | Saved speed calibration matrix |
| `lanes.json` | Saved lane/zone polygon definitions |

---

## Project Structure

```
veloxis/
├── app_windows.py        # Main GUI application (~2,200 lines, CustomTkinter)
├── detector.py           # Detection engine — YOLOv11 + custom _iou_track() + metrics
├── config.py             # All configurable settings (ASCII-safe)
├── lane_tool.py          # Standalone lane drawing tool (OpenCV)
├── live_detect.py        # Headless live detection (CLI)
├── file_detect.py        # Headless file detection (CLI)
├── dashboard.py          # Web dashboard (Flask, port 5000)
├── train_yolo11_kaggle.py # Kaggle training notebook
├── setup_windows.bat     # One-click dependency installer
├── data/                 # Session logs, calibration files
└── videos/               # Place input videos here
```

---

## Detection Pipeline

```
Video / Camera
      ↓
Night enhancement (CLAHE, if dark frame)
      ↓
Resize to 416px (CPU mode) or 640px (GPU mode)
      ↓
YOLOv11 inference  →  coord_scale back to original resolution
      ↓
BoTSORT tracking  →  replaced by custom _iou_track() below
      ↓
_iou_track()      (Hungarian assignment + velocity prediction, no ReID file needed)
      ↓
Re-ID cache lookup  (90-frame window, prevents double-counting)
      ↓
Line crossing detection  →  direction, zone, speed
      ↓
PHF · headway · V85 · queue · occupancy
      ↓
HUD overlay  +  CSV log  +  summary
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| App won't start | `pip install customtkinter pillow matplotlib pandas ultralytics` |
| "Loading YOLO model..." hangs | Check `bd_vehicles_yolo11.pt` is in the project folder; first load takes 20–30s |
| Slow detection | Enable CPU Performance Mode in Settings, or set `FRAME_SKIP=2` in `config.py` |
| No camera detected | Live Detection → Auto-scan; try camera index 0, 1, or 2 |
| DroidCam not connecting | Ensure phone and laptop are on the same WiFi; test connection in the app |
| Speed values unreliable | Use Calibrate Speed page for homography calibration |
| ID switches / double counting | Already mitigated by Re-ID cache (90-frame) and custom IoU tracker — if still occurring, lower `CONFIDENCE` slightly |

---

## Acknowledgements

- Vehicle detection model trained using the [Roboflow](https://roboflow.com) platform and Kaggle GPU compute
- Object detection: [Ultralytics YOLOv11](https://docs.ultralytics.com)
- Multi-object tracking: custom `_iou_track()` — Hungarian assignment + velocity prediction (replaces BoTSORT)
- UI framework: [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)

---

## License

MIT License · Copyright © 2026 Nishan, NextCity Tessera

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

