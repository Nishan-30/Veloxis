# Bangladesh Road Vehicle Counter
### Built for CEE transportation research — Rickshaws included 🛺

**100% free | YOLOv8 + DeepSORT | Flask dashboard**

---

## What this system does
- Detects and counts vehicles in real-time or from a saved video
- Identifies: cars, rickshaws/CNGs, buses, trucks, bicycles
- Estimates vehicle speed (km/h)
- Supports multi-lane / intersection zone counting
- Saves all data to CSV logs
- Shows daily, monthly, and hourly graphs in a browser dashboard

---

## Step-by-step Setup

### Step 1 — Install Python
1. Go to https://python.org/downloads
2. Download Python **3.10** (recommended)
3. Run the installer — **tick "Add Python to PATH"**
4. Click Install

### Step 2 — Download this project
Put the whole `vehicle_counter` folder anywhere on your laptop (e.g. Desktop).

### Step 3 — Install packages

**Windows users — easy way:**
Double-click `setup_windows.bat` — it does everything automatically.

**Manual (all platforms):**
Open Command Prompt / Terminal and run:
```
pip install ultralytics deep-sort-realtime opencv-python numpy pandas matplotlib flask Pillow
```
This takes 3–5 minutes. The YOLOv8 model (~6 MB) downloads automatically on first run.

---

## How to run

### Option A — Main menu (recommended)
```
python main.py
```
Choose from the menu:
1. Live detection (webcam or phone)
2. File detection (video you recorded)
3. Dashboard (graphs in browser)

### Option B — Run directly
```
python file_detect.py          # analyse a saved video
python live_detect.py          # live camera
python dashboard.py            # open dashboard at localhost:5000
```

---

## Using your phone as a camera
1. Install **DroidCam** app (free, Google Play Store)
2. Connect phone & laptop to the **same WiFi**
3. Open DroidCam on phone — note the IP address shown
4. Choose option 3 (DroidCam) in `live_detect.py` menu
5. Enter the IP address

---

## Configuring for your road

Edit `config.py` to customise:

| Setting | What it does |
|---------|-------------|
| `CONFIDENCE` | Detection sensitivity (raise if too many false detections) |
| `PIXELS_PER_METER` | For speed calibration — measure a known distance in the frame |
| `COUNTING_LINE_POSITION` | Where the counting line sits (0.5 = middle of frame) |
| `ENABLE_ZONES` | Turn on multi-lane / intersection counting |
| `ZONES` | Define road directions as % of frame |
| `SHOW_WINDOW` | Show/hide the video window |

### Speed calibration
1. In your video, find something with a known real-world size (e.g. lane width = 3.5 m)
2. Count how many pixels that distance is in the frame
3. Set `PIXELS_PER_METER = pixel_count / real_metres`

### Intersection zone setup
1. Set `ENABLE_ZONES = True` in config.py
2. Edit the `ZONES` dictionary:
   - Values are fractions of frame: (x_start, y_start, x_end, y_end) from 0.0 to 1.0
   - Example: `"North": (0.0, 0.0, 1.0, 0.5)` = top half of frame

---

## Output files
All saved in the `data/` folder:
- `log_YYYYMMDD_HHMMSS.csv` — raw count log (one row per vehicle)
- `output_<videoname>.mp4` — annotated video (file mode only)

CSV columns: `timestamp, track_id, vehicle_type, zone, speed_kmh, session`

---

## Tips for field data collection
- Mount camera on a tripod or pole above the road for best results
- Aim for a top-down or 45-degree angle
- More vehicles visible at once = better counting accuracy
- For intersections: position camera so all road arms are visible
- Record at 720p or higher for better detection

---

## Troubleshooting

**"Module not found" error**
→ Run `pip install ultralytics deep-sort-realtime opencv-python numpy pandas matplotlib flask Pillow`

**Camera not opening**
→ Check another app isn't using the camera
→ For DroidCam: check both devices are on same WiFi

**Too many false detections**
→ Increase `CONFIDENCE` in config.py (try 0.5–0.6)

**Speed shows 0 or unrealistic values**
→ Calibrate `PIXELS_PER_METER` for your specific video frame

---

## Project files
```
vehicle_counter/
├── main.py          ← Start here
├── config.py        ← All settings
├── detector.py      ← Core YOLO + tracking engine
├── live_detect.py   ← Live camera mode
├── file_detect.py   ← Video file mode
├── dashboard.py     ← Web dashboard (Flask)
├── requirements.txt ← Package list
├── setup_windows.bat← One-click Windows setup
├── data/            ← CSV logs saved here
└── videos/          ← Put your test videos here
```
