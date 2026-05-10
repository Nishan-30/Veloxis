# ================================================================
#  config.py  --  VELOXIS v2.0
#  Author: NextCity Tessera
#  Product: VELOXIS
#  ASCII only -- no special characters
# ================================================================

# --- Model ---
# Custom YOLOv11 model trained on Bangladeshi vehicle dataset
# Place bd_vehicles_yolo11.pt in same folder as app_windows.py
# Fallback chain: bd_vehicles_best.pt -> yolo11s.pt -> yolov8n.pt
YOLO_MODEL = "bd_vehicles_yolo11.pt"

# YOLOv11 confidence guide:
#   Daylight clear road   : 0.40 - 0.50
#   Night / low light     : 0.25 - 0.35
#   Crowded intersection  : 0.30 - 0.40
CONFIDENCE = 0.30

# --- Performance ---
FRAME_SKIP    = 1      # 1 = full accuracy -- set 2 for slow CPU
RESIZE_BEFORE = True
RESIZE_WIDTH  = 640

# --- CPU Performance Mode (for machines without a dedicated GPU) ---
# Enable this on any laptop or desktop running on integrated/shared graphics.
# This enables: auto frame-skip, smaller inference resolution
# Auto-detected at startup -- you can also toggle it in Settings.
CPU_PERFORMANCE_MODE = True   # True = optimized for CPU / integrated graphics
CPU_RESIZE_WIDTH     = 416    # smaller than 640 -- faster on CPU (416 recommended)
CPU_FRAME_SKIP       = 2      # process every 2nd frame -- halves CPU load
# Speed display still works -- frame_skip is compensated in speed math

# --- Night enhancement ---
ENHANCE_NIGHT   = True
NIGHT_THRESHOLD = 60

# --- Vehicle classes (COCO fallback only -- used when bd_vehicles model not found) ---
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    6: "train",
}

# --- Tracking (note: these values are used by _iou_track in detector.py) ---
# IOU_MATCH and MAX_LOST are set directly in detector._iou_track() for stability.
# The values below are kept for reference / future externalization.
MAX_AGE       = 20   # reference only -- _iou_track uses MAX_LOST=15 internally
MIN_HITS      = 1    # reference only -- _iou_track uses MIN_HITS=1 internally
IOU_THRESHOLD = 0.18 # reference only -- _iou_track uses IOU_MATCH=0.18 internally

# --- Speed estimation ---
# Use Calibrate Speed page in VELOXIS for accurate homography.
# Typical px/m values:
#   Camera 3-4m high : 30-60  px/m
#   Camera 6-8m high : 15-30  px/m
#   Dashcam          : 80-150 px/m
# Set 0 to disable speed display.
PIXELS_PER_METER = 55
VIDEO_FPS = 30

# --- Display ---
SHOW_SPEED = True
SHOW_IDS = True
SHOW_WINDOW   = True
WINDOW_WIDTH  = 960
DATA_FOLDER   = "data"

# --- Human / pedestrian detection ---
DETECT_HUMANS = True

# --- Counting line ---
COUNTING_LINE_POSITION = 0.55

# --- Dual counting lines (bidirectional roads) ---
USE_DUAL_LINES = False
LINE_POS_A = 0.38
LINE_POS_B = 0.70

# --- Zones ---
ENABLE_ZONES = False
ZONES = {
    "North": (0.0, 0.0, 1.0, 0.45),
    "South": (0.0, 0.55, 1.0, 1.0),
}
