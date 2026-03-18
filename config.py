# ============================================================
#  config.py  –  All settings for the Vehicle Counter system
#  Edit this file to customize for your specific road/setup
# ============================================================

# --- Model ---
YOLO_MODEL = "yolov8n.pt"   # 'n'=nano (fast), 's'=small (more accurate)
CONFIDENCE  = 0.40           # Detection confidence (0–1). Raise if too many false detections.

# --- Vehicle classes (YOLO class IDs) ---
# YOLO was trained on COCO dataset. Rickshaws/CNGs are detected as 'motorcycle' (class 3).
VEHICLE_CLASSES = {
    2:  "car",
    3:  "rickshaw/motorcycle",   # covers rickshaws, CNGs, motorcycles
    5:  "bus",
    7:  "truck",
    1:  "bicycle",
}

# --- Tracking ---
MAX_AGE       = 30   # frames to keep a track alive when vehicle is hidden
MIN_HITS      = 2    # detections before track is confirmed
IOU_THRESHOLD = 0.3

# --- Speed estimation ---
# Measure a real-world distance visible in your video frame.
# e.g. a standard Bangladeshi lane = ~3.5 metres wide.
# Then count how many pixels that same distance occupies in the video.
# pixels_per_meter = pixel_distance / real_distance_in_meters
PIXELS_PER_METER = 55    # ← YOU MUST CALIBRATE THIS for your video
VIDEO_FPS        = 25    # Frames per second of your video / camera

# --- Counting line ---
# A horizontal line drawn across the road frame.
# Vehicles crossing this line get counted.
# Value = fraction of frame height (0.0=top, 1.0=bottom)
COUNTING_LINE_POSITION = 0.55   # 55% down from top

# --- Intersection zones ---
# Define rectangular regions (x1%, y1%, x2%, y2%) as fractions of frame size.
# These split the frame into separate road directions.
# Set ENABLE_ZONES=False to disable multi-lane counting.
ENABLE_ZONES = False
ZONES = {
    "North": (0.0,  0.0,  1.0,  0.45),   # top half
    "South": (0.0,  0.55, 1.0,  1.0 ),   # bottom half
    # "East":  (0.55, 0.0,  1.0,  1.0 ),
    # "West":  (0.0,  0.0,  0.45, 1.0 ),
}

# --- Output ---
DATA_FOLDER  = "data"    # CSV logs are saved here
SHOW_WINDOW  = True      # Show video window while processing (set False for headless)
SHOW_SPEED   = True      # Overlay speed on screen
SHOW_IDS     = True      # Overlay track IDs on screen
WINDOW_WIDTH = 960       # Display window width in pixels
