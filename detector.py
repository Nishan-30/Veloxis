# ============================================================
#  detector.py  –  Core detection + tracking engine
#  Uses YOLOv8 for detection, DeepSORT for tracking
# ============================================================

import cv2
import math
import datetime
import csv
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import config

# Load polygon zones from lane_tool if available
try:
    from lane_tool import load_polygon_zones, point_in_polygon
    _POLYGON_ZONES = load_polygon_zones()   # None if not yet drawn
except Exception:
    _POLYGON_ZONES = None
    def point_in_polygon(px, py, pts, w, h): return False

# Colours for drawing boxes (one per vehicle type)
COLOURS = {
    "car":                (57,  197, 187),  # teal
    "rickshaw/motorcycle":(255, 165,  30),  # orange
    "bus":                (220,  80,  80),  # red
    "truck":              (130,  80, 220),  # purple
    "bicycle":            (80,  180,  80),  # green
}
DEFAULT_COLOUR = (200, 200, 200)


class VehicleDetector:
    """
    Loads YOLO model + DeepSORT tracker.
    Call process_frame() on each video frame.
    """

    def __init__(self, session_label="session"):
        print("[INFO] Loading YOLO model... (downloads ~6 MB on first run)")
        self.model   = YOLO(config.YOLO_MODEL)
        self.tracker = DeepSort(
            max_age       = config.MAX_AGE,
            n_init        = config.MIN_HITS,
            nms_max_overlap = 1.0,
            max_cosine_distance = 0.3,
        )
        self.counted_ids   = set()          # global unique vehicle IDs already counted
        self.zone_counts   = {z: {} for z in config.ZONES}  # zone → {vtype: count}
        self.total_counts  = {}             # {vtype: count}
        self.speed_history = {}             # track_id → list of (x_centre, frame_no)
        self.frame_no      = 0
        self.session_label = session_label

        # Prepare CSV log file
        os.makedirs(config.DATA_FOLDER, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(config.DATA_FOLDER, f"log_{ts}.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "track_id", "vehicle_type",
                             "zone", "speed_kmh", "session"])

        print(f"[INFO] Logging to {self.csv_path}")

    # ----------------------------------------------------------
    def process_frame(self, frame):
        """
        Run detection + tracking on one frame.
        Returns annotated frame + summary dict.
        """
        self.frame_no += 1
        h, w = frame.shape[:2]
        line_y = int(h * config.COUNTING_LINE_POSITION)

        # ── 1. YOLO detection ──────────────────────────────────
        results = self.model(frame, verbose=False, conf=config.CONFIDENCE)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in config.VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # DeepSORT expects [left, top, width, height]
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        # ── 2. DeepSORT tracking ───────────────────────────────
        tracks = self.tracker.update_tracks(detections, frame=frame)

        newly_counted = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid  = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx   = (l + r) // 2
            cy   = (t + b) // 2

            # Vehicle type from class id
            det_cls  = getattr(track, "det_class", None)
            vtype    = config.VEHICLE_CLASSES.get(det_cls, "vehicle") if det_cls else "vehicle"
            colour   = COLOURS.get(vtype, DEFAULT_COLOUR)

            # ── Speed estimation ───────────────────────────────
            speed_kmh = self._estimate_speed(tid, cx, self.frame_no)

            # ── Counting line logic ────────────────────────────
            if tid not in self.counted_ids and (t < line_y < b or cy > line_y):
                self.counted_ids.add(tid)
                zone = self._get_zone(cx, cy, w, h) if config.ENABLE_ZONES else "all"
                self.total_counts[vtype] = self.total_counts.get(vtype, 0) + 1
                if config.ENABLE_ZONES:
                    if zone not in self.zone_counts:
                        self.zone_counts[zone] = {}
                    self.zone_counts[zone][vtype] = self.zone_counts[zone].get(vtype, 0) + 1
                self._log(tid, vtype, zone, speed_kmh)
                newly_counted.append(tid)

            # ── Draw bounding box ──────────────────────────────
            cv2.rectangle(frame, (l, t), (r, b), colour, 2)
            label = vtype.split("/")[0]   # show "rickshaw" not full string
            if config.SHOW_IDS:
                label += f" #{tid}"
            if config.SHOW_SPEED and speed_kmh:
                label += f"  {speed_kmh:.0f} km/h"
            cv2.rectangle(frame, (l, t - 22), (l + len(label) * 9, t), colour, -1)
            cv2.putText(frame, label, (l + 3, t - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # ── Draw counting line ─────────────────────────────────
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 100), 2)
        cv2.putText(frame, "Counting line", (8, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)

        # ── Draw zones ────────────────────────────────────────
        if config.ENABLE_ZONES:
            self._draw_zones(frame, w, h)

        # ── Overlay counts ─────────────────────────────────────
        frame = self._draw_counts(frame)

        summary = {
            "total_unique": len(self.counted_ids),
            "by_type":      self.total_counts,
            "by_zone":      self.zone_counts,
            "frame":        self.frame_no,
        }
        return frame, summary

    # ----------------------------------------------------------
    def _estimate_speed(self, tid, cx, frame_no):
        if config.PIXELS_PER_METER <= 0:
            return None
        hist = self.speed_history.setdefault(tid, [])
        hist.append((cx, frame_no))
        if len(hist) < 5:
            return None
        # Use oldest and newest sample in window
        old_cx, old_f = hist[-5]
        pixel_dist    = abs(cx - old_cx)
        frames_elapsed = max(frame_no - old_f, 1)
        mps  = (pixel_dist / config.PIXELS_PER_METER) / (frames_elapsed / config.VIDEO_FPS)
        kmh  = mps * 3.6
        # Keep history short
        if len(hist) > 20:
            self.speed_history[tid] = hist[-10:]
        return round(kmh, 1) if 2 < kmh < 150 else None   # sanity filter

    # ----------------------------------------------------------
    def _get_zone(self, cx, cy, w, h):
        # Prefer polygon zones drawn with lane_tool
        if _POLYGON_ZONES:
            for lane in _POLYGON_ZONES:
                if point_in_polygon(cx, cy, lane["points"], w, h):
                    return lane["name"]
            return "other"
        # Fallback: bounding-box zones from config.py
        fx, fy = cx / w, cy / h
        for name, (x1, y1, x2, y2) in config.ZONES.items():
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                return name
        return "other"

    # ----------------------------------------------------------
    def _draw_zones(self, frame, w, h):
        ZONE_COLS = [(255,165,30),(57,197,187),(220,80,80),(130,80,220),
                     (80,200,80),(220,180,40),(200,80,160),(80,160,220)]

        # Polygon zones (from lane_tool)
        if _POLYGON_ZONES:
            for i, lane in enumerate(_POLYGON_ZONES):
                c   = ZONE_COLS[i % len(ZONE_COLS)]
                pts = np.array([[int(fx*w), int(fy*h)]
                                for fx, fy in lane["points"]], dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], c)
                cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
                cv2.polylines(frame, [pts], isClosed=True, color=c, thickness=2)
                cnt = sum(self.zone_counts.get(lane["name"], {}).values())
                cx_ = int(pts[:,0].mean()); cy_ = int(pts[:,1].mean())
                cv2.putText(frame, f"{lane['name']}: {cnt}",
                            (cx_-50, cy_), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,255), 3, cv2.LINE_AA)
                cv2.putText(frame, f"{lane['name']}: {cnt}",
                            (cx_-50, cy_), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, c, 1, cv2.LINE_AA)
            return

        # Fallback: bounding-box zones
        for i, (name, (x1,y1,x2,y2)) in enumerate(config.ZONES.items()):
            c = ZONE_COLS[i % len(ZONE_COLS)]
            px1,py1 = int(x1*w), int(y1*h)
            px2,py2 = int(x2*w), int(y2*h)
            cv2.rectangle(frame, (px1,py1),(px2,py2), c, 1)
            cnt = sum(self.zone_counts.get(name,{}).values())
            cv2.putText(frame, f"{name}: {cnt}", (px1+6,py1+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)

    # ----------------------------------------------------------
    def _draw_counts(self, frame):
        panel = frame.copy()
        x, y0 = 10, 30
        overlay_lines = [f"Total vehicles: {len(self.counted_ids)}"]
        for vtype, cnt in self.total_counts.items():
            overlay_lines.append(f"  {vtype}: {cnt}")
        for i, line in enumerate(overlay_lines):
            cv2.putText(panel, line, (x, y0 + i*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(panel, line, (x, y0 + i*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20,20,20), 1, cv2.LINE_AA)
        return panel

    # ----------------------------------------------------------
    def _log(self, tid, vtype, zone, speed_kmh):
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ts, tid, vtype, zone,
                                    speed_kmh if speed_kmh else "",
                                    self.session_label])

    # ----------------------------------------------------------
    def print_summary(self):
        print("\n" + "="*45)
        print(f"  SESSION COMPLETE — {self.session_label}")
        print("="*45)
        print(f"  Total unique vehicles counted: {len(self.counted_ids)}")
        for vtype, cnt in self.total_counts.items():
            bar = "█" * min(cnt, 40)
            print(f"  {vtype:<22} {cnt:>4}  {bar}")
        if config.ENABLE_ZONES:
            print("\n  ── By zone ──")
            for zone, counts in self.zone_counts.items():
                if counts:
                    print(f"  {zone}: {counts}")
        print(f"\n  Log saved to: {self.csv_path}")
        print("="*45)
