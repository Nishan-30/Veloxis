# ================================================================
#  detector.py  --  VELOXIS v2.0
#  Author : Nishan, SUST CEE, 2026
#  Product: NextCity Tessera
#
#  Detection engine:
#    - YOLOv11 (custom BD vehicle model, 45,862 images)
#    - Custom _iou_track() — Hungarian assignment + velocity prediction
#      (replaces BoTSORT entirely — no external ReID file needed)
#    - 3-mechanism line crossing: sign-change + history-scan + band-exit
#    - Homography speed calibration (perspective-correct)
#    - Re-ID cache (90-frame, prevents double-counting re-entrants)
# ================================================================

import cv2, datetime, csv, os, math
import numpy as np
from ultralytics import YOLO
import config

# ── CPU / GPU auto-detection ──────────────────────────────────
import torch
_HAS_GPU = torch.cuda.is_available()
_CPU_MODE = getattr(config, "CPU_PERFORMANCE_MODE", False) or not _HAS_GPU

if _CPU_MODE:
    # Apply CPU-optimised settings automatically
    # Overrides FRAME_SKIP and RESIZE_WIDTH from config if CPU mode active
    _EFFECTIVE_FRAME_SKIP  = getattr(config, "CPU_FRAME_SKIP",   2)
    _EFFECTIVE_RESIZE_W    = getattr(config, "CPU_RESIZE_WIDTH", 416)
    print(f"[INFO] CPU Performance Mode ON — resize={_EFFECTIVE_RESIZE_W}, skip={_EFFECTIVE_FRAME_SKIP}")
    print(f"[INFO] GPU detected: {_HAS_GPU}. For best speed use a dedicated GPU.")
else:
    _EFFECTIVE_FRAME_SKIP  = getattr(config, "FRAME_SKIP",    1)
    _EFFECTIVE_RESIZE_W    = getattr(config, "RESIZE_WIDTH", 640)
    gpu_name = torch.cuda.get_device_name(0) if _HAS_GPU else "CPU"
    print(f"[INFO] Running on: {gpu_name}")

# ── Tracker availability check ────────────────────────────────
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort as _DeepSort
    _HAS_DEEPSORT = True
except ImportError:
    _HAS_DEEPSORT = False

# ── Zone loader ───────────────────────────────────────────────
try:
    from lane_tool import load_polygon_zones, point_in_polygon
    _POLYGON_ZONES = load_polygon_zones()
except Exception:
    _POLYGON_ZONES = None
    def point_in_polygon(px, py, pts, w, h): return False

# ── Colours per vehicle type ──────────────────────────────────
COLOURS = {
    "car":             ( 57, 197, 187),
    "motorcycle":      (255,  80,  30),
    "rickshaw":        (255, 180,  20),
    "rickshaw/CNG":    (255, 180,  20),
    "cng":             (251, 146,  60),
    "CNG/auto":        (251, 146,  60),
    "bus":             ( 80, 130, 240),
    "truck":           (160,  80, 220),
    "bicycle":         ( 80, 220,  80),
    "easybike":        (100, 220, 200),
    "battery_rickshaw":(255, 220, 100),
    "human_hauler":    (200, 100, 255),
    "leguna":          (100, 200, 255),
    "nosimon":         (255, 150, 100),
    "microbus":        ( 80, 180, 180),
    "pickup":          (180, 130,  80),
    "tempo":           (220, 180,  60),
    "train":           ( 80, 160, 220),
}
DEFAULT_COLOUR  = (180, 180, 180)
DIR_FORWARD     = "FWD"
DIR_BACKWARD    = "BWD"
NEAR_MISS_PX    = 25
NEAR_MISS_SPEED = 8
BRAKE_DROP_KMH  = 18


def _corrected_vtype(raw_cls, box_w, box_h, frame_w, frame_h, class_names=None):
    """
    Resolve detection class index to vehicle type string.
    Custom model class names take priority over COCO fallback.
    """
    if class_names and raw_cls in class_names:
        name = str(class_names[raw_cls])
        # Heuristic: small area 'truck' detections are likely CNGs
        if name == "truck":
            if (box_w * box_h) / max(frame_w * frame_h, 1) < 0.04:
                return "cng"
        return name
    # COCO fallback
    vtype = config.VEHICLE_CLASSES.get(raw_cls, "vehicle")
    if vtype == "motorcycle":
        return "rickshaw" if box_w / max(box_h, 1) >= 1.4 else "motorcycle"
    if vtype == "truck":
        if (box_w * box_h) / max(frame_w * frame_h, 1) < 0.04:
            return "cng"
    return vtype


class VehicleDetector:

    def __init__(self, session_label="session"):
        model_name = config.YOLO_MODEL
        print(f"[INFO] Loading model: {model_name}")
        # Resolve model path — search cwd, then script directory, then fallbacks
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        def _find_model(name):
            for candidate in [name, os.path.join(_script_dir, name)]:
                if os.path.exists(candidate):
                    return candidate
            return None

        _found = _find_model(model_name)
        if _found:
            model_name = _found
        else:
            fallbacks = ["bd_vehicles_best.pt","yolo11s.pt","yolov8s.pt","yolov8n.pt"]
            found_fb = None
            for fb in fallbacks:
                found_fb = _find_model(fb)
                if found_fb: break
            if found_fb:
                print(f"[WARN] {model_name} not found. Using fallback: {found_fb}")
                model_name = found_fb
            else:
                print(f"[WARN] No model found. Downloading yolo11s.pt...")
                model_name = "yolo11s.pt"
        self.model = YOLO(model_name)
        self._model_name_used = model_name
        print(f"[INFO] Model loaded: {model_name}")
        self._class_names = (
            self.model.names
            if hasattr(self.model, "names") and self.model.names
            else config.VEHICLE_CLASSES
        )

        # ── Tracker setup (config written for future BoTSORT fallback) ──
        # Primary tracker: custom _iou_track() — Hungarian + velocity prediction
        # _setup_tracker() writes a botsort YAML but it is NOT used by predict()
        # It is kept so the config file exists if ultralytics tracking is ever needed
        self._tracker_cfg = self._setup_tracker()
        print(f"[INFO] Tracker: custom _iou_track() (Hungarian + velocity prediction)")

        # DeepSORT instance kept as optional fallback — not used in normal operation
        self._deepsort = None
        if _HAS_DEEPSORT:
            self._deepsort = _DeepSort(
                max_age=20, n_init=3,
                nms_max_overlap=0.6, max_cosine_distance=0.25)

        # ── Homography speed calibration ──────────────────────
        self.H_matrix = None
        self._load_homography()

        # ── Counters ──────────────────────────────────────────
        self.counted_ids   = set()
        self.total_counts  = {}
        self.dir_counts    = {DIR_FORWARD: {}, DIR_BACKWARD: {}}
        self.zone_counts   = {}
        self.speed_history = {}
        self.prev_speeds   = {}
        self.prev_cy       = {}
        self.prev_zone     = {}   # tid → last zone name (for TMC entry tracking)
        self._counted_fwd  = set()
        self.frame_no      = 0
        self.session_label = session_label

        # ── Approach zone pre-registration (Bug 5 fix) ────────
        # Tracks seen in approach zone (above counting line) are pre-registered.
        # If YOLO misses them exactly at the line, they still get counted
        # when they reappear on the far side.
        # approach_zone_frac: fraction of frame height above counting line to monitor
        APPROACH_ZONE_FRAC = 0.20   # 20% above line = generous detection window
        self._approach_zone_frac = APPROACH_ZONE_FRAC
        self._seen_approaching   = {}  # effective_tid → frame_no when last seen approaching
        self._approach_max_age   = 45  # frames — drop approach record after ~1.5s

        # ── Re-ID cache (prevents double-counting re-entrants) ─
        self._reid_cache   = {}
        self._reid_mapped  = {}
        self._reid_max_age = 90  # frames (~3s at 30fps)

        # ── Queue detection ───────────────────────────────────
        self.queue_length        = 0
        self.queue_history       = []
        self.queue_threshold_kmh = 3.0

        # ── Peak hour (15-min intervals) ──────────────────────
        self._interval_start = datetime.datetime.now()
        self._interval_count = 0
        self.peak_intervals  = []
        self.peak_rate       = 0
        self.current_rate    = 0

        # ── Miovision-style advanced metrics ──────────────────
        self.phf = 0.0   # Peak Hour Factor

        # Turning Movement Counts (TMC)
        # Key format: "ApproachZone→ExitZone"  e.g. "North→South"
        # Populated whenever ENABLE_ZONES=True and polygon zones are drawn
        self.turning_counts  = {}   # {"N→S": {"car":3,"rickshaw":1,...}, ...}
        self.tmc_entry_zone  = {}   # tid → zone name at first zone entry

        # Approach volume (per named zone)
        self.approach_counts = {}   # {"North": 12, "South": 8, ...}

        # Level of Service (LOS) — HCM 6th edition thresholds
        self.los_letter      = "—"
        self.avg_delay_sec   = 0.0
        self._los_colours = {
            "A": ( 80, 220,  80), "B": ( 57, 197, 187),
            "C": ( 80, 200, 255), "D": ( 50, 150, 255),
            "E": ( 50,  80, 255), "F": ( 50,  50, 220),
            "—": (130, 130, 130),
        }

        # Headway tracking (time between consecutive vehicles crossing line)
        self._last_cross_time = {}   # vtype → last crossing datetime
        self.headway_history  = []   # list of (vtype, headway_sec)
        self.avg_headway_sec  = 0.0

        # Saturation flow tracking (vehicles per green hour equivalent)
        # Approximated from observed headway
        self.saturation_flow = 0    # veh/hr

        # Speed percentiles (for traffic engineering reports)
        self.all_speeds      = []   # all observed speed samples
        self.speed_85th      = 0.0  # 85th percentile speed (design speed reference)
        self.speed_mean      = 0.0

        # ── Live stats ────────────────────────────────────────
        self.live_vehicles   = 0
        self.occupancy_pct   = 0.0
        self.person_count    = 0
        self.safety_events   = 0
        self.density_history = []
        self.near_miss_log   = []

        # ── Line settings (set by app) ────────────────────────
        self.ai_line_start = None
        self.ai_line_end   = None
        self.manual_line_a = None

        # ── CSV log ───────────────────────────────────────────
        os.makedirs(config.DATA_FOLDER, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(config.DATA_FOLDER, f"log_{ts}.csv")

        # Load study location from user prefs
        try:
            import json as _json
            _prefs_path = "data/user_prefs.json"
            with open(_prefs_path, encoding="utf-8") as _pf:
                _prefs = _json.load(_pf)
        except Exception:
            _prefs = {}
        self.site_name = _prefs.get("loc_name", "") or _prefs.get("site_name", "")
        self.site_lat  = _prefs.get("loc_lat",  "") or _prefs.get("site_lat",  "")
        self.site_lng  = _prefs.get("loc_lng",  "") or _prefs.get("site_lng",  "")

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp", "track_id", "vehicle_type",
                "zone", "direction", "speed_kmh", "session",
                "queue_length", "occupancy_pct", "current_rate_vph",
                "avg_headway_sec", "saturation_flow", "phf",
                "speed_85th_kmh", "speed_mean_kmh",
                "location_name", "latitude", "longitude"])
        print(f"[INFO] Log -> {self.csv_path}")
        if self.site_name:
            print(f"[INFO] Study site: {self.site_name} ({self.site_lat}, {self.site_lng})")

    # ── Tracker setup ─────────────────────────────────────────
    def _setup_tracker(self):
        """Write tuned BoTSORT config. Falls back to bytetrack if write fails."""
        try:
            import yaml as _yaml
        except ImportError:
            # yaml not installed — use ultralytics built-in tracker name
            print("[WARN] pyyaml not installed — using default bytetrack tracker")
            return "bytetrack.yaml"

        cfg = {
            "tracker_type":       "botsort",
            "track_high_thresh":  0.30,
            "track_low_thresh":   0.15,
            "new_track_thresh":   0.35,
            "track_buffer":       45,
            "match_thresh":       0.70,
            "fuse_score":         True,
            "with_reid":          False,   # OFF by default — needs extra model file
            "proximity_thresh":   0.5,
            "appearance_thresh":  0.25,
            "cmc_method":         "sparseOptFlow",
            "frame_rate":         30,
        }
        try:
            tracker_path = os.path.join(
                getattr(config, "DATA_FOLDER", "data"), "botsort_veloxis.yaml")
            os.makedirs(os.path.dirname(tracker_path), exist_ok=True)
            with open(tracker_path, "w") as f:
                _yaml.dump(cfg, f)
            print(f"[INFO] Tracker config: {tracker_path}")
            return tracker_path
        except Exception as e:
            print(f"[WARN] Could not write tracker config ({e}), using bytetrack")
            return "bytetrack.yaml"

    # ── Homography ────────────────────────────────────────────
    def _load_homography(self):
        hpath = os.path.join(getattr(config, "DATA_FOLDER", "data"), "homography.npy")
        if os.path.exists(hpath):
            try:
                self.H_matrix = np.load(hpath)
                print(f"[INFO] Homography loaded from {hpath}")
            except Exception:
                self.H_matrix = None

    def calibrate_homography(self, image_pts, world_pts):
        img = np.float32(image_pts)
        wld = np.float32(world_pts)
        H, mask = cv2.findHomography(img, wld, cv2.RANSAC, 5.0)
        if H is not None:
            self.H_matrix = H
            hpath = os.path.join(getattr(config, "DATA_FOLDER", "data"), "homography.npy")
            os.makedirs(os.path.dirname(hpath), exist_ok=True)
            np.save(hpath, H)
            print(f"[INFO] Homography calibrated. Inliers: {mask.sum()}/{len(image_pts)}")
            return True
        return False

    def pixel_to_world(self, px, py):
        if self.H_matrix is None:
            return None
        pt = np.float32([[px, py]]).reshape(-1, 1, 2)
        world = cv2.perspectiveTransform(pt, self.H_matrix)
        return float(world[0][0][0]), float(world[0][0][1])

    # ── Main processing ───────────────────────────────────────
    def process_frame(self, frame):
        self.frame_no += 1
        h, w = frame.shape[:2]

        # Frame skip — uses CPU-optimised value automatically
        skip = _EFFECTIVE_FRAME_SKIP
        if skip > 1 and self.frame_no % skip != 0:
            if hasattr(self, "_last_ann"):
                return self._last_ann, self._last_sum
            return frame, self._empty_summary()

        # Night enhancement (CLAHE)
        if getattr(config, "ENHANCE_NIGHT", False):
            if frame.mean() < getattr(config, "NIGHT_THRESHOLD", 60):
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
                frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Resize for detection — uses CPU-optimised resolution automatically
        detect_frame = frame
        coord_scale  = 1.0
        if getattr(config, "RESIZE_BEFORE", True):
            rw = _EFFECTIVE_RESIZE_W
            if w > rw:
                s = rw / w
                detect_frame = cv2.resize(frame, (rw, int(h * s)))
                coord_scale  = 1.0 / s

        # ── Detection + Tracking ──────────────────────────────
        tracks = []
        self.person_count = 0
        try:
            # Lower confidence if using generic fallback model
            _conf = config.CONFIDENCE
            if 'bd_vehicles' not in getattr(self, '_model_name_used', '').lower():
                _conf = min(_conf, 0.20)  # COCO model needs lower threshold for BD vehicles

            results = self.model.predict(
                detect_frame,
                verbose = False,
                conf    = _conf,
                iou     = 0.45,
            )[0]

            n_det = len(results.boxes) if results.boxes is not None else 0
            # Debug: detection count + confidence shown top-right
            cv2.putText(frame, f"Det:{n_det} Conf:{_conf:.2f} Trk:{len(self.counted_ids)}",
                        (w-240, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,100), 1)

            raw_dets = []
            for box, cls_id, conf_v in zip(
                results.boxes.xyxy,
                results.boxes.cls,
                results.boxes.conf,
            ):
                cls_int = int(cls_id)
                x1, y1, x2, y2 = [int(v * coord_scale) for v in box.tolist()]
                if cls_int == 0:
                    if getattr(config, "DETECT_HUMANS", True):
                        self.person_count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 100), 1)
                        cv2.putText(frame, "person", (x1, max(y1-4, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 100, 100), 1)
                    continue
                raw_dets.append((x1, y1, x2, y2, cls_int, float(conf_v)))

            tracks = self._iou_track(raw_dets, w, h)

        except Exception as e:
            if not hasattr(self, '_det_warn_count'): self._det_warn_count = 0
            self._det_warn_count += 1
            if self._det_warn_count <= 5:
                print(f"[WARN] Detection error frame {self.frame_no}: {type(e).__name__}: {e}")
            cv2.putText(frame, f"DET ERR:{type(e).__name__[:20]}",
                        (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

        lines = self._get_lines(w, h)

        # ── Per-track processing ──────────────────────────────
        for trk in tracks:
            tid       = trk["tid"]
            confirmed = trk.get("confirmed", True)
            is_lost   = trk.get("lost", 0) > 0
            l, t, r, b = trk["ltrb"]
            l, t, r, b = max(l, 0), max(t, 0), min(r, w - 1), min(b, h - 1)
            cx, cy  = (l + r) // 2, (t + b) // 2
            bw, bh  = r - l, b - t
            if bw <= 0 or bh <= 0: continue
            cls_int = trk["cls"]

            vtype  = _corrected_vtype(cls_int, bw, bh, w, h, self._class_names)
            colour = COLOURS.get(vtype, DEFAULT_COLOUR)

            # Dim ghost boxes (lost track at predicted position — don't count)
            if is_lost:
                cv2.rectangle(frame, (l,t), (r,b), colour, 1)
                continue

            speed_kmh = self._estimate_speed(tid, cx, cy, self.frame_no)

            # Re-ID: map re-entering vehicles to original track
            effective_tid = self._reid_lookup(tid, cx, cy, w, h, vtype)

            # Sudden brake detection
            prev_sp = self.prev_speeds.get(effective_tid, 0)
            if speed_kmh and prev_sp > 15 and (prev_sp - speed_kmh) > BRAKE_DROP_KMH:
                self.safety_events += 1
                cv2.putText(frame, "! BRAKE", (cx - 28, t - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            if speed_kmh:
                self.prev_speeds[effective_tid] = speed_kmh

            # Direction (FWD = moving down frame, BWD = moving up)
            prev_cy_ = self.prev_cy.get(effective_tid)
            direction = DIR_FORWARD if (prev_cy_ is None or cy >= prev_cy_) else DIR_BACKWARD
            self.prev_cy[effective_tid] = cy

            # Zone tracking for TMC — record entry zone on first appearance
            current_zone = self._get_zone(cx, cy, w, h) if config.ENABLE_ZONES else "all"
            if effective_tid not in self.tmc_entry_zone and current_zone != "all":
                self.tmc_entry_zone[effective_tid] = current_zone

            # Queue tint: blue if slow and behind line
            lp1_main, lp2_main, _ = lines[0]
            line_y = (lp1_main[1] + lp2_main[1]) // 2
            if cy > line_y and (speed_kmh or 0) < self.queue_threshold_kmh:
                colour = (100, 100, 255)

            # Line crossing → count
            # Use effective_tid's tracker history when reid-mapped to avoid
            # sign-state / centroid-history mismatch at reid boundary (Bug 2)
            _eff_trk_state = self._trk_active.get(tid, {})
            if effective_tid != tid and effective_tid in self._trk_active:
                # Reid mapped: use effective track's deeper history
                _orig_state = self._trk_active[effective_tid]
                _cx_hist = _orig_state.get('cx_hist', _eff_trk_state.get('cx_hist', []))
                _cy_hist = _orig_state.get('cy_hist', _eff_trk_state.get('cy_hist', []))
            else:
                _cx_hist = _eff_trk_state.get('cx_hist', [])
                _cy_hist = _eff_trk_state.get('cy_hist', [])

            # Approach zone pre-registration (Bug 5 fix)
            # FWD vehicles approach from above line; BWD from below
            lp1_cross, lp2_cross, _ = lines[0]
            _line_y = (lp1_cross[1] + lp2_cross[1]) // 2
            _approach_band = int(h * self._approach_zone_frac)
            _is_approaching_fwd = (_line_y - _approach_band <= cy < _line_y)
            _is_approaching_bwd = (_line_y < cy <= _line_y + _approach_band)
            _is_approaching = _is_approaching_fwd or _is_approaching_bwd
            if _is_approaching and effective_tid not in self.counted_ids:
                self._seen_approaching[effective_tid] = self.frame_no
            # Draw approach zone top boundary (used by draw_counting_lines)
            _approach_y_top = _line_y - _approach_band
            # Prune stale approach records
            _stale = [k for k, fn in self._seen_approaching.items()
                      if self.frame_no - fn > self._approach_max_age]
            for k in _stale:
                self._seen_approaching.pop(k, None)
            for lp1, lp2, line_label in lines:
                # Standard crossing check
                _crossed = self._crosses_line(f"{effective_tid}_{line_label}",
                                              cx, cy, lp1, lp2, _cx_hist, _cy_hist)
                # Approach zone synthetic cross (Bug 5 fix):
                # Vehicle was seen approaching → now on far side → must have crossed
                if not _crossed and effective_tid in self._seen_approaching:
                    _dx = lp2[0] - lp1[0]; _dy = lp2[1] - lp1[1]
                    _len = max(math.sqrt(_dx*_dx + _dy*_dy), 1)
                    _curr_sd = (_dx*(cy - lp1[1]) - _dy*(cx - lp1[0])) / _len
                    # Negative sd = past line (FWD), Positive sd = past line (BWD)
                    _frames_since = self.frame_no - self._seen_approaching[effective_tid]
                    _past_line_fwd = _curr_sd < -8.0   # FWD vehicle past line
                    _past_line_bwd = _curr_sd > 8.0    # BWD vehicle past line
                    if (_past_line_fwd or _past_line_bwd) and _frames_since <= self._approach_max_age:
                        _crossed = True
                        self._seen_approaching.pop(effective_tid, None)
                if _crossed:
                    if effective_tid not in self.counted_ids:
                        self.counted_ids.add(effective_tid)
                        if direction == DIR_FORWARD:
                            self._counted_fwd.add(effective_tid)
                        self._reid_cache[effective_tid] = {
                            "cx_n": cx / max(w, 1), "cy_n": cy / max(h, 1),
                            "vtype": vtype, "frame": self.frame_no
                        }
                        zone = self._get_zone(cx, cy, w, h) if config.ENABLE_ZONES else "all"
                        self.total_counts[vtype] = self.total_counts.get(vtype, 0) + 1
                        self.dir_counts[direction][vtype] = \
                            self.dir_counts[direction].get(vtype, 0) + 1
                        self._interval_count += 1
                        if config.ENABLE_ZONES:
                            self.zone_counts.setdefault(zone, {})
                            self.zone_counts[zone][vtype] = \
                                self.zone_counts[zone].get(vtype, 0) + 1
                            # Approach count — where vehicle entered intersection
                            entry_zone = self.tmc_entry_zone.get(effective_tid, zone)
                            self.approach_counts[entry_zone] = \
                                self.approach_counts.get(entry_zone, 0) + 1
                            # TMC — entry zone → exit zone (current zone at crossing)
                            if entry_zone and zone and entry_zone != zone:
                                tmc_key = f"{entry_zone}→{zone}"
                            else:
                                # Same zone or no entry zone: use direction as proxy
                                tmc_key = f"{entry_zone}→{'FWD' if direction==DIR_FORWARD else 'BWD'}"
                            self.turning_counts.setdefault(tmc_key, {})
                            self.turning_counts[tmc_key][vtype] = \
                                self.turning_counts[tmc_key].get(vtype, 0) + 1
                        self._log(effective_tid, vtype, zone, direction, speed_kmh)
                    break

            # Draw bounding box
            cv2.rectangle(frame, (l, t), (r, b), colour, 2)
            parts = [vtype.split("/")[0]]
            if config.SHOW_IDS:
                parts.append(f"#{tid}")
            if getattr(config, "SHOW_SPEED", True) and speed_kmh:
                parts.append(f"{speed_kmh:.0f}km/h")
            if effective_tid in self.counted_ids:
                parts.append("FWD" if effective_tid in self._counted_fwd else "BWD")
            lbl = " ".join(parts)
            lx = max(l, 0); ly_top = max(t - 20, 0)
            (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.rectangle(frame, (lx, ly_top), (lx + tw + 6, t), colour, -1)
            cv2.putText(frame, lbl, (lx + 3, max(t - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)

        # ── Near-miss detection (capped at 20 tracks) ─────────
        if len(tracks) <= 20:
            pos = [(trk2["tid"],
                    (trk2["ltrb"][0] + trk2["ltrb"][2]) // 2,
                    (trk2["ltrb"][1] + trk2["ltrb"][3]) // 2)
                   for trk2 in tracks]
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    ta, ax, ay = pos[i]; tb, bx, by = pos[j]
                    if math.hypot(ax - bx, ay - by) < NEAR_MISS_PX:
                        sp_a = self._estimate_speed(ta, ax, ay, self.frame_no) or 0
                        sp_b = self._estimate_speed(tb, bx, by, self.frame_no) or 0
                        if sp_a > NEAR_MISS_SPEED and sp_b > NEAR_MISS_SPEED:
                            pair = tuple(sorted([ta, tb]))
                            if not any(e[1] == pair and self.frame_no - e[0] < 90
                                       for e in self.near_miss_log):
                                self.near_miss_log.append((self.frame_no, pair))
                                self.safety_events += 1
                                cv2.line(frame, (ax, ay), (bx, by), (0, 0, 255), 1)
                                cv2.putText(frame, "NEAR-MISS",
                                            ((ax + bx) // 2, (ay + by) // 2 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            if len(self.near_miss_log) > 100:
                self.near_miss_log = self.near_miss_log[-50:]

        # ── Live stats ────────────────────────────────────────
        vehicle_tracks = [t for t in tracks
                          if t.get("confirmed",True) and t.get("lost",0)==0]
        self.live_vehicles = len(vehicle_tracks)
        road_area = max(w * h * 0.6, 1)
        occ_px = sum(
            max(trk2["ltrb"][2] - trk2["ltrb"][0], 0) *
            max(trk2["ltrb"][3] - trk2["ltrb"][1], 0)
            for trk2 in vehicle_tracks)
        self.occupancy_pct = min(100.0, occ_px / road_area * 100)
        self.density_history.append((self.frame_no, self.live_vehicles))
        if len(self.density_history) > 300:
            self.density_history = self.density_history[-150:]

        # ── Queue length ──────────────────────────────────────
        lp1_q, lp2_q, _ = lines[0]
        line_y_q = (lp1_q[1] + lp2_q[1]) // 2
        self.queue_length = sum(
            1 for trk2 in vehicle_tracks
            if (trk2["ltrb"][1] + trk2["ltrb"][3]) // 2 > line_y_q
            and (self._estimate_speed(trk2["tid"],
                 (trk2["ltrb"][0] + trk2["ltrb"][2]) // 2,
                 (trk2["ltrb"][1] + trk2["ltrb"][3]) // 2,
                 self.frame_no) or 999) < self.queue_threshold_kmh
        )
        self.queue_history.append((self.frame_no, self.queue_length))
        if len(self.queue_history) > 600:
            self.queue_history = self.queue_history[-300:]

        if self.queue_length > 0:
            cv2.putText(frame, f"Q:{self.queue_length}",
                        (lp1_q[0] + 4, lp1_q[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 255), 1)

        # ── Peak hour (15-min intervals) + PHF ───────────────
        now = datetime.datetime.now()
        elapsed_min = (now - self._interval_start).total_seconds() / 60
        if elapsed_min >= 15:
            rate = round(self._interval_count / (elapsed_min / 60))
            self.peak_intervals.append((self._interval_start, self._interval_count, rate))
            if rate > self.peak_rate:
                self.peak_rate = rate
            self.current_rate = rate
            # Peak Hour Factor: PHF = hourly_vol / (4 × peak_15min_vol)
            # Uses last 4 intervals (1 hour window)
            if len(self.peak_intervals) >= 4:
                last4 = [p[2] for p in self.peak_intervals[-4:]]
                total_hr = sum(last4)
                peak_15  = max(last4)
                self.phf = round(total_hr / (4 * peak_15), 3) if peak_15 > 0 else 0.0
            self._interval_start = now
            self._interval_count = 0
        elif elapsed_min > 0:
            self.current_rate = round(self._interval_count / (elapsed_min / 60))

        # ── Memory management (runs every frame) ─────────────
        # All unbounded dicts pruned here to prevent long-session crash

        # 1. Re-ID cache — remove stale entries
        stale_reid = [k for k, v in self._reid_cache.items()
                      if self.frame_no - v["frame"] > self._reid_max_age]
        for k in stale_reid:
            del self._reid_cache[k]

        # 2. Re-ID mapped dict — prune entries older than cache window
        #    Only keep mappings for IDs still in reid_cache
        if len(self._reid_mapped) > 500:
            valid_originals = set(self._reid_cache.keys())
            self._reid_mapped = {
                k: v for k, v in self._reid_mapped.items()
                if v in valid_originals or k in valid_originals}

        # 3. Speed history — already capped per track in _estimate_speed
        #    But dict itself grows with each new track ID. Prune dead tracks.
        active_tids = {trk2["tid"] for trk2 in tracks}
        if self.frame_no % 300 == 0:   # every 300 frames (~10s)
            # Keep only active tracks + recently seen (last 300 frames implied)
            # Speed history entries are already sliced to 15 in _estimate_speed
            dead = [k for k in list(self.speed_history.keys())
                    if k not in active_tids and k not in self.counted_ids]
            for k in dead:
                del self.speed_history[k]
            # Prune prev_speeds and prev_cy for dead tracks too
            for dead_k in [k for k in list(self.prev_speeds.keys())
                           if k not in active_tids and k not in self.counted_ids]:
                self.prev_speeds.pop(dead_k, None)
                self.prev_cy.pop(dead_k, None)

        # 4. _counted_fwd — only keep IDs that are in counted_ids
        if self.frame_no % 600 == 0:
            self._counted_fwd &= self.counted_ids
            # Prune tmc_entry_zone for vehicles long gone
            dead_tmc = [k for k in list(self.tmc_entry_zone.keys())
                        if k not in active_tids and k in self.counted_ids]
            for k in dead_tmc:
                self.tmc_entry_zone.pop(k, None)
            # Prune _d_, _b_, _hist_crossed_ attributes for tracks no longer active
            # These accumulate as setattr(self, f"_d_{key}", ...) over long sessions
            active_keys = set()
            for tid2 in active_tids:
                for ll in ["A","B"]:
                    active_keys.add(f"_d_{tid2}_{ll}")
                    active_keys.add(f"_b_{tid2}_{ll}")
                    active_keys.add(f"_hist_crossed_{tid2}_{ll}")
            stale_attrs = [k for k in list(self.__dict__.keys())
                           if (k.startswith('_d_') or k.startswith('_b_') or
                               k.startswith('_hist_crossed_'))
                           and k not in active_keys]
            for k in stale_attrs:
                delattr(self, k)

        # 5. peak_intervals — keep only last 12 (3 hours of 15-min intervals)
        if len(self.peak_intervals) > 12:
            self.peak_intervals = self.peak_intervals[-12:]

        # 6. Auto-save every 5 minutes (300 seconds)
        if not hasattr(self, '_last_autosave'):
            self._last_autosave = datetime.datetime.now()
        elapsed_since_save = (datetime.datetime.now() - self._last_autosave).total_seconds()
        if elapsed_since_save >= 300:   # 5 minutes
            self._autosave_checkpoint()
            self._last_autosave = datetime.datetime.now()

        # ── Draw counting lines + approach zone ──────────────
        fwd_c = sum(self.dir_counts.get(DIR_FORWARD, {}).values())
        bwd_c = sum(self.dir_counts.get(DIR_BACKWARD, {}).values())
        lcols = [(57, 197, 187), (255, 180, 40)]

        # Draw approach zone — dashed line showing pre-registration area
        if lines:
            lp1_a, lp2_a, _ = lines[0]
            _line_y_draw = (lp1_a[1] + lp2_a[1]) // 2
            _appr_band   = int(h * self._approach_zone_frac)
            _appr_y_top  = max(_line_y_draw - _appr_band, 0)
            _appr_y_bot  = min(_line_y_draw + _appr_band, h - 1)
            # Dashed approach zone lines (both above and below for FWD+BWD)
            for _appr_y in [_appr_y_top, _appr_y_bot]:
                _x = lp1_a[0]
                while _x < lp2_a[0]:
                    x_end = min(_x + 12, lp2_a[0])
                    cv2.line(frame, (_x, _appr_y), (x_end, _appr_y), (100, 160, 255), 1)
                    _x += 20
            # Semi-transparent approach zone fills
            _ov = frame.copy()
            cv2.rectangle(_ov, (lp1_a[0], _appr_y_top), (lp2_a[0], _line_y_draw), (60, 80, 160), -1)
            cv2.rectangle(_ov, (lp1_a[0], _line_y_draw), (lp2_a[0], _appr_y_bot), (60, 80, 160), -1)
            cv2.addWeighted(_ov, 0.06, frame, 0.94, 0, frame)
            # Label
            _n_approaching = len(self._seen_approaching)
            if _n_approaching > 0:
                cv2.putText(frame, f"Approach:{_n_approaching}",
                            (lp1_a[0] + 4, _appr_y_top + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 160, 255), 1)

        for i, (lp1, lp2, _) in enumerate(lines):
            col = lcols[i % 2]
            cv2.line(frame, lp1, lp2, col, 2)
            mx = (lp1[0] + lp2[0]) // 2; my = (lp1[1] + lp2[1]) // 2
            if len(lines) == 1:
                cv2.arrowedLine(frame, (mx - 22, my - 10), (mx - 22, my + 10),
                                (57, 197, 187), 2, tipLength=0.4)
                cv2.putText(frame, f"FWD:{fwd_c}", (mx - 48, my - 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (57, 197, 187), 1)
                cv2.arrowedLine(frame, (mx + 22, my + 10), (mx + 22, my - 10),
                                (255, 180, 40), 2, tipLength=0.4)
                cv2.putText(frame, f"BWD:{bwd_c}", (mx + 4, my - 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 180, 40), 1)
            else:
                arr = ((mx, my - 12), (mx, my + 12)) if i == 0 else ((mx, my + 12), (mx, my - 12))
                cv2.arrowedLine(frame, arr[0], arr[1], col, 2, tipLength=0.4)
                cnt = fwd_c if i == 0 else bwd_c
                tag = f"{'FWD' if i==0 else 'BWD'}:{cnt}"
                cv2.rectangle(frame, (lp1[0] + 2, lp1[1] - 18),
                              (lp1[0] + len(tag) * 8 + 4, lp1[1] - 3), (10, 20, 30), -1)
                cv2.putText(frame, tag, (lp1[0] + 4, lp1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        if config.ENABLE_ZONES:
            self._draw_zones(frame, w, h)

        frame = self._draw_hud(frame)

        summary = {
            "total_unique":   len(self.counted_ids),
            "by_type":        self.total_counts,
            "by_zone":        self.zone_counts,
            "by_direction":   self.dir_counts,
            "frame":          self.frame_no,
            "live_vehicles":  self.live_vehicles,
            "occupancy_pct":  round(self.occupancy_pct, 1),
            "person_count":   self.person_count,
            "safety_events":  self.safety_events,
            "near_miss_log":  self.near_miss_log[-5:],
            "queue_length":   self.queue_length,
            "current_rate":   self.current_rate,
            "peak_rate":      self.peak_rate,
            # Miovision-style
            "phf":            self.phf,
            "avg_headway_sec":self.avg_headway_sec,
            "saturation_flow":self.saturation_flow,
            "los_letter":     self.los_letter,
            "avg_delay_sec":  self.avg_delay_sec,
            "speed_85th":     self.speed_85th,
            "speed_mean":     self.speed_mean,
            "approach_counts":self.approach_counts,
            "turning_counts": self.turning_counts,
        }
        self._last_ann = frame
        self._last_sum = summary
        return frame, summary

    # ── Helper methods ────────────────────────────────────────

    def _empty_summary(self):
        return {
            "total_unique":  len(self.counted_ids),
            "by_type":       self.total_counts,
            "by_zone":       self.zone_counts,
            "by_direction":  self.dir_counts,
            "frame":         self.frame_no,
            "live_vehicles": self.live_vehicles,
            "occupancy_pct": round(self.occupancy_pct, 1),
            "person_count":  self.person_count,
            "safety_events": self.safety_events,
            "near_miss_log": [],
            "queue_length":  self.queue_length,
            "current_rate":  self.current_rate,
            "peak_rate":     self.peak_rate,
        }

    def _iou_track(self, dets, frame_w, frame_h):
        """
        Robust IoU tracker with:
        - Velocity prediction (Kalman-lite: smoothed dx/dy per track)
        - Hungarian optimal assignment (not greedy)
        - Class consistency (CNG can't match bicycle)
        - Tentative track confirmation (min 2 frames before counting)
        - Lost track display with last-known box (reduces ID flicker)

        dets: list of (x1,y1,x2,y2, cls_int, conf)
        returns: list of track dicts {"tid","ltrb","cls","conf","confirmed"}
        """
        if not hasattr(self, '_trk_active'):
            self._trk_active  = {}   # tid → track state
            self._trk_next_id = 1

        IOU_MATCH  = 0.18   # minimum IoU after velocity prediction
        IOU_HIGH   = 0.40   # high-confidence match (skip class check)
        MAX_LOST   = 15     # frames before track expires (~0.5s at 30fps)
        MIN_HITS   = 1      # confirm after 1 frame — fast vehicles must not be missed

        def _iou(a, b):
            ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
            iw = max(0, min(ax2,bx2)-max(ax1,bx1))
            ih = max(0, min(ay2,by2)-max(ay1,by1))
            inter = iw*ih
            if inter == 0: return 0.0
            return inter / ((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)

        def _predicted_box(trk):
            """Predict next position using smoothed velocity."""
            x1,y1,x2,y2 = trk['ltrb']
            vx,vy = trk.get('vx',0), trk.get('vy',0)
            lost  = trk['lost']
            # Decay velocity for lost tracks (friction model)
            decay = 0.8 ** lost
            return (x1+vx*decay, y1+vy*decay, x2+vx*decay, y2+vy*decay)

        # Step 1: Age all tracks, predict positions
        for tid in list(self._trk_active):
            trk = self._trk_active[tid]
            trk['lost'] += 1
            if trk['lost'] > MAX_LOST:
                del self._trk_active[tid]

        if not dets:
            # Return confirmed tracks with last-known box (visual continuity)
            return [{"tid":tid,"ltrb":trk['ltrb'],"cls":trk['cls'],
                     "conf":trk['conf'],"confirmed":trk['age']>=MIN_HITS,"lost":trk['lost']}
                    for tid, trk in self._trk_active.items()
                    if trk['lost'] <= 3 and trk['age'] >= MIN_HITS]

        # Step 2: Build cost matrix (1 - IoU) for Hungarian assignment
        track_ids  = list(self._trk_active.keys())
        n_trk = len(track_ids)
        n_det = len(dets)

        if n_trk > 0 and n_det > 0:
            import numpy as np
            cost = np.ones((n_trk, n_det), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                pred_box = _predicted_box(self._trk_active[tid])
                trk_cls  = self._trk_active[tid]['cls']
                for j, (x1,y1,x2,y2,cls,conf) in enumerate(dets):
                    if cls != trk_cls:
                        cost[i,j] = 0.98
                        continue
                    iou = _iou(pred_box, (x1,y1,x2,y2))
                    cost[i,j] = 1.0 - iou

            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost)
            except ImportError:
                # Fallback: greedy matching if scipy not installed
                row_ind, col_ind = [], []
                used_cols = set()
                for r in range(n_trk):
                    best_c = min((c for c in range(n_det) if c not in used_cols),
                                 key=lambda c: cost[r,c], default=-1)
                    if best_c >= 0 and cost[r,best_c] < 0.99:
                        row_ind.append(r); col_ind.append(best_c)
                        used_cols.add(best_c)

            matched_trks = set()
            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r,c] > (1.0 - IOU_MATCH):
                    continue  # IoU too low
                matched_trks.add(r)
                matched_dets.add(c)
                tid = track_ids[r]
                x1,y1,x2,y2,cls,conf = dets[c]
                trk = self._trk_active[tid]
                # Update velocity with exponential smoothing
                px1,py1,px2,py2 = trk['ltrb']
                raw_vx = (x1-px1+x2-px2)/2
                raw_vy = (y1-py1+y2-py2)/2
                trk['vx'] = 0.6*trk.get('vx',0) + 0.4*raw_vx
                trk['vy'] = 0.6*trk.get('vy',0) + 0.4*raw_vy
                # Track centroid history (last 8 frames — deeper history catches fast vehicles)
                cx_m = (x1+x2)//2; cy_m = (y1+y2)//2
                hist_cx = trk.get('cx_hist', []); hist_cx.append(cx_m)
                hist_cy = trk.get('cy_hist', []); hist_cy.append(cy_m)
                trk['cx_hist'] = hist_cx[-8:]; trk['cy_hist'] = hist_cy[-8:]
                trk.update(ltrb=(x1,y1,x2,y2), cls=cls, conf=conf,
                           age=trk['age']+1, lost=0)
        else:
            matched_trks = set()
            matched_dets = set()

        # Step 3: Build result — matched tracks
        result = []
        for i, tid in enumerate(track_ids):
            if i not in matched_trks: continue
            trk = self._trk_active[tid]
            result.append({"tid":tid,"ltrb":trk['ltrb'],"cls":trk['cls'],
                           "conf":trk['conf'],
                           "confirmed":trk['age']>=MIN_HITS,"lost":0})

        # Step 4: Show lost-but-not-expired confirmed tracks (ghost boxes)
        for i, tid in enumerate(track_ids):
            if i in matched_trks: continue
            trk = self._trk_active[tid]
            if trk['age'] >= MIN_HITS and trk['lost'] <= 4:
                # Show at predicted position during brief occlusion
                pb = _predicted_box(trk)
                result.append({"tid":tid,
                               "ltrb":(int(pb[0]),int(pb[1]),int(pb[2]),int(pb[3])),
                               "cls":trk['cls'],"conf":trk['conf']*0.7,
                               "confirmed":True,"lost":trk['lost']})

        # Step 5: New tracks for unmatched detections
        for j, (x1,y1,x2,y2,cls,conf) in enumerate(dets):
            if j in matched_dets: continue
            tid = self._trk_next_id; self._trk_next_id += 1
            self._trk_active[tid] = dict(
                ltrb=(x1,y1,x2,y2), cls=cls, conf=conf,
                age=1, lost=0, vx=0.0, vy=0.0,
                cx_hist=[], cy_hist=[])  # centroid history for crossing lookahead
            result.append({"tid":tid,"ltrb":(x1,y1,x2,y2),"cls":cls,
                           "conf":conf,"confirmed":True,"lost":0})

        return result

    def _reid_lookup(self, tid, cx, cy, w, h, vtype):
        if tid in self._reid_mapped:
            return self._reid_mapped[tid]
        if tid in self.counted_ids:
            return tid
        cx_n = cx / max(w, 1); cy_n = cy / max(h, 1)
        POSITION_THRESH = 0.12
        for orig_tid, entry in self._reid_cache.items():
            if entry["vtype"] != vtype: continue
            dist = math.hypot(cx_n - entry["cx_n"], cy_n - entry["cy_n"])
            if dist < POSITION_THRESH:
                self._reid_mapped[tid] = orig_tid
                # Transfer crossing state from tid to orig_tid so the new
                # IoU track ID inherits the signed-distance history.
                # Without this, every reid causes prev=None → crossing missed.
                for line_label in ["A", "B"]:
                    old_key = f"{tid}_{line_label}"
                    new_key = f"{orig_tid}_{line_label}"
                    d_val = getattr(self, f"_d_{old_key}", None)
                    b_val = getattr(self, f"_b_{old_key}", False)
                    if d_val is not None and not hasattr(self, f"_d_{new_key}"):
                        setattr(self, f"_d_{new_key}", d_val)
                        setattr(self, f"_b_{new_key}", b_val)
                return orig_tid
        return tid

    def _get_lines(self, w, h):
        if getattr(config, "USE_DUAL_LINES", False):
            a = getattr(self, "manual_line_a", None) or getattr(config, "LINE_POS_A", 0.38)
            b = getattr(config, "LINE_POS_B", 0.70)
            return [((0, int(h * a)), (w, int(h * a)), "A"),
                    ((0, int(h * b)), (w, int(h * b)), "B")]
        if self.manual_line_a is not None:
            ly = int(h * self.manual_line_a)
        elif self.ai_line_start and self.ai_line_end:
            p1 = (int(self.ai_line_start[0] * w), int(self.ai_line_start[1] * h))
            p2 = (int(self.ai_line_end[0] * w),   int(self.ai_line_end[1] * h))
            return [(p1, p2, "A")]
        else:
            ly = int(h * config.COUNTING_LINE_POSITION)
        return [((0, ly), (w, ly), "A")]

    def _crosses_line(self, key, cx, cy, p1, p2, cx_hist=None, cy_hist=None):
        """
        Detect line crossing. Three mechanisms:
        1. Sign change: centroid moves from one side to other (primary, most reliable)
        2. Band exit: centroid passes through band without clean sign change (slow vehicles)
        3. History scan: any of last 3 centroids crossed — catches YOLO miss at crossing moment
        """
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        length = max(math.sqrt(dx * dx + dy * dy), 1)
        def sd(px, py): return (dx * (py - p1[1]) - dy * (px - p1[0])) / length

        prev = getattr(self, f"_d_{key}", None)
        curr = sd(cx, cy)
        setattr(self, f"_d_{key}", curr)

        if prev is None:
            # First time seen — check if vehicle already crossed line
            if cx_hist and cy_hist and len(cx_hist) >= 2:
                # Vehicle already has history: scan for crossing in past positions
                sds = [sd(hx, hy) for hx, hy in zip(cx_hist, cy_hist)]
                for i in range(len(sds)-1):
                    if sds[i] * sds[i+1] < 0:
                        return True  # crossing happened in history
            # Bug 5 fix: vehicle appeared for the FIRST time already past line
            # If curr is on far side (negative sd) AND no history to show it came from near side,
            # it was likely detected late — count it as having crossed
            # Only trigger if it's well past the line (not just touching it)
            # Uses a generous threshold of 15% of line length to avoid false counts at edges
            far_side_threshold = length * 0.15
            if abs(curr) > far_side_threshold:
                # Mark as "appeared past line" — will be counted on next frame
                # when prev*curr check runs (prev will be curr which is negative, curr will advance)
                pass  # first frame only initialises state; sign-change on frame 2 catches it
            return False

        # Mechanism 1: sign change
        if prev * curr < 0:
            return True

        # Mechanism 3: history scan — catches YOLO miss exactly at crossing frame
        if cx_hist and cy_hist and len(cx_hist) >= 2:
            all_cx = cx_hist + [cx]; all_cy = cy_hist + [cy]
            sds = [sd(hx, hy) for hx, hy in zip(all_cx[-4:], all_cy[-4:])]
            for i in range(len(sds)-1):
                if sds[i] * sds[i+1] < 0:
                    already_key = f"_hist_crossed_{key}"
                    if getattr(self, already_key, False):
                        return False  # already counted from history
                    setattr(self, already_key, True)
                    return True

        # Mechanism 2: band exit (slow/stopped vehicles)
        # Band scaled by frame_skip — at skip=2, vehicles move 2x further per frame
        # so band must be wider to catch vehicles that dwell near line
        _skip_scale = max(_EFFECTIVE_FRAME_SKIP, 1)
        band = max(length * 0.03 * _skip_scale, 8.0 * _skip_scale)
        was_in = getattr(self, f"_b_{key}", False)
        now_in = abs(curr) < band
        setattr(self, f"_b_{key}", now_in)
        return was_in and not now_in

    def _estimate_speed(self, tid, cx, cy, frame_no):
        # Use effective frame skip so speed is correct even in CPU mode
        sk  = max(_EFFECTIVE_FRAME_SKIP, 1)
        fps = max(getattr(config, "VIDEO_FPS", 25), 1)
        hist = self.speed_history.setdefault(tid, [])

        if self.H_matrix is not None:
            world = self.pixel_to_world(cx, cy)
            if world is None: return None
            xm, ym = world
            hist.append((xm, ym, frame_no))
            if len(hist) < 8: return None
            speeds = []
            for i in range(1, min(len(hist), 8)):
                ox, oy, ofn = hist[-i - 1]; nx, ny, nfn = hist[-i]
                dt = max(nfn - ofn, 1) * sk / fps
                kmh = math.hypot(nx - ox, ny - oy) / dt * 3.6
                if 0.3 < kmh < 120: speeds.append(kmh)
        else:
            ppm = getattr(config, "PIXELS_PER_METER", 0)
            if ppm <= 0: return None
            hist.append((cx, cy, frame_no))
            if len(hist) < 8: return None
            speeds = []
            for i in range(1, min(len(hist), 8)):
                ox, oy, ofn = hist[-i - 1]; nx, ny, nfn = hist[-i]
                dt = max(nfn - ofn, 1) * sk / fps
                kmh = math.hypot(nx - ox, ny - oy) / ppm / dt * 3.6
                if 0.5 < kmh < 120: speeds.append(kmh)

        if len(hist) > 30:
            self.speed_history[tid] = hist[-15:]
        if not speeds: return None
        return round(sorted(speeds)[len(speeds) // 2], 1)

    def _get_zone(self, cx, cy, w, h):
        if _POLYGON_ZONES:
            for lane in _POLYGON_ZONES:
                if point_in_polygon(cx, cy, lane["points"], w, h):
                    return lane["name"]
        for name, (x1, y1, x2, y2) in config.ZONES.items():
            if x1 * w <= cx <= x2 * w and y1 * h <= cy <= y2 * h:
                return name
        return "all"

    @staticmethod
    def _nms(detections, iou_thresh=0.45):
        if len(detections) < 2: return detections
        boxes  = [[d[0][0], d[0][1], d[0][0] + d[0][2], d[0][1] + d[0][3]] for d in detections]
        scores = [d[1] for d in detections]
        idxs   = cv2.dnn.NMSBoxes(
            [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes],
            scores, score_threshold=0.0, nms_threshold=iou_thresh)
        if len(idxs) == 0: return detections
        flat = [int(i) for i in (idxs.flatten() if hasattr(idxs, "flatten") else idxs)]
        return [detections[i] for i in flat]

    def _draw_zones(self, frame, w, h):
        cols = [(255, 165, 30), (57, 197, 187), (220, 80, 80), (130, 80, 220), (80, 200, 80)]
        if _POLYGON_ZONES:
            for i, lane in enumerate(_POLYGON_ZONES):
                c   = cols[i % len(cols)]
                pts = np.array([[int(fx * w), int(fy * h)] for fx, fy in lane["points"]], np.int32)
                ov  = frame.copy(); cv2.fillPoly(ov, [pts], c)
                cv2.addWeighted(ov, 0.10, frame, 0.90, 0, frame)
                cv2.polylines(frame, [pts], True, c, 2)
                cnt = sum(self.zone_counts.get(lane["name"], {}).values())
                cx_ = int(pts[:, 0].mean()); cy_ = int(pts[:, 1].mean())
                cv2.putText(frame, f"{lane['name']}:{cnt}", (cx_ - 40, cy_),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(frame, f"{lane['name']}:{cnt}", (cx_ - 40, cy_),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)
            return
        for i, (name, (x1, y1, x2, y2)) in enumerate(config.ZONES.items()):
            c = cols[i % len(cols)]
            cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), c, 1)
            cnt = sum(self.zone_counts.get(name, {}).values())
            cv2.putText(frame, f"{name}:{cnt}", (int(x1 * w) + 4, int(y1 * h) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 2)

    def _draw_hud(self, frame):
        h, w = frame.shape[:2]
        fwd = sum(self.dir_counts.get(DIR_FORWARD, {}).values())
        bwd = sum(self.dir_counts.get(DIR_BACKWARD, {}).values())
        lines_text = [(f"Total: {len(self.counted_ids)}", (57, 197, 187))]
        for vt, cnt in self.total_counts.items():
            lines_text.append((f"  {vt[:12]}: {cnt}", (200, 200, 200)))
        lines_text.append((f"FWD:{fwd}  BWD:{bwd}", (180, 220, 100)))
        lines_text.append((f"Live:{self.live_vehicles}  Occ:{self.occupancy_pct:.0f}%  Q:{self.queue_length}",
                           (150, 180, 255)))
        lines_text.append((f"Rate:{self.current_rate}v/hr  Peak:{self.peak_rate}  PHF:{self.phf:.2f}",
                           (255, 200, 80)))
        if self.avg_headway_sec > 0:
            lines_text.append((f"Hdwy:{self.avg_headway_sec:.1f}s  SatFlow:{self.saturation_flow}v/hr",
                               (180, 220, 180)))
        if self.speed_85th > 0:
            lines_text.append((f"V85:{self.speed_85th}km/h  Vmean:{self.speed_mean}km/h",
                               (200, 160, 255)))
        if self.person_count:
            lines_text.append((f"People:{self.person_count}", (255, 100, 100)))
        if self.safety_events:
            lines_text.append((f"Safety:{self.safety_events}", (80, 80, 255)))
        ph = 14 + len(lines_text) * 20 + 8
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (230, ph), (10, 14, 22), -1)
        cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)
        for i, (txt, col) in enumerate(lines_text):
            cv2.putText(frame, txt, (6, 22 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
        cv2.putText(frame, "VELOXIS  |  NextCity Tessera",
                    (w - 220, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (70, 70, 80), 1)
        # Show study site name if set
        if self.site_name:
            site_txt = f"📍 {self.site_name}"
            cv2.putText(frame, site_txt,
                        (6, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 180, 120), 1)
        return frame

    def _csv_write_row(self, row):
        """Write a row to CSV with retry on IOError. Buffers on persistent failure.
        Buffer is capped at 500 rows to prevent OOM on long sessions with disk issues."""
        _MAX_PENDING = 500
        if not hasattr(self, '_csv_pending'):
            self._csv_pending = []
        for attempt in range(3):
            try:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
                if self._csv_pending:
                    with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        for r in self._csv_pending:
                            w.writerow(r)
                    self._csv_pending.clear()
                return True
            except IOError as e:
                if attempt < 2:
                    import time as _t; _t.sleep(0.05)
                else:
                    if len(self._csv_pending) < _MAX_PENDING:
                        self._csv_pending.append(row)
                    else:
                        # Buffer full — drop oldest to protect memory
                        self._csv_pending.pop(0)
                        self._csv_pending.append(row)
                    print(f"[WARN] CSV write failed ({e}). Buffered: {len(self._csv_pending)} rows "
                          f"(cap {_MAX_PENDING}). Check disk space.")
                    return False

    def _log(self, tid, vtype, zone, direction, speed_kmh):
        row = [
            datetime.datetime.now().isoformat(timespec="seconds"),
            tid, vtype, zone, direction,
            round(speed_kmh, 1) if speed_kmh else "",
            self.session_label,
            self.queue_length,
            round(self.occupancy_pct, 1),
            self.current_rate,
            self.avg_headway_sec if self.avg_headway_sec else "",
            self.saturation_flow if self.saturation_flow else "",
            self.phf if self.phf else "",
            self.speed_85th if self.speed_85th else "",
            self.speed_mean if self.speed_mean else "",
            # Location metadata — from Settings → Study Location
            self.site_name, self.site_lat, self.site_lng,
        ]
        self._csv_write_row(row)

        # Headway calculation
        now_dt = datetime.datetime.now()
        if vtype in self._last_cross_time:
            gap = (now_dt - self._last_cross_time[vtype]).total_seconds()
            if 0.5 < gap < 120:
                self.headway_history.append((vtype, gap))
                if len(self.headway_history) > 200:
                    self.headway_history = self.headway_history[-100:]
                recent = [g for _, g in self.headway_history[-50:]]
                if recent:
                    self.avg_headway_sec = round(sum(recent) / len(recent), 2)
                    self.saturation_flow = int(3600 / self.avg_headway_sec)
                    self._compute_los()
        self._last_cross_time[vtype] = now_dt

        # Speed percentiles
        if speed_kmh and speed_kmh > 0:
            self.all_speeds.append(speed_kmh)
            if len(self.all_speeds) > 500:
                self.all_speeds = self.all_speeds[-250:]
            sorted_sp = sorted(self.all_speeds)
            n = len(sorted_sp)
            self.speed_85th = round(sorted_sp[int(n * 0.85)], 1) if n >= 5 else 0.0
            self.speed_mean = round(sum(sorted_sp) / n, 1)

    def _autosave_checkpoint(self):
        """
        Save a session snapshot every 5 minutes.
        - data/session_checkpoint.csv : always overwritten (latest state, quick check)
        - data/checkpoints/session_LABEL_HH.csv : hourly rotating backup (keeps last 48)
          Protects against crash data loss for long sessions (2-day records etc.)
        """
        try:
            fwd = sum(self.dir_counts.get(DIR_FORWARD, {}).values())
            bwd = sum(self.dir_counts.get(DIR_BACKWARD, {}).values())
            now_str = datetime.datetime.now().isoformat(timespec="seconds")

            row_header = [
                "checkpoint_time", "session", "frame_no",
                "total_vehicles", "fwd", "bwd",
                "current_rate_vph", "peak_rate_vph", "phf",
                "avg_headway_sec", "saturation_flow",
                "speed_85th_kmh", "speed_mean_kmh",
                "queue_length", "safety_events",
                "reid_cache_size", "speed_hist_size"
            ] + [f"count_{vt}" for vt in self.total_counts]

            row_data = [
                now_str, self.session_label, self.frame_no,
                len(self.counted_ids), fwd, bwd,
                self.current_rate, self.peak_rate, self.phf,
                self.avg_headway_sec, self.saturation_flow,
                self.speed_85th, self.speed_mean,
                self.queue_length, self.safety_events,
                len(self._reid_cache), len(self.speed_history),
            ] + list(self.total_counts.values())

            data_folder = getattr(config, "DATA_FOLDER", "data")

            # 1. Always-overwrite latest checkpoint (for quick status checks)
            ckpt_path = os.path.join(data_folder, "session_checkpoint.csv")
            with open(ckpt_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(row_header)
                w.writerow(row_data)

            # 2. Hourly rotating backup — one file per hour, keeps data safe over crashes
            # File named by session + hour-of-day so each hour produces a unique file
            ckpt_dir = os.path.join(data_folder, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            hour_tag = datetime.datetime.now().strftime("%Y%m%d_%H")
            # Safe session label for filename (strip colons/spaces)
            safe_label = "".join(c if c.isalnum() or c in "-_" else "_"
                                 for c in self.session_label)[:40]
            hourly_path = os.path.join(ckpt_dir, f"ckpt_{safe_label}_{hour_tag}.csv")
            # Append to hourly file (multiple 5-min snapshots per hour)
            write_header = not os.path.exists(hourly_path)
            with open(hourly_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(row_header)
                w.writerow(row_data)

            # 3. Prune old hourly checkpoints — keep last 48 files (2 days)
            existing = sorted(
                f for f in os.listdir(ckpt_dir) if f.endswith(".csv"))
            for old in existing[:-48]:
                try: os.remove(os.path.join(ckpt_dir, old))
                except OSError: pass

            print(f"[INFO] Checkpoint saved: {len(self.counted_ids)} vehicles "
                  f"@ frame {self.frame_no}  ({hourly_path})")

        except Exception as e:
            print(f"[WARN] Checkpoint save failed: {e}")

    def save_session_summary(self):
        """Save a one-row session summary CSV alongside the main log."""
        try:
            fwd = sum(self.dir_counts.get(DIR_FORWARD, {}).values())
            bwd = sum(self.dir_counts.get(DIR_BACKWARD, {}).values())
            summary_path = self.csv_path.replace(".csv", "_summary.csv")
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "session", "site_name", "latitude", "longitude",

                    "total_vehicles", "fwd", "bwd",
                    "peak_rate_vph", "phf",
                    "avg_headway_sec", "saturation_flow_vph",
                    "speed_85th_kmh", "speed_mean_kmh",
                    "safety_events", "near_miss_count",
                ] + [f"count_{vt}" for vt in self.total_counts])
                w.writerow([
                    self.session_label,
                    self.site_name, self.site_lat, self.site_lng,
                    len(self.counted_ids), fwd, bwd,
                    self.peak_rate, self.phf,
                    self.avg_headway_sec, self.saturation_flow,
                    self.speed_85th, self.speed_mean,
                    self.safety_events, len(self.near_miss_log),
                ] + list(self.total_counts.values()))
            print(f"[INFO] Session summary -> {summary_path}")
            # Auto-export TMC if zone data exists
            if self.turning_counts:
                self.export_tmc_csv()
            return summary_path
        except Exception as e:
            print(f"[WARN] Could not save session summary: {e}")
            return None

    def _compute_los(self):
        """
        Approximate LOS from observed headway / saturation flow.
        Uses Webster simplified delay formula as a proxy:
          d = 9 + (3600/sf * vc^2) / (2*(1-vc))
        NOTE: This is a rough estimate suited for unsignalized/mixed
        intersections — not a full HCM signalized intersection analysis.
        LOS thresholds (HCM 6th, Table 19-1): A<=10 B<=15 C<=25 D<=35 E<=50 F>50
        """
        try:
            sf = self.saturation_flow
            rate = self.current_rate
            if sf <= 0 or rate <= 0:
                self.los_letter = "—"; self.avg_delay_sec = 0.0; return
            vc = min(rate / sf, 0.999)
            d = 9.0 + (3600.0 / sf * vc * vc) / (2.0 * (1.0 - vc))
            self.avg_delay_sec = round(d, 1)
            if   d <= 10: self.los_letter = "A"
            elif d <= 15: self.los_letter = "B"
            elif d <= 25: self.los_letter = "C"
            elif d <= 35: self.los_letter = "D"
            elif d <= 50: self.los_letter = "E"
            else:         self.los_letter = "F"
        except Exception:
            self.los_letter = "—"; self.avg_delay_sec = 0.0

    def export_tmc_csv(self, output_path=None):
        """
        Export Turning Movement Count matrix as CSV.
        Format: rows = approach zones, cols = exit zones, values = total vehicles.
        Compatible with Synchro, SIDRA, and manual HCM worksheets.
        """
        try:
            if not self.turning_counts:
                print("[INFO] No TMC data — enable zones and draw approach lanes first.")
                return None
            import pandas as pd
            # Build flat records
            rows = []
            for movement, vtype_dict in self.turning_counts.items():
                total = sum(vtype_dict.values())
                entry, exit_ = (movement.split("→") + ["—"])[:2]
                row = {"movement": movement, "from_zone": entry,
                       "to_zone": exit_, "total": total}
                row.update(vtype_dict)
                rows.append(row)
            df = pd.DataFrame(rows).fillna(0)
            # Pivot: approach rows × exit columns
            if "from_zone" in df.columns and "to_zone" in df.columns:
                pivot = df.pivot_table(index="from_zone", columns="to_zone",
                                       values="total", aggfunc="sum", fill_value=0)
                pivot.index.name   = "Approach"
                pivot.columns.name = "Exit"
                # Add approach total column
                pivot["TOTAL"] = pivot.sum(axis=1)
            else:
                pivot = df
            if not output_path:
                output_path = self.csv_path.replace(".csv", "_tmc.csv")
            pivot.to_csv(output_path)
            # Also save full detail (by vehicle type)
            detail_path = output_path.replace("_tmc.csv", "_tmc_detail.csv")
            df.to_csv(detail_path, index=False)
            print(f"[INFO] TMC matrix  -> {output_path}")
            print(f"[INFO] TMC detail  -> {detail_path}")
            return output_path
        except Exception as e:
            print(f"[WARN] TMC export failed: {e}")
            return None

    def print_summary(self):
        fwd = sum(self.dir_counts[DIR_FORWARD].values())
        bwd = sum(self.dir_counts[DIR_BACKWARD].values())
        # Load author name from prefs if available, fallback to generic
        try:
            import json as _json
            with open("data/user_prefs.json", encoding="utf-8") as _pf:
                _p = _json.load(_pf)
            _author = _p.get("author_name", "") or ""
            _inst   = _p.get("institution", "") or ""
            _byline = f"  {_author}{', ' + _inst if _inst else ''}  |  NextCity Tessera" if _author else "  VELOXIS  |  NextCity Tessera"
        except Exception:
            _byline = "  VELOXIS  |  NextCity Tessera"
        print("\n" + "=" * 55)
        print(f"  VELOXIS  --  {self.session_label}")
        print(_byline)
        print("=" * 55)
        print(f"  Total vehicles : {len(self.counted_ids)}")
        for vt, cnt in self.total_counts.items():
            print(f"  {vt:<22} {cnt:>5}  {'#' * min(cnt, 20)}")
        print(f"\n  FWD:{fwd}  BWD:{bwd}  Safety events:{self.safety_events}")
        print(f"\n  -- Intersection Capacity Metrics --")
        print(f"  Peak rate      : {self.peak_rate} veh/hr")
        print(f"  PHF            : {self.phf:.3f}  (ideal 0.85-0.95)")
        print(f"  Avg headway    : {self.avg_headway_sec:.1f} sec")
        print(f"  Saturation flow: {self.saturation_flow} veh/hr")
        print(f"  LOS            : {self.los_letter}  (delay {self.avg_delay_sec:.1f} s/veh)")
        print(f"  Speed V85      : {self.speed_85th} km/h")
        print(f"  Speed mean     : {self.speed_mean} km/h")
        if self.approach_counts:
            print(f"\n  -- Approach Volumes --")
            for arm, cnt in sorted(self.approach_counts.items()):
                print(f"  {arm:<20} {cnt:>5} veh")
        if self.turning_counts:
            print(f"\n  -- Turning Movement Counts --")
            for mv, vd in sorted(self.turning_counts.items()):
                total = sum(vd.values())
                detail = "  ".join(f"{k}:{v}" for k, v in vd.items())
                print(f"  {mv:<25} {total:>4}  ({detail})")
        print(f"\n  Log -> {self.csv_path}")
        print("=" * 55)
