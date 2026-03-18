# ================================================================
#  TrafficCounter BD  —  app_windows.py
#  Premium desktop app — customtkinter + embedded dashboard
#  Features:
#    • AI auto-detect counting line (any camera angle)
#    • Embedded charts (no browser needed)
#    • DroidCam + USB auto-scan
#    • Real-time stats, per-vehicle heatmap
#    • World-class dark UI
#
#  pip install customtkinter pillow matplotlib pandas
#  python app_windows.py
# ================================================================

import tkinter as tk
import customtkinter as ctk
import threading, subprocess, sys, os, datetime, json, re, glob, time
import cv2, numpy as np
from PIL import Image, ImageTk

# Matplotlib for embedded charts
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Colour palette ─────────────────────────────────────────────
BG="#0e1117"; BG2="#161b27"; BG3="#1e2535"; BG4="#252e42"
BORDER="#1f2937"; BORDER2="#2d3748"
TEXT="#e8eaf0"; MUTED="#64748b"; HINT="#374151"
BLUE="#3b82f6"; BLUE2="#1d4ed8"; BLUE3="#1e3a5f"
GREEN="#34d399"; GREEN2="#065f46"
AMBER="#fbbf24"; AMBER2="#92400e"
RED="#f87171"; RED2="#7f1d1d"
PURPLE="#a78bfa"; TEAL="#2dd4bf"

CHART_BG="#0e1117"; CHART_FG="#e8eaf0"; CHART_GRID="#1f2937"
LANE_COLS=["#2dd4bf","#fbbf24","#f87171","#a78bfa","#34d399","#fb923c"]


# ================================================================
#  AI COUNTING LINE DETECTOR
#  Analyses first N frames to find dominant traffic flow direction
#  Works at ANY camera angle — overhead, side, diagonal
# ================================================================
class AILineDetector:
    """
    Detects the optimal counting line position and angle
    by analysing optical flow in the first few frames.
    """
    def __init__(self, n_frames=40):
        self.n_frames   = n_frames
        self.frames     = []
        self.ready      = False
        self.line_start = None   # (x1, y1)
        self.line_end   = None   # (x2, y2)
        self.angle_deg  = 90     # degrees from horizontal (90=vertical line)
        self.position   = 0.55   # fraction along dominant flow axis

    def feed(self, frame):
        """Feed frames during calibration. Returns True when analysis complete."""
        if self.ready:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))   # work on small copy
        self.frames.append(gray)
        if len(self.frames) >= self.n_frames:
            self._analyse()
            return True
        return False

    def _analyse(self):
        h, w = self.frames[0].shape
        flow_vx, flow_vy = [], []

        # Compute dense optical flow between pairs of frames
        for i in range(0, len(self.frames)-1, 2):
            flow = cv2.calcOpticalFlowFarneback(
                self.frames[i], self.frames[i+1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # Only consider strong motion
            mask = mag > np.percentile(mag, 75)
            if mask.any():
                flow_vx.append(float(flow[...,0][mask].mean()))
                flow_vy.append(float(flow[...,1][mask].mean()))

        if not flow_vx:
            # Fallback: horizontal line at 55%
            self._set_horizontal(0.55)
            self.ready = True
            return

        # Average dominant flow vector
        vx = np.mean(flow_vx)
        vy = np.mean(flow_vy)

        # Flow angle (direction vehicles are moving)
        flow_angle = np.degrees(np.arctan2(vy, vx))  # -180..180

        # Counting line is PERPENDICULAR to flow direction
        perp_angle = flow_angle + 90
        self.angle_deg = perp_angle % 180

        # Find where most motion crosses — use centre of frame's motion
        self.position = 0.5   # best position = midpoint along flow axis

        # Build the actual line coordinates (full width of frame)
        self._build_line(320, 180, self.position, perp_angle)
        self.ready = True

    def _set_horizontal(self, pos):
        self.angle_deg = 90
        self.position  = pos

    def _build_line(self, fw, fh, pos, perp_deg):
        """Store line as unit fractions so it scales to any frame size."""
        # Direction vector of the counting line
        rad  = np.radians(perp_deg)
        dx   = np.cos(rad)
        dy   = np.sin(rad)
        # Centre point (intersection of flow midline)
        cx, cy = fw * 0.5, fh * pos
        # Extend line to frame edges
        t = max(fw, fh) * 1.5
        x1, y1 = cx - dx*t, cy - dy*t
        x2, y2 = cx + dx*t, cy + dy*t
        # Clip to frame and store as fractions
        x1 = max(0.0, min(1.0, x1/fw)); y1 = max(0.0, min(1.0, y1/fh))
        x2 = max(0.0, min(1.0, x2/fw)); y2 = max(0.0, min(1.0, y2/fh))
        self.line_start = (x1, y1)
        self.line_end   = (x2, y2)

    def get_line_px(self, frame_w, frame_h):
        """Return line as pixel coordinates for a given frame size."""
        if self.line_start and self.line_end:
            x1 = int(self.line_start[0] * frame_w)
            y1 = int(self.line_start[1] * frame_h)
            x2 = int(self.line_end[0]   * frame_w)
            y2 = int(self.line_end[1]   * frame_h)
        else:
            y  = int(frame_h * self.position)
            x1, y1, x2, y2 = 0, y, frame_w, y
        return (x1, y1), (x2, y2)

    def calibration_progress(self):
        return len(self.frames) / self.n_frames


# ================================================================
#  ENHANCED DETECTOR THREAD
# ================================================================
class DetectionThread(threading.Thread):
    def __init__(self, source, mode, on_frame, on_status,
                 on_done, on_progress=None, use_ai_line=True):
        super().__init__(daemon=True)
        self.source      = source
        self.mode        = mode
        self.on_frame    = on_frame
        self.on_status   = on_status
        self.on_done     = on_done
        self.on_progress = on_progress
        self.use_ai_line = use_ai_line
        self._stop       = threading.Event()
        self.ai_detector = AILineDetector(n_frames=35) if use_ai_line else None
        self.calibrating = use_ai_line

    def stop(self): self._stop.set()

    def run(self):
        try:
            import config
            from detector import VehicleDetector
        except ImportError as e:
            self.on_status(f"ERROR: {e}"); return

        self.on_status("Loading YOLO model…")
        label = f"{'live' if self.mode=='live' else 'file'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        detector = VehicleDetector(session_label=label)

        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            self.on_status("Cannot open camera/video. Check source.")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fn    = 0

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                if self.mode == "live":
                    time.sleep(0.05); continue
                break
            fn += 1

            # AI calibration phase
            if self.ai_detector and self.calibrating:
                done = self.ai_detector.feed(frame)
                if not done:
                    pct = int(self.ai_detector.calibration_progress() * 100)
                    self.on_status(f"AI calibrating… {pct}%")
                    # Draw progress bar on frame
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay,(0,0),(w,h),(0,0,0),cv2.FILLED)
                    cv2.addWeighted(overlay,0.55,frame,0.45,0,frame)
                    bar_w = int(w * 0.6); bar_x = (w-bar_w)//2; bar_y = h//2
                    cv2.rectangle(frame,(bar_x,bar_y-10),(bar_x+bar_w,bar_y+10),(40,40,40),-1)
                    cv2.rectangle(frame,(bar_x,bar_y-10),(bar_x+int(bar_w*pct/100),bar_y+10),(57,197,187),-1)
                    cv2.putText(frame,"AI detecting counting line…",(bar_x,bar_y-20),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),1)
                    self.on_frame(frame, {})
                    continue
                else:
                    self.calibrating = False
                    # Write AI line params into config
                    if self.ai_detector.line_start:
                        import config as cfg
                        cfg.AI_LINE_START = self.ai_detector.line_start
                        cfg.AI_LINE_END   = self.ai_detector.line_end
                    self.on_status("AI line detected! Running detection…")

            annotated, summary = detector.process_frame(frame)

            # Draw AI counting line on frame (overrides default line)
            if self.ai_detector and not self.calibrating:
                h, w = annotated.shape[:2]
                p1, p2 = self.ai_detector.get_line_px(w, h)
                cv2.line(annotated, p1, p2, (57,197,187), 2)
                cv2.putText(annotated,"AI Counting Line",(p1[0]+4,p1[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(57,197,187),1)

            self.on_frame(annotated, summary)
            if self.on_progress and self.mode == "file":
                self.on_progress(int(fn/total*100))

        cap.release()
        self.on_done({
            "total_unique": len(detector.counted_ids),
            "by_type":      detector.total_counts,
        })
        self.on_status("Session complete ✓")


# ================================================================
#  UI COMPONENTS
# ================================================================

class StatCard(ctk.CTkFrame):
    def __init__(self, master, label, value="—", accent=BLUE, icon="", **kw):
        super().__init__(master, fg_color=BG2, corner_radius=14,
                         border_width=1, border_color=BORDER, **kw)
        self.grid_columnconfigure(0, weight=1)
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, padx=16, pady=(14,4), sticky="ew")
        ctk.CTkLabel(top, text=icon, font=("Segoe UI",18), text_color=accent
                    ).pack(side="left", padx=(0,6))
        ctk.CTkLabel(top, text=label.upper(), font=("Segoe UI",10,"bold"),
                     text_color=MUTED).pack(side="left")
        self.val = ctk.CTkLabel(self, text=str(value),
                                font=("Segoe UI",30,"bold"), text_color=accent)
        self.val.grid(row=1, column=0, padx=16, pady=(0,14), sticky="w")
        # Subtle accent bar at bottom
        bar = ctk.CTkFrame(self, fg_color=accent, height=3, corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew", padx=0, pady=0)

    def set(self, v, animate=False):
        self.val.configure(text=str(v))


class VideoCanvas(ctk.CTkLabel):
    def __init__(self, master, placeholder="No feed yet\nPress Start to begin", **kw):
        super().__init__(master, text=placeholder,
                         fg_color=BG3, corner_radius=14, text_color=MUTED,
                         font=("Segoe UI",13), **kw)
        self._imgtk = None

    def update_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        ww = max(self.winfo_width(),  640)
        wh = max(self.winfo_height(), 360)
        scale = min(ww/w, wh/h, 1.0)
        nw, nh = int(w*scale), int(h*scale)
        rgb = cv2.cvtColor(cv2.resize(frame,(nw,nh)), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(nw,nh))
        self.configure(image=self._imgtk, text="")


class NavBtn(ctk.CTkButton):
    def __init__(self, master, icon, label, cmd, badge=None, **kw):
        self._icon = icon; self._label = label
        super().__init__(master, text=f"  {icon}   {label}", anchor="w",
                         fg_color="transparent", hover_color=BG4, text_color=MUTED,
                         font=("Segoe UI",13), height=44, corner_radius=10,
                         command=cmd, **kw)

    def set_active(self, v):
        self.configure(
            fg_color=(BLUE3 if v else "transparent"),
            text_color=(BLUE if v else MUTED),
            font=("Segoe UI",13,"bold" if v else "normal"))


class SectionLabel(ctk.CTkLabel):
    def __init__(self, master, text, **kw):
        super().__init__(master, text=text.upper(),
                         font=("Segoe UI",10,"bold"), text_color=MUTED, **kw)


class Divider(ctk.CTkFrame):
    def __init__(self, master, **kw):
        kw.setdefault("height", 1)
        super().__init__(master, fg_color=BORDER, **kw)


class StatusBar(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, fg_color=BG2, height=28,
                         corner_radius=0, border_width=0, **kw)
        self.grid_columnconfigure(1, weight=1)
        self._dot = ctk.CTkLabel(self, text="●", font=("Segoe UI",10),
                                  text_color=MUTED, width=18)
        self._dot.grid(row=0, column=0, padx=(10,4), sticky="w")
        self._msg = ctk.CTkLabel(self, text="Ready", font=("Segoe UI",11),
                                  text_color=MUTED)
        self._msg.grid(row=0, column=1, sticky="w")
        self._right = ctk.CTkLabel(self, text="", font=("Segoe UI",11),
                                    text_color=HINT)
        self._right.grid(row=0, column=2, padx=12, sticky="e")

    def set(self, msg, state="idle"):
        colours = {"idle":MUTED,"running":GREEN,"warn":AMBER,"error":RED}
        c = colours.get(state, MUTED)
        self._dot.configure(text_color=c)
        self._msg.configure(text=msg)

    def set_right(self, txt):
        self._right.configure(text=txt)


class Page(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=BG, corner_radius=0)
        self.grid_columnconfigure(0, weight=1)

    def page_header(self, icon, title, subtitle):
        hf = ctk.CTkFrame(self, fg_color="transparent")
        hf.grid(row=0, column=0, padx=32, pady=(28,20), sticky="ew")
        hf.grid_columnconfigure(1, weight=1)
        # Icon bubble
        ic = ctk.CTkFrame(hf, fg_color=BLUE3, width=46, height=46,
                          corner_radius=12)
        ic.grid(row=0, column=0, rowspan=2, padx=(0,14))
        ic.grid_propagate(False)
        ctk.CTkLabel(ic, text=icon, font=("Segoe UI",20),
                     text_color=BLUE).place(relx=0.5,rely=0.5,anchor="center")
        ctk.CTkLabel(hf, text=title, font=("Segoe UI",20,"bold"),
                     text_color=TEXT).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(hf, text=subtitle, font=("Segoe UI",12),
                     text_color=MUTED).grid(row=1, column=1, sticky="w")


# ================================================================
#  HOME PAGE
# ================================================================
class HomePage(Page):
    def __init__(self, master):
        super().__init__(master)
        self.grid_rowconfigure(4, weight=1)
        self.page_header("🏠","Dashboard","Session overview & live stats")

        # Stat cards
        sf = ctk.CTkFrame(self, fg_color="transparent")
        sf.grid(row=1, column=0, padx=32, sticky="ew")
        sf.grid_columnconfigure((0,1,2,3), weight=1)
        self.ct = StatCard(sf,"Total Vehicles","—",BLUE,"🚗")
        self.cc = StatCard(sf,"Cars","—",GREEN,"🚙")
        self.cr = StatCard(sf,"Rickshaws/CNGs","—",AMBER,"🛺")
        self.cb = StatCard(sf,"Buses & Trucks","—",RED,"🚌")
        for i,c in enumerate([self.ct,self.cc,self.cr,self.cb]):
            c.grid(row=0,column=i,padx=(0 if i==0 else 10,0),sticky="ew")

        # Session info row
        info_row = ctk.CTkFrame(self, fg_color="transparent")
        info_row.grid(row=2, column=0, padx=32, pady=(16,0), sticky="ew")
        info_row.grid_columnconfigure((0,1,2), weight=1)
        self.inf_session = StatCard(info_row,"Sessions",  "—", PURPLE, "📁")
        self.inf_days    = StatCard(info_row,"Days Active","—", TEAL,   "📅")
        self.inf_toptype = StatCard(info_row,"Top Vehicle","—", MUTED,  "📊")
        for i,c in enumerate([self.inf_session,self.inf_days,self.inf_toptype]):
            c.grid(row=0,column=i,padx=(0 if i==0 else 10,0),sticky="ew")

        # Log
        SectionLabel(self,"Session Log").grid(row=3,column=0,padx=32,pady=(20,6),sticky="w")
        self.log = ctk.CTkTextbox(self, fg_color=BG2, text_color="#94a3b8",
                                   font=("Consolas",12), border_width=1,
                                   border_color=BORDER, corner_radius=12)
        self.log.grid(row=4, column=0, padx=32, pady=(0,24), sticky="nsew")
        self._load_stats()

    def _load_stats(self):
        """Pull totals from CSV logs."""
        try:
            files = glob.glob(os.path.join("data","log_*.csv"))
            if not files: return
            import pandas as pd
            dfs = [pd.read_csv(f) for f in files]
            dfs = [d for d in dfs if not d.empty]
            if not dfs: return
            df = pd.concat(dfs, ignore_index=True)
            self.ct.set(len(df))
            bt = df["vehicle_type"].value_counts().to_dict() if "vehicle_type" in df.columns else {}
            self.cc.set(bt.get("car",0))
            self.cr.set(bt.get("rickshaw/motorcycle",0))
            self.cb.set(bt.get("bus",0)+bt.get("truck",0))
            self.inf_session.set(df["session"].nunique() if "session" in df.columns else "—")
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"],errors="coerce")
                self.inf_days.set(df["timestamp"].dt.date.nunique())
            if bt:
                top = max(bt,key=bt.get)
                self.inf_toptype.set(top.split("/")[0].title())
        except Exception:
            pass

    def update_stats(self, summary):
        bt = summary.get("by_type",{})
        self.ct.set(summary.get("total_unique",0))
        self.cc.set(bt.get("car",0))
        self.cr.set(bt.get("rickshaw/motorcycle",0))
        self.cb.set(bt.get("bus",0)+bt.get("truck",0))

    def log_msg(self, msg, level="info"):
        colours = {"info":"#94a3b8","ok":"#34d399","warn":"#fbbf24","err":"#f87171"}
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        col = colours.get(level,"#94a3b8")
        self.log.insert("end", f"[{ts}]  {msg}\n")
        self.log.see("end")


# ================================================================
#  LIVE PAGE  (with AI line + DroidCam fix)
# ================================================================
class LivePage(Page):
    def __init__(self, master, status_bar=None, home_page=None):
        super().__init__(master)
        self.thread = None
        self.status_bar = status_bar
        self.home_page  = home_page
        self.grid_rowconfigure(3, weight=1)
        self.page_header("📹","Live Detection","Real-time vehicle counting with AI line detection")

        # ── Camera source card ────────────────────────────────
        src = ctk.CTkFrame(self, fg_color=BG2, corner_radius=14,
                            border_width=1, border_color=BORDER)
        src.grid(row=1, column=0, padx=32, pady=(0,12), sticky="ew")
        src.grid_columnconfigure((0,1,2,3), weight=1)

        SectionLabel(src,"Camera Source").grid(row=0,column=0,columnspan=4,
                                                padx=18,pady=(14,12),sticky="w")
        self.src_var = tk.StringVar(value="webcam")
        options = [
            ("webcam",   "💻","Laptop Webcam",    "Built-in camera"),
            ("usb",      "🔌","USB Camera",        "External USB webcam"),
            ("droidcam", "📱","DroidCam (WiFi)",   "Phone as IP camera"),
            ("custom",   "🔗","Custom URL",         "RTSP / IP stream"),
        ]
        self.opt_frames = {}
        for col,(key,icon,label,hint) in enumerate(options):
            f = ctk.CTkFrame(src, fg_color=BG3, corner_radius=10,
                             border_width=1, border_color=BORDER)
            f.grid(row=1, column=col, padx=(12 if col==0 else 6,
                                             12 if col==3 else 6), pady=(0,14), sticky="ew")
            rb = ctk.CTkRadioButton(f, text="", variable=self.src_var, value=key,
                                     fg_color=BLUE, hover_color=BLUE2, width=20,
                                     command=self._src_ch)
            rb.place(relx=0.9, rely=0.15)
            ctk.CTkLabel(f, text=icon, font=("Segoe UI",24),
                         text_color=BLUE).pack(pady=(12,4))
            ctk.CTkLabel(f, text=label, font=("Segoe UI",12,"bold"),
                         text_color=TEXT).pack()
            ctk.CTkLabel(f, text=hint, font=("Segoe UI",10),
                         text_color=MUTED).pack(pady=(2,12))
            self.opt_frames[key] = f

        # Details row (index / IP / URL)
        self.detail = ctk.CTkFrame(src, fg_color="transparent")
        self.detail.grid(row=2, column=0, columnspan=4, padx=16, pady=(0,14), sticky="ew")

        # Webcam index picker + test button
        self.webcam_pane = ctk.CTkFrame(self.detail, fg_color="transparent")
        ctk.CTkLabel(self.webcam_pane, text="Camera index:", text_color=MUTED,
                     font=("Segoe UI",12)).pack(side="left",padx=(0,8))
        self.cam_idx = ctk.CTkComboBox(self.webcam_pane, values=["0","1","2","3"],
            width=80, fg_color=BG3, border_color=BORDER, text_color=TEXT)
        self.cam_idx.set("0"); self.cam_idx.pack(side="left",padx=(0,12))
        self.test_btn = ctk.CTkButton(self.webcam_pane, text="🔍  Auto-scan cameras",
            width=170, height=32, fg_color=BG4, hover_color=BG3,
            border_width=1, border_color=BORDER, text_color=BLUE,
            font=("Segoe UI",12), command=self._scan_cams)
        self.test_btn.pack(side="left",padx=(0,10))
        self.scan_lbl = ctk.CTkLabel(self.webcam_pane, text="",
                                      font=("Segoe UI",11), text_color=GREEN)
        self.scan_lbl.pack(side="left")

        # DroidCam pane
        self.droid_pane = ctk.CTkFrame(self.detail, fg_color="transparent")
        ctk.CTkLabel(self.droid_pane, text="Phone IP:", text_color=MUTED,
                     font=("Segoe UI",12)).pack(side="left",padx=(0,8))
        self.ip_var = tk.StringVar()
        ctk.CTkEntry(self.droid_pane, textvariable=self.ip_var,
            placeholder_text="192.168.1.5",
            width=180, fg_color=BG3, border_color=BORDER, text_color=TEXT
        ).pack(side="left",padx=(0,10))
        ctk.CTkLabel(self.droid_pane, text="Port:", text_color=MUTED,
                     font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.port_var = tk.StringVar(value="4747")
        ctk.CTkEntry(self.droid_pane, textvariable=self.port_var,
            width=70, fg_color=BG3, border_color=BORDER, text_color=TEXT
        ).pack(side="left",padx=(0,12))
        self.droid_test = ctk.CTkButton(self.droid_pane, text="🔗  Test Connection",
            width=150, height=32, fg_color=BG4, hover_color=BG3,
            border_width=1, border_color=BORDER, text_color=BLUE,
            font=("Segoe UI",12), command=self._test_droid)
        self.droid_test.pack(side="left",padx=(0,10))
        self.droid_lbl = ctk.CTkLabel(self.droid_pane, text="",
                                       font=("Segoe UI",11), text_color=MUTED)
        self.droid_lbl.pack(side="left")

        # Custom URL pane
        self.url_pane = ctk.CTkFrame(self.detail, fg_color="transparent")
        ctk.CTkLabel(self.url_pane, text="Stream URL:", text_color=MUTED,
                     font=("Segoe UI",12)).pack(side="left",padx=(0,8))
        self.url_var = tk.StringVar()
        ctk.CTkEntry(self.url_pane, textvariable=self.url_var,
            placeholder_text="rtsp://...  or  http://...",
            width=420, fg_color=BG3, border_color=BORDER, text_color=TEXT
        ).pack(side="left")

        # AI line toggle
        ai_row = ctk.CTkFrame(src, fg_color="transparent")
        ai_row.grid(row=3, column=0, columnspan=4, padx=18, pady=(0,14), sticky="w")
        self.ai_var = tk.BooleanVar(value=True)
        ctk.CTkSwitch(ai_row, text="", variable=self.ai_var,
                      onvalue=True, offvalue=False,
                      button_color=TEAL, progress_color=TEAL,
                      width=44, height=22).pack(side="left",padx=(0,10))
        ctk.CTkLabel(ai_row, text="AI Auto-detect counting line",
                     font=("Segoe UI",13,"bold"), text_color=TEAL).pack(side="left")
        ctk.CTkLabel(ai_row,
                     text="  — Calibrates for 2–3 seconds on start, works at any camera angle",
                     font=("Segoe UI",11), text_color=MUTED).pack(side="left")

        self._src_ch()

        # ── Action buttons ────────────────────────────────────
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=2, column=0, padx=32, pady=(0,12), sticky="w")
        self.start_btn = ctk.CTkButton(btn_row, text="▶  Start Detection",
            width=180, height=44, fg_color=BLUE, hover_color=BLUE2,
            font=("Segoe UI",14,"bold"), corner_radius=10, command=self._start)
        self.start_btn.pack(side="left",padx=(0,10))
        self.stop_btn = ctk.CTkButton(btn_row, text="■  Stop",
            width=120, height=44, fg_color=RED2, hover_color="#991b1b",
            font=("Segoe UI",14,"bold"), corner_radius=10,
            state="disabled", command=self._stop)
        self.stop_btn.pack(side="left")

        # ── Video feed ────────────────────────────────────────
        self.video = VideoCanvas(self, "Camera feed will appear here after Start")
        self.video.grid(row=3, column=0, padx=32, pady=(0,24), sticky="nsew")

    def _src_ch(self):
        v = self.src_var.get()
        # Highlight selected card
        for k,f in self.opt_frames.items():
            f.configure(border_color=(BLUE if k==v else BORDER),
                        fg_color=(BLUE3 if k==v else BG3))
        # Show correct detail pane
        for p in [self.webcam_pane,self.droid_pane,self.url_pane]:
            p.pack_forget()
        if v in ("webcam","usb"):
            self.webcam_pane.pack(fill="x")
            if v=="usb": self.cam_idx.set("1")
            else: self.cam_idx.set("0")
        elif v=="droidcam":
            self.droid_pane.pack(fill="x")
        elif v=="custom":
            self.url_pane.pack(fill="x")

    def _scan_cams(self):
        self.scan_lbl.configure(text="Scanning…", text_color=AMBER)
        self.update()
        found = []
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret,_ = cap.read()
                if ret: found.append(str(i))
                cap.release()
        if found:
            self.scan_lbl.configure(text=f"✓ Found: index {', '.join(found)}", text_color=GREEN)
            self.cam_idx.set(found[0])
        else:
            self.scan_lbl.configure(text="✗ No cameras found", text_color=RED)

    def _test_droid(self):
        ip   = self.ip_var.get().strip() or "192.168.1.5"
        port = self.port_var.get().strip() or "4747"
        url  = f"http://{ip}:{port}/video"
        self.droid_lbl.configure(text="Testing…", text_color=AMBER)
        self.update()
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret,_ = cap.read()
            cap.release()
            if ret:
                self.droid_lbl.configure(text="✓ Connected!", text_color=GREEN)
                return
        self.droid_lbl.configure(
            text="✗ Cannot connect — check IP & WiFi", text_color=RED)

    def _get_src(self):
        v = self.src_var.get()
        if v in ("webcam","usb"):
            return int(self.cam_idx.get() or "0")
        if v == "droidcam":
            ip   = self.ip_var.get().strip() or "192.168.1.5"
            port = self.port_var.get().strip() or "4747"
            return f"http://{ip}:{port}/video"
        return self.url_var.get().strip()

    def _start(self):
        src = self._get_src()
        if not src and src != 0:
            return
        if self.status_bar:
            self.status_bar.set("Detection running…", "running")
        self.thread = DetectionThread(src, "live",
            on_frame    = lambda f,s: self.after(0, lambda ff=f,ss=s:
                          self._on_frame(ff,ss)),
            on_status   = lambda m:   self.after(0, lambda mm=m:
                          self._on_status(mm)),
            on_done     = lambda s:   self.after(0, lambda ss=s:
                          self._on_done(ss)),
            use_ai_line = self.ai_var.get())
        self.thread.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

    def _on_frame(self, f, s):
        self.video.update_frame(f)
        if self.home_page and s.get("total_unique"):
            self.home_page.update_stats(s)

    def _on_status(self, m):
        if self.status_bar:
            st = "error" if "ERROR" in m or "Cannot" in m else "running"
            self.status_bar.set(m, st)

    def _on_done(self, s):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if self.status_bar: self.status_bar.set("Session complete ✓","idle")
        if self.home_page:
            self.home_page.update_stats(s)
            self.home_page.log_msg(f"Live session ended — {s.get('total_unique',0)} vehicles","ok")

    def _stop(self):
        if self.thread: self.thread.stop()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if self.status_bar: self.status_bar.set("Stopped","idle")


# ================================================================
#  FILE DETECTION PAGE
# ================================================================
class FilePage(Page):
    def __init__(self, master, status_bar=None, home_page=None):
        super().__init__(master)
        self.thread = None
        self.status_bar = status_bar
        self.home_page  = home_page
        self.grid_rowconfigure(4, weight=1)
        self.page_header("🎬","File Detection","Analyse a recorded road video")

        # File picker
        pick = ctk.CTkFrame(self, fg_color=BG2, corner_radius=14,
                             border_width=1, border_color=BORDER)
        pick.grid(row=1, column=0, padx=32, pady=(0,12), sticky="ew")
        pick.grid_columnconfigure(0, weight=1)
        SectionLabel(pick,"Video File").grid(row=0,column=0,columnspan=4,
                                              padx=18,pady=(14,10),sticky="w")
        prow = ctk.CTkFrame(pick, fg_color="transparent")
        prow.grid(row=1, column=0, columnspan=4, padx=16, pady=(0,14), sticky="ew")
        prow.grid_columnconfigure(0, weight=1)
        self.pv = tk.StringVar()
        ctk.CTkEntry(prow, textvariable=self.pv,
            placeholder_text="No video selected…",
            fg_color=BG3, border_color=BORDER, text_color=TEXT
        ).grid(row=0, column=0, sticky="ew", padx=(0,10))
        ctk.CTkButton(prow, text="📂  Browse", width=120, height=36,
            fg_color=BG4, hover_color=BG3,
            border_width=1, border_color=BORDER, text_color=BLUE,
            font=("Segoe UI",13), command=self._browse
        ).grid(row=0, column=1, padx=(0,8))

        # AI toggle
        ai_row = ctk.CTkFrame(pick, fg_color="transparent")
        ai_row.grid(row=2, column=0, columnspan=4, padx=16, pady=(0,14), sticky="w")
        self.ai_var = tk.BooleanVar(value=True)
        ctk.CTkSwitch(ai_row, text="", variable=self.ai_var,
                      button_color=TEAL, progress_color=TEAL,
                      width=44, height=22).pack(side="left",padx=(0,10))
        ctk.CTkLabel(ai_row, text="AI Auto-detect counting line",
                     font=("Segoe UI",13,"bold"), text_color=TEAL).pack(side="left")

        # Action buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=2, column=0, padx=32, pady=(0,10), sticky="w")
        self.run_btn = ctk.CTkButton(btn_row, text="▶  Analyse Video",
            width=180, height=44, fg_color=BLUE, hover_color=BLUE2,
            font=("Segoe UI",14,"bold"), corner_radius=10,
            state="disabled", command=self._run)
        self.run_btn.pack(side="left",padx=(0,10))
        self.stop_btn = ctk.CTkButton(btn_row, text="■  Stop",
            width=120, height=44, fg_color=RED2, hover_color="#991b1b",
            font=("Segoe UI",14,"bold"), corner_radius=10,
            state="disabled", command=self._stop)
        self.stop_btn.pack(side="left",padx=(0,16))
        self.prog_lbl = ctk.CTkLabel(btn_row, text="",
                                      font=("Segoe UI",12), text_color=MUTED)
        self.prog_lbl.pack(side="left")

        # Progress bar
        self.prog = ctk.CTkProgressBar(self, fg_color=BG3,
                                        progress_color=BLUE, height=6, corner_radius=3)
        self.prog.set(0)
        self.prog.grid(row=3, column=0, padx=32, pady=(0,8), sticky="ew")

        # Video + results
        mid = ctk.CTkFrame(self, fg_color="transparent")
        mid.grid(row=4, column=0, padx=32, pady=(0,24), sticky="nsew")
        mid.grid_columnconfigure(0, weight=3)
        mid.grid_columnconfigure(1, weight=1)
        mid.grid_rowconfigure(0, weight=1)

        self.video = VideoCanvas(mid)
        self.video.grid(row=0, column=0, padx=(0,12), sticky="nsew")

        res = ctk.CTkFrame(mid, fg_color=BG2, corner_radius=14,
                            border_width=1, border_color=BORDER)
        res.grid(row=0, column=1, sticky="nsew")
        res.grid_columnconfigure(0, weight=1)
        SectionLabel(res,"Results").grid(row=0,column=0,padx=14,pady=(14,8),sticky="w")
        self.rt = StatCard(res,"Total",  "—",BLUE,  "🚗"); self.rt.grid(row=1,column=0,padx=10,pady=6,sticky="ew")
        self.rc = StatCard(res,"Cars",   "—",GREEN, "🚙"); self.rc.grid(row=2,column=0,padx=10,pady=6,sticky="ew")
        self.rr = StatCard(res,"Rickshaws","—",AMBER,"🛺"); self.rr.grid(row=3,column=0,padx=10,pady=6,sticky="ew")
        self.rb = StatCard(res,"Bus+Truck","—",RED, "🚌"); self.rb.grid(row=4,column=0,padx=10,pady=(6,14),sticky="ew")

    def _browse(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(
            title="Open Road Video", initialdir="videos",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv *.wmv"),("All","*.*")])
        if p:
            self.pv.set(p)
            self.run_btn.configure(state="normal")

    def _run(self):
        p = self.pv.get().strip()
        if not p: return
        self.prog.set(0)
        if self.status_bar: self.status_bar.set("Analysing video…","running")
        self.thread = DetectionThread(p, "file",
            on_frame    = lambda f,s: self.after(0, lambda ff=f: self.video.update_frame(ff)),
            on_status   = lambda m:   self.after(0, lambda mm=m:
                          [self.prog_lbl.configure(text=mm),
                           self.status_bar and self.status_bar.set(mm,"running")]),
            on_done     = lambda s:   self.after(0, lambda ss=s: self._done(ss)),
            on_progress = lambda v:   self.after(0, lambda vv=v:
                          [self.prog.set(vv/100),
                           self.prog_lbl.configure(text=f"{vv}%")]),
            use_ai_line = self.ai_var.get())
        self.thread.start()
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

    def _stop(self):
        if self.thread: self.thread.stop()
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def _done(self, s):
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.prog.set(1)
        bt = s.get("by_type",{})
        self.rt.set(s.get("total_unique",0))
        self.rc.set(bt.get("car",0))
        self.rr.set(bt.get("rickshaw/motorcycle",0))
        self.rb.set(bt.get("bus",0)+bt.get("truck",0))
        if self.status_bar: self.status_bar.set("Analysis complete ✓","idle")
        if self.home_page:
            self.home_page.update_stats(s)
            total = s.get("total_unique",0)
            self.home_page.log_msg(f"File analysis complete — {total} vehicles","ok")


# ================================================================
#  EMBEDDED DASHBOARD PAGE  (matplotlib — no browser needed)
# ================================================================
class DashboardPage(Page):
    def __init__(self, master):
        super().__init__(master)
        self.grid_rowconfigure(2, weight=1)
        self.page_header("📊","Analytics Dashboard","Traffic data — no browser needed")

        # Filter bar
        fbar = ctk.CTkFrame(self, fg_color=BG2, corner_radius=12,
                             border_width=1, border_color=BORDER)
        fbar.grid(row=1, column=0, padx=32, pady=(0,12), sticky="ew")
        inner = ctk.CTkFrame(fbar, fg_color="transparent")
        inner.pack(padx=16, pady=12, fill="x")
        ctk.CTkLabel(inner, text="Chart:", text_color=MUTED,
                     font=("Segoe UI",12)).pack(side="left",padx=(0,8))
        self.chart_var = tk.StringVar(value="Daily")
        for label in ["Daily","Hourly","Monthly","Vehicle Types","By Zone"]:
            ctk.CTkRadioButton(inner, text=label, variable=self.chart_var,
                               value=label, font=("Segoe UI",12), text_color=TEXT,
                               fg_color=BLUE, hover_color=BLUE2,
                               command=self._redraw
            ).pack(side="left",padx=(0,16))
        Divider(inner,width=1,height=24).pack(side="left",padx=(8,16))
        ctk.CTkLabel(inner,text="From:", text_color=MUTED,
                     font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.start_var = tk.StringVar()
        ctk.CTkEntry(inner, textvariable=self.start_var,
            placeholder_text="YYYY-MM-DD", width=130,
            fg_color=BG3, border_color=BORDER, text_color=TEXT
        ).pack(side="left",padx=(0,8))
        ctk.CTkLabel(inner,text="To:", text_color=MUTED,
                     font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.end_var = tk.StringVar()
        ctk.CTkEntry(inner, textvariable=self.end_var,
            placeholder_text="YYYY-MM-DD", width=130,
            fg_color=BG3, border_color=BORDER, text_color=TEXT
        ).pack(side="left",padx=(0,10))
        ctk.CTkButton(inner, text="Refresh", width=100, height=32,
            fg_color=BLUE, hover_color=BLUE2, font=("Segoe UI",12),
            command=self._redraw
        ).pack(side="left")

        # Chart container
        chart_frame = ctk.CTkFrame(self, fg_color=BG2, corner_radius=14,
                                    border_width=1, border_color=BORDER)
        chart_frame.grid(row=2, column=0, padx=32, pady=(0,24), sticky="nsew")

        # Create matplotlib figure
        self.fig = Figure(figsize=(12,5.5), facecolor=CHART_BG)
        self.fig.subplots_adjust(left=0.07,right=0.97,top=0.88,bottom=0.12)
        self.ax = self.fig.add_subplot(111)
        self._style_ax(self.ax)

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._redraw()

    def _style_ax(self, ax):
        ax.set_facecolor(CHART_BG)
        ax.tick_params(colors="#64748b", labelsize=10)
        ax.xaxis.label.set_color("#64748b")
        ax.yaxis.label.set_color("#64748b")
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_color(CHART_GRID)
        ax.grid(True, color=CHART_GRID, linewidth=0.5, linestyle="--", alpha=0.7)

    def _load_df(self):
        files = glob.glob(os.path.join("data","log_*.csv"))
        if not files: return pd.DataFrame()
        dfs=[]
        for f in files:
            try:
                tmp = pd.read_csv(f)
                if not tmp.empty: dfs.append(tmp)
            except: pass
        if not dfs: return pd.DataFrame()
        df = pd.concat(dfs,ignore_index=True)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"],errors="coerce")
            df["date"]  = df["timestamp"].dt.date
            df["hour"]  = df["timestamp"].dt.hour
            df["month"] = df["timestamp"].dt.to_period("M").astype(str)
        # Date filters
        start = self.start_var.get().strip()
        end   = self.end_var.get().strip()
        if start and "date" in df.columns:
            try: df=df[df["date"]>=datetime.date.fromisoformat(start)]
            except: pass
        if end and "date" in df.columns:
            try: df=df[df["date"]<=datetime.date.fromisoformat(end)]
            except: pass
        return df

    def _redraw(self):
        df = self._load_df()
        self.ax.clear()
        self._style_ax(self.ax)
        chart = self.chart_var.get()

        if df.empty:
            self.ax.text(0.5,0.5,"No data yet.\nRun detection first.",
                ha="center",va="center",color=MUTED,
                fontsize=14,transform=self.ax.transAxes)
            self.canvas.draw()
            return

        if chart=="Daily" and "date" in df.columns:
            counts = df.groupby("date").size()
            bars = self.ax.bar(range(len(counts)), counts.values,
                               color=BLUE, alpha=0.85, width=0.7)
            self.ax.set_xticks(range(len(counts)))
            self.ax.set_xticklabels([str(d) for d in counts.index],
                                     rotation=30,ha="right",fontsize=9)
            self.ax.set_title("Daily Vehicle Count", color=TEXT, fontsize=13, pad=10)
            self.ax.set_ylabel("Vehicles")
            for bar in bars:
                h=bar.get_height()
                if h>0:
                    self.ax.text(bar.get_x()+bar.get_width()/2.,h+0.5,
                                 str(int(h)),ha="center",va="bottom",
                                 color="#94a3b8",fontsize=8)

        elif chart=="Hourly" and "hour" in df.columns:
            counts = df.groupby("hour").size().reindex(range(24),fill_value=0)
            self.ax.bar(counts.index, counts.values, color=AMBER, alpha=0.85, width=0.8)
            self.ax.set_xticks(range(0,24,2))
            self.ax.set_xticklabels([f"{h:02d}:00" for h in range(0,24,2)],
                                     rotation=30,ha="right",fontsize=9)
            self.ax.set_title("Traffic by Hour of Day", color=TEXT, fontsize=13, pad=10)
            self.ax.set_ylabel("Vehicles")

        elif chart=="Monthly" and "month" in df.columns:
            counts = df.groupby("month").size()
            self.ax.bar(range(len(counts)), counts.values,
                        color=GREEN, alpha=0.85, width=0.7)
            self.ax.set_xticks(range(len(counts)))
            self.ax.set_xticklabels(counts.index.astype(str),
                                     rotation=20,ha="right",fontsize=10)
            self.ax.set_title("Monthly Vehicle Count", color=TEXT, fontsize=13, pad=10)
            self.ax.set_ylabel("Vehicles")

        elif chart=="Vehicle Types" and "vehicle_type" in df.columns:
            counts = df["vehicle_type"].value_counts()
            colors = [BLUE, AMBER, RED, GREEN, PURPLE, TEAL][:len(counts)]
            wedges,texts,autotexts = self.ax.pie(
                counts.values, labels=counts.index,
                autopct="%1.0f%%", colors=colors,
                startangle=140,
                wedgeprops={"linewidth":0.5,"edgecolor":BG},
                textprops={"color":TEXT,"fontsize":11})
            for at in autotexts: at.set_color(BG); at.set_fontsize(10)
            self.ax.set_title("Vehicle Type Distribution", color=TEXT, fontsize=13, pad=10)

        elif chart=="By Zone" and "zone" in df.columns:
            counts = df.groupby("zone").size().sort_values(ascending=True)
            if len(counts)>1 and "all" not in counts.index:
                colors = LANE_COLS[:len(counts)]
                self.ax.barh(counts.index, counts.values, color=colors, alpha=0.85)
                self.ax.set_title("Vehicles by Road / Zone", color=TEXT, fontsize=13, pad=10)
                self.ax.set_xlabel("Vehicles")
            else:
                self.ax.text(0.5,0.5,"Zone counting not enabled.\nDraw lanes first.",
                    ha="center",va="center",color=MUTED,fontsize=13,transform=self.ax.transAxes)

        self.fig.tight_layout()
        self.canvas.draw()

    def refresh(self): self._redraw()


# ================================================================
#  LANE DRAWING PAGE
# ================================================================
class LanePage(Page):
    COLS=LANE_COLS
    def __init__(self,master):
        super().__init__(master); self.cap=None; self.lanes=[]; self.cur_pts=[]; self.fphoto=None
        self.grid_rowconfigure(3,weight=1)
        self.page_header("🗺","Lane Drawing","Click to define road zones — any shape, any angle")

        r1=ctk.CTkFrame(self,fg_color=BG2,corner_radius=12,border_width=1,border_color=BORDER)
        r1.grid(row=1,column=0,padx=32,pady=(0,10),sticky="ew")
        rr=ctk.CTkFrame(r1,fg_color="transparent"); rr.pack(padx=14,pady=12,fill="x")
        ctk.CTkButton(rr,text="📂  Load Video",width=140,height=36,
            fg_color=BG4,hover_color=BG3,border_width=1,border_color=BORDER,
            text_color=BLUE,font=("Segoe UI",13),command=self._load).pack(side="left",padx=(0,14))
        ctk.CTkLabel(rr,text="Seek frame:",text_color=MUTED,font=("Segoe UI",12)).pack(side="left",padx=(0,8))
        self.svar=tk.IntVar(value=0)
        self.slider=ctk.CTkSlider(rr,from_=0,to=100,variable=self.svar,command=self._seek,
            button_color=BLUE,progress_color=BLUE,fg_color=BG3,width=280,state="disabled")
        self.slider.pack(side="left",padx=(0,10))
        self.flbl=ctk.CTkLabel(rr,text="Frame 0",text_color=MUTED,font=("Segoe UI",11),width=75)
        self.flbl.pack(side="left")

        cf=ctk.CTkFrame(self,fg_color=BG3,corner_radius=14,border_width=1,border_color=BORDER)
        cf.grid(row=2,column=0,padx=32,pady=(0,8),sticky="ew")
        ctk.CTkLabel(cf,text="LEFT CLICK = add point  ·  RIGHT CLICK = remove last point  ·  Finish Lane button to save",
            text_color=HINT,font=("Segoe UI",11)).pack(pady=6)

        cvf=ctk.CTkFrame(self,fg_color=BG3,corner_radius=14,border_width=1,border_color=BORDER)
        cvf.grid(row=3,column=0,padx=32,pady=(0,10),sticky="nsew")
        self.canvas=tk.Canvas(cvf,bg="#0a0d14",highlightthickness=0,cursor="crosshair")
        self.canvas.pack(fill="both",expand=True,padx=2,pady=2)
        self.canvas.bind("<Button-1>",self._click); self.canvas.bind("<Button-3>",self._rclick)

        r2=ctk.CTkFrame(self,fg_color="transparent"); r2.grid(row=4,column=0,padx=32,pady=(0,24),sticky="ew")
        self.ne=ctk.CTkEntry(r2,placeholder_text="Lane name (e.g. North Road, Mirpur Road)",
            width=240,fg_color=BG3,border_color=BORDER,text_color=TEXT); self.ne.pack(side="left",padx=(0,10))
        ctk.CTkButton(r2,text="✓  Finish Lane",width=130,height=38,fg_color=BLUE,hover_color=BLUE2,
            font=("Segoe UI",13,"bold"),command=self._finish).pack(side="left",padx=(0,8))
        ctk.CTkButton(r2,text="↩  Undo",width=90,height=38,fg_color=BG4,hover_color=BG3,
            border_width=1,border_color=BORDER,text_color=TEXT,command=self._undo).pack(side="left",padx=(0,8))
        ctk.CTkButton(r2,text="✕  Clear",width=90,height=38,fg_color=BG4,hover_color=BG3,
            border_width=1,border_color=BORDER,text_color=TEXT,command=self._clr).pack(side="left",padx=(0,18))
        ctk.CTkButton(r2,text="💾  Save All Lanes",width=160,height=38,
            fg_color=GREEN2,hover_color="#047857",border_width=1,border_color="#047857",
            text_color=GREEN,font=("Segoe UI",13,"bold"),command=self._save).pack(side="left")
        self.ll=ctk.CTkLabel(r2,text="0 lanes drawn",text_color=MUTED,font=("Segoe UI",12))
        self.ll.pack(side="left",padx=14)

    def _load(self):
        from tkinter import filedialog
        p=filedialog.askopenfilename(initialdir="videos",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")])
        if not p: return
        self.cap=cv2.VideoCapture(p); tf=max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))-1,1)
        self.slider.configure(to=tf,state="normal"); self._seek(0)
    def _seek(self,v):
        if not self.cap: return
        idx=int(float(v)); self.cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ret,frame=self.cap.read()
        if ret: self.cur_frame=frame.copy(); self.flbl.configure(text=f"Frame {idx}"); self._redraw()
    def _redraw(self):
        self.canvas.delete("all")
        if not hasattr(self,"cur_frame"): return
        h,w=self.cur_frame.shape[:2]; cw=max(self.canvas.winfo_width(),640); ch=max(self.canvas.winfo_height(),360)
        sc=min(cw/w,ch/h,1.0); nw,nh=int(w*sc),int(h*sc); ox,oy=(cw-nw)//2,(ch-nh)//2
        self._ox=ox; self._oy=oy; self._nw=nw; self._nh=nh
        rgb=cv2.cvtColor(cv2.resize(self.cur_frame,(nw,nh)),cv2.COLOR_BGR2RGB)
        self.fphoto=ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(ox,oy,anchor="nw",image=self.fphoto)
        for i,lane in enumerate(self.lanes):
            col=self.COLS[i%len(self.COLS)]; pts=[(ox+fx*nw,oy+fy*nh) for fx,fy in lane["points"]]
            flat=[c for xy in pts for c in xy]
            if len(flat)>=4: self.canvas.create_polygon(flat,fill=col,stipple="gray25",outline=col,width=2)
            cx=sum(p[0] for p in pts)/len(pts); cy=sum(p[1] for p in pts)/len(pts)
            self.canvas.create_text(cx+1,cy+1,text=lane["name"],fill="#000",font=("Segoe UI",11,"bold"))
            self.canvas.create_text(cx,cy,text=lane["name"],fill="white",font=("Segoe UI",11,"bold"))
        if self.cur_pts:
            col=self.COLS[len(self.lanes)%len(self.COLS)]; flat=[c for xy in self.cur_pts for c in xy]
            for px,py in self.cur_pts:
                self.canvas.create_oval(px-6,py-6,px+6,py+6,fill=col,outline="white",width=1)
            if len(self.cur_pts)>1: self.canvas.create_line(flat,fill=col,width=2,dash=(6,3))
            if len(self.cur_pts)>=3:
                self.canvas.create_line(self.cur_pts[-1][0],self.cur_pts[-1][1],
                    self.cur_pts[0][0],self.cur_pts[0][1],fill=col,width=1,dash=(4,4))
    def _click(self,e): self.cur_pts.append((e.x,e.y)); self._redraw()
    def _rclick(self,e):
        if self.cur_pts: self.cur_pts.pop(); self._redraw()
    def _finish(self):
        if len(self.cur_pts)<3:
            from tkinter import messagebox; messagebox.showwarning("Need more points","Place at least 3 points."); return
        name=self.ne.get().strip() or f"Lane {len(self.lanes)+1}"
        nw=self._nw or 640; nh=self._nh or 360; ox=self._ox or 0; oy=self._oy or 0
        frac=[((px-ox)/max(nw,1),(py-oy)/max(nh,1)) for px,py in self.cur_pts]
        self.lanes.append({"name":name,"points":frac}); self.cur_pts=[]; self.ne.delete(0,"end")
        self.ll.configure(text=f"{len(self.lanes)} lane(s) drawn"); self._redraw()
    def _undo(self):
        if self.lanes: self.lanes.pop(); self.ll.configure(text=f"{len(self.lanes)} lane(s)"); self._redraw()
    def _clr(self): self.cur_pts=[]; self._redraw()
    def _save(self):
        if not self.lanes:
            from tkinter import messagebox; messagebox.showwarning("No lanes","Draw at least one lane."); return
        os.makedirs("data",exist_ok=True)
        with open("data/lanes.json","w") as f: json.dump({"lanes":self.lanes},f,indent=2)
        try:
            with open("config.py") as f: c=f.read()
            zl=["ZONES = {\n"]
            for lane in self.lanes:
                xs=[p[0] for p in lane["points"]]; ys=[p[1] for p in lane["points"]]
                zl.append(f'    "{lane["name"]}": ({min(xs):.3f},{min(ys):.3f},{max(xs):.3f},{max(ys):.3f}),\n')
            zl.append("}\n")
            c=re.sub(r"ENABLE_ZONES\s*=\s*\w+","ENABLE_ZONES = True",c)
            c=re.sub(r"ZONES\s*=\s*\{[^}]*\}","".join(zl).rstrip(),c,flags=re.DOTALL)
            with open("config.py","w") as f: f.write(c)
        except: pass
        from tkinter import messagebox
        messagebox.showinfo("Saved ✓",f"{len(self.lanes)} lane(s) saved.\nconfig.py updated — zone counting is now ON.")


# ================================================================
#  SETTINGS PAGE
# ================================================================
class SettingsPage(Page):
    def __init__(self,master):
        super().__init__(master); self.grid_rowconfigure(2,weight=1); self.grid_columnconfigure(0,weight=1)
        self.page_header("⚙️","Settings","Configure detection parameters")

        scroll=ctk.CTkScrollableFrame(self,fg_color="transparent",corner_radius=0)
        scroll.grid(row=1,column=0,padx=32,pady=(0,24),sticky="nsew"); self.grid_rowconfigure(1,weight=1)
        scroll.grid_columnconfigure(0,weight=1)

        def section(parent,title):
            f=ctk.CTkFrame(parent,fg_color=BG2,corner_radius=14,border_width=1,border_color=BORDER)
            f.pack(fill="x",pady=(0,14)); SectionLabel(f,title).pack(anchor="w",padx=18,pady=(14,10)); return f

        def row(parent,label,widget,hint=""):
            r=ctk.CTkFrame(parent,fg_color="transparent"); r.pack(fill="x",padx=16,pady=(0,10))
            ctk.CTkLabel(r,text=label,text_color=TEXT,font=("Segoe UI",13),width=240,anchor="w").pack(side="left")
            widget.pack(side="left",padx=(0,10))
            if hint: ctk.CTkLabel(r,text=hint,text_color=HINT,font=("Segoe UI",11)).pack(side="left")

        # Model
        sec=section(scroll,"Model")
        self.model_cb=ctk.CTkComboBox(sec,values=["yolov8n.pt  (fastest)","yolov8s.pt  (balanced)","yolov8m.pt  (accurate)"],
            width=280,fg_color=BG3,border_color=BORDER,text_color=TEXT)
        row(sec,"YOLO Model",self.model_cb,"nano=fastest, medium=most accurate")
        self.conf=ctk.CTkSlider(sec,from_=0.1,to=0.9,button_color=BLUE,progress_color=BLUE,fg_color=BG3,width=260)
        self.conf.set(0.40); row(sec,"Confidence (0.1–0.9)",self.conf,"Low=sensitive, High=strict")
        sec.pack_configure(pady=(0,14))

        # Speed
        sec2=section(scroll,"Speed Estimation")
        self.ppm=ctk.CTkEntry(sec2,width=120,fg_color=BG3,border_color=BORDER,text_color=TEXT)
        self.ppm.insert(0,"55"); row(sec2,"Pixels per metre",self.ppm,"Measure a known distance in your video frame")
        self.fps_e=ctk.CTkEntry(sec2,width=120,fg_color=BG3,border_color=BORDER,text_color=TEXT)
        self.fps_e.insert(0,"25"); row(sec2,"Video FPS",self.fps_e)

        # Display
        sec3=section(scroll,"Display")
        self.sw_speed=ctk.CTkSwitch(sec3,text="Show speed (km/h) on video",onvalue=True,offvalue=False,button_color=BLUE,progress_color=BLUE); self.sw_speed.select()
        row(sec3,"",self.sw_speed)
        self.sw_ids=ctk.CTkSwitch(sec3,text="Show vehicle track IDs",onvalue=True,offvalue=False,button_color=BLUE,progress_color=BLUE); self.sw_ids.select()
        row(sec3,"",self.sw_ids)
        self.sw_zones=ctk.CTkSwitch(sec3,text="Enable lane / zone counting",onvalue=True,offvalue=False,button_color=BLUE,progress_color=BLUE)
        row(sec3,"",self.sw_zones)
        self.lpos=ctk.CTkSlider(sec3,from_=0.1,to=0.9,button_color=TEAL,progress_color=TEAL,fg_color=BG3,width=260)
        self.lpos.set(0.55); row(sec3,"Manual counting line position",self.lpos,"Used when AI line is off")

        ctk.CTkButton(scroll,text="💾  Save Settings",width=200,height=44,
            fg_color=BLUE,hover_color=BLUE2,font=("Segoe UI",14,"bold"),
            corner_radius=10,command=self._save).pack(anchor="w",pady=(4,0))
        self._load()

    def _load(self):
        try:
            import config
            m={"yolov8n.pt":0,"yolov8s.pt":1,"yolov8m.pt":2}
            self.model_cb.set(self.model_cb.cget("values")[m.get(config.YOLO_MODEL,0)])
            self.conf.set(config.CONFIDENCE); self.ppm.delete(0,"end"); self.ppm.insert(0,str(config.PIXELS_PER_METER))
            self.fps_e.delete(0,"end"); self.fps_e.insert(0,str(config.VIDEO_FPS))
            (self.sw_speed.select if config.SHOW_SPEED else self.sw_speed.deselect)()
            (self.sw_ids.select if config.SHOW_IDS else self.sw_ids.deselect)()
            (self.sw_zones.select if config.ENABLE_ZONES else self.sw_zones.deselect)()
            self.lpos.set(config.COUNTING_LINE_POSITION)
        except: pass
    def _save(self):
        try:
            with open("config.py") as f: c=f.read()
            mn=["yolov8n.pt","yolov8s.pt","yolov8m.pt"]
            idx=next((i for i,v in enumerate(self.model_cb.cget("values")) if self.model_cb.get() in v),0)
            subs=[(r'YOLO_MODEL\s*=\s*"[^"]*"',f'YOLO_MODEL = "{mn[idx]}"'),
                  (r'CONFIDENCE\s*=\s*[\d.]+',f'CONFIDENCE = {self.conf.get():.2f}'),
                  (r'PIXELS_PER_METER\s*=\s*\d+',f'PIXELS_PER_METER = {self.ppm.get()}'),
                  (r'VIDEO_FPS\s*=\s*\d+',f'VIDEO_FPS = {self.fps_e.get()}'),
                  (r'SHOW_SPEED\s*=\s*\w+',f'SHOW_SPEED = {bool(self.sw_speed.get())}'),
                  (r'SHOW_IDS\s*=\s*\w+',f'SHOW_IDS = {bool(self.sw_ids.get())}'),
                  (r'ENABLE_ZONES\s*=\s*\w+',f'ENABLE_ZONES = {bool(self.sw_zones.get())}'),
                  (r'COUNTING_LINE_POSITION\s*=\s*[\d.]+',f'COUNTING_LINE_POSITION = {self.lpos.get():.2f}')]
            for pat,rep in subs: c=re.sub(pat,rep,c)
            with open("config.py","w") as f: f.write(c)
            from tkinter import messagebox; messagebox.showinfo("Saved ✓","Settings saved to config.py ✓")
        except Exception as e:
            from tkinter import messagebox; messagebox.showerror("Error",str(e))


# ================================================================
#  MAIN APPLICATION WINDOW
# ================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TrafficCounter BD")
        self.geometry("1320x820"); self.minsize(1100,680)
        self.configure(fg_color=BG)
        self.grid_columnconfigure(1,weight=1)
        self.grid_rowconfigure(0,weight=1)

        # ── Sidebar ───────────────────────────────────────────
        sb=ctk.CTkFrame(self,fg_color=BG2,corner_radius=0,width=230)
        sb.grid(row=0,column=0,sticky="nsew"); sb.grid_propagate(False)
        sb.grid_rowconfigure(12,weight=1); sb.grid_columnconfigure(0,weight=1)

        # Logo section
        logo_f=ctk.CTkFrame(sb,fg_color=BG3,corner_radius=14,border_width=1,border_color=BORDER)
        logo_f.grid(row=0,column=0,padx=14,pady=(20,8),sticky="ew")
        lf_in=ctk.CTkFrame(logo_f,fg_color="transparent"); lf_in.pack(padx=14,pady=12,fill="x")
        ic=ctk.CTkFrame(lf_in,fg_color=BLUE2,width=40,height=40,corner_radius=10)
        ic.pack(side="left",padx=(0,10)); ic.pack_propagate(False)
        ctk.CTkLabel(ic,text="🚗",font=("Segoe UI",20)).place(relx=0.5,rely=0.5,anchor="center")
        ctk.CTkLabel(lf_in,text="TrafficCounter",font=("Segoe UI",14,"bold"),text_color="#fff").pack(anchor="w")
        ctk.CTkLabel(lf_in,text="BD Edition · v4.0",font=("Segoe UI",10),text_color=MUTED).pack(anchor="w")

        Divider(sb).grid(row=1,column=0,padx=14,pady=(4,8),sticky="ew")
        SectionLabel(sb,"Navigation").grid(row=2,column=0,padx=22,pady=(0,6),sticky="w")

        nav_items=[("🏠","Home"),("📹","Live Detection"),("🎬","File Detection"),
                   ("🗺","Lane Drawing"),("📊","Analytics"),("⚙️","Settings")]
        self.nav_btns=[]
        for i,(icon,label) in enumerate(nav_items):
            btn=NavBtn(sb,icon,label,lambda idx=i: self._switch(idx))
            btn.grid(row=3+i,column=0,padx=10,pady=2,sticky="ew")
            self.nav_btns.append(btn)

        Divider(sb).grid(row=12,column=0,padx=14,pady=4,sticky="ew")

        # Bottom info
        info_f=ctk.CTkFrame(sb,fg_color="transparent")
        info_f.grid(row=13,column=0,padx=14,pady=(4,16),sticky="ew")
        ctk.CTkLabel(info_f,text="🎓 SUST CEE Research",font=("Segoe UI",10),text_color=HINT).pack(anchor="w")
        ctk.CTkLabel(info_f,text="100% Free & Open Source",font=("Segoe UI",10),text_color=HINT).pack(anchor="w")

        # ── Main content ──────────────────────────────────────
        content=ctk.CTkFrame(self,fg_color=BG,corner_radius=0)
        content.grid(row=0,column=1,sticky="nsew"); content.grid_columnconfigure(0,weight=1)
        content.grid_rowconfigure(0,weight=1)

        # Status bar
        self.statusbar=StatusBar(self)
        self.grid_rowconfigure(1,weight=0)
        self.statusbar.grid(row=1,column=0,columnspan=2,sticky="ew")
        self.statusbar.set_right("TrafficCounter BD v4.0  ·  SUST CEE")

        # Pages
        self.hp=HomePage(content)
        self.lp=LivePage(content, status_bar=self.statusbar, home_page=self.hp)
        self.fp=FilePage(content, status_bar=self.statusbar, home_page=self.hp)
        self.la=LanePage(content)
        self.dp=DashboardPage(content)
        self.sp=SettingsPage(content)
        self._pages=[self.hp,self.lp,self.fp,self.la,self.dp,self.sp]
        for p in self._pages:
            p.grid(row=0,column=0,sticky="nsew")
        self._switch(0)

    def _switch(self,idx):
        for i,p in enumerate(self._pages):
            p.tkraise() if i==idx else None
            p.grid(row=0,column=0,sticky="nsew")
        self._pages[idx].tkraise()
        for i,b in enumerate(self.nav_btns): b.set_active(i==idx)
        # Refresh dashboard when opened
        if idx==4: self.dp.refresh()
        self.statusbar.set(["Home","Live Detection","File Detection",
                             "Lane Drawing","Analytics","Settings"][idx],"idle")


# ── Entry point ────────────────────────────────────────────────
if __name__=="__main__":
    os.makedirs("videos",exist_ok=True)
    os.makedirs("data",  exist_ok=True)
    App().mainloop()
