# ============================================================
#  dashboard.py  –  Web dashboard  (http://localhost:5000)
#  Run:  python dashboard.py
# ============================================================

import os
import io
import glob
import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-GUI backend (required for Flask)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, render_template_string, send_file, request, jsonify
import config

app = Flask(__name__)

# ── Helper: load all CSV logs ─────────────────────────────────
def load_all_logs():
    csv_files = glob.glob(os.path.join(config.DATA_FOLDER, "log_*.csv"))
    if not csv_files:
        return pd.DataFrame()
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, parse_dates=["timestamp"])
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"]  = df["timestamp"].dt.date
    df["hour"]  = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    return df


# ── Chart generators ──────────────────────────────────────────
def make_chart(fig):
    """Convert matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#f8f8f8")
    buf.seek(0)
    plt.close(fig)
    return buf


def chart_daily(df, start=None, end=None):
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    counts = df.groupby("date").size()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(counts.index, counts.values, color="#185FA5", width=0.7)
    ax.set_title("Daily vehicle count", fontsize=13, pad=8)
    ax.set_xlabel("Date"); ax.set_ylabel("Vehicles")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return make_chart(fig)


def chart_monthly(df):
    counts = df.groupby("month").size()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(counts.index.astype(str), counts.values, color="#0F6E56", width=0.6)
    ax.set_title("Monthly vehicle count", fontsize=13, pad=8)
    ax.set_xlabel("Month"); ax.set_ylabel("Vehicles")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return make_chart(fig)


def chart_hourly(df):
    counts = df.groupby("hour").size().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(counts.index, counts.values, color="#BA7517", width=0.8)
    ax.set_title("Traffic by hour of day", fontsize=13, pad=8)
    ax.set_xlabel("Hour (0–23)"); ax.set_ylabel("Vehicles")
    ax.set_xticks(range(0, 24))
    fig.tight_layout()
    return make_chart(fig)


def chart_vehicle_type(df):
    counts = df["vehicle_type"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    wedge_colours = ["#185FA5","#BA7517","#A32D2D","#3B6D11","#534AB7"]
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
           colors=wedge_colours[:len(counts)], startangle=140,
           wedgeprops={"linewidth":0.5,"edgecolor":"white"})
    ax.set_title("Vehicle type breakdown", fontsize=13, pad=8)
    fig.tight_layout()
    return make_chart(fig)


def chart_zone(df):
    if "zone" not in df.columns:
        return None
    counts = df.groupby("zone").size()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(counts.index, counts.values, color="#534AB7")
    ax.set_title("Vehicles by road / zone", fontsize=13, pad=8)
    ax.set_xlabel("Vehicles")
    fig.tight_layout()
    return make_chart(fig)


# ── Flask routes ──────────────────────────────────────────────
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vehicle Counter Dashboard</title>
<style>
  body{margin:0;font-family:system-ui,sans-serif;background:#f2f4f7;color:#1a1a1a}
  header{background:#185FA5;color:#fff;padding:14px 24px;display:flex;align-items:center;gap:14px}
  header h1{margin:0;font-size:20px;font-weight:500}
  header span{font-size:13px;opacity:.75}
  .container{max-width:1100px;margin:24px auto;padding:0 16px}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:24px}
  .stat{background:#fff;border-radius:10px;padding:16px 20px;border:0.5px solid #e0e0e0}
  .stat .label{font-size:12px;color:#666;margin-bottom:4px}
  .stat .value{font-size:26px;font-weight:500}
  .charts{display:grid;grid-template-columns:1fr;gap:18px}
  .chart-card{background:#fff;border-radius:10px;padding:16px 20px;border:0.5px solid #e0e0e0}
  .chart-card h2{margin:0 0 12px;font-size:14px;font-weight:500;color:#444}
  .chart-card img{width:100%;border-radius:6px}
  .filters{background:#fff;border-radius:10px;padding:16px 20px;border:0.5px solid #e0e0e0;margin-bottom:18px;display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end}
  .filters label{font-size:13px;color:#555;display:flex;flex-direction:column;gap:4px}
  .filters input,.filters select{border:0.5px solid #ccc;border-radius:6px;padding:6px 10px;font-size:13px}
  .filters button{background:#185FA5;color:#fff;border:none;border-radius:6px;padding:8px 18px;cursor:pointer;font-size:13px}
  .filters button:hover{background:#0c447c}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:620px){.grid2{grid-template-columns:1fr}}
  footer{text-align:center;font-size:12px;color:#999;padding:24px}
</style>
</head>
<body>
<header>
  <div>
    <h1>🚗 Vehicle Counter Dashboard</h1>
    <span>Bangladeshi Road Traffic Analysis</span>
  </div>
</header>

<div class="container">

  <div class="stats">
    <div class="stat"><div class="label">Total vehicles</div>
      <div class="value" style="color:#185FA5">{{ total }}</div></div>
    <div class="stat"><div class="label">Sessions logged</div>
      <div class="value">{{ sessions }}</div></div>
    <div class="stat"><div class="label">Days with data</div>
      <div class="value">{{ days }}</div></div>
    <div class="stat"><div class="label">Most common</div>
      <div class="value" style="font-size:16px">{{ top_type }}</div></div>
  </div>

  <form class="filters" method="GET">
    <label>Start date <input type="date" name="start" value="{{ start }}"></label>
    <label>End date   <input type="date" name="end"   value="{{ end }}"></label>
    <button type="submit">Apply filter</button>
    <a href="/" style="font-size:13px;color:#666;margin-left:4px">Reset</a>
  </form>

  <div class="charts">

    <div class="chart-card">
      <h2>Daily count</h2>
      <img src="/chart/daily?start={{ start }}&end={{ end }}" alt="daily chart">
    </div>

    <div class="grid2">
      <div class="chart-card">
        <h2>Vehicle types</h2>
        <img src="/chart/type?start={{ start }}&end={{ end }}" alt="type chart">
      </div>
      <div class="chart-card">
        <h2>Traffic by hour</h2>
        <img src="/chart/hourly?start={{ start }}&end={{ end }}" alt="hourly chart">
      </div>
    </div>

    <div class="chart-card">
      <h2>Monthly trend</h2>
      <img src="/chart/monthly?start={{ start }}&end={{ end }}" alt="monthly chart">
    </div>

    {% if has_zones %}
    <div class="chart-card">
      <h2>By road / zone</h2>
      <img src="/chart/zone?start={{ start }}&end={{ end }}" alt="zone chart">
    </div>
    {% endif %}

  </div>
</div>
<footer>Vehicle Counter System — All data local, 100% free</footer>
</body></html>
"""


@app.route("/")
def index():
    df    = load_all_logs()
    start = request.args.get("start", "")
    end   = request.args.get("end",   "")

    if df.empty:
        return ("<h2 style='font-family:sans-serif;padding:40px'>"
                "No data yet. Run a detection session first, then refresh.</h2>")

    dff = df.copy()
    if start:
        dff = dff[dff["date"] >= datetime.date.fromisoformat(start)]
    if end:
        dff = dff[dff["date"] <= datetime.date.fromisoformat(end)]

    total    = len(dff)
    sessions = dff["session"].nunique() if "session" in dff.columns else "—"
    days     = dff["date"].nunique()
    top_type = (dff["vehicle_type"].value_counts().idxmax()
                if not dff.empty else "—")
    has_zones = ("zone" in dff.columns and
                 dff["zone"].nunique() > 1 and
                 "all" not in dff["zone"].unique())

    return render_template_string(TEMPLATE,
        total=total, sessions=sessions, days=days, top_type=top_type,
        start=start, end=end, has_zones=has_zones)


def _filtered_df(request):
    df    = load_all_logs()
    start = request.args.get("start","")
    end   = request.args.get("end","")
    if df.empty:
        return df
    if start:
        df = df[df["date"] >= datetime.date.fromisoformat(start)]
    if end:
        df = df[df["date"] <= datetime.date.fromisoformat(end)]
    return df


@app.route("/chart/daily")
def chart_daily_route():
    df = _filtered_df(request)
    if df.empty:
        return "no data", 404
    start = request.args.get("start") or None
    end   = request.args.get("end")   or None
    if start:
        start = datetime.date.fromisoformat(start)
    if end:
        end = datetime.date.fromisoformat(end)
    buf = chart_daily(df, start, end)
    return send_file(buf, mimetype="image/png")


@app.route("/chart/monthly")
def chart_monthly_route():
    df = _filtered_df(request)
    if df.empty: return "no data", 404
    return send_file(chart_monthly(df), mimetype="image/png")


@app.route("/chart/hourly")
def chart_hourly_route():
    df = _filtered_df(request)
    if df.empty: return "no data", 404
    return send_file(chart_hourly(df), mimetype="image/png")


@app.route("/chart/type")
def chart_type_route():
    df = _filtered_df(request)
    if df.empty: return "no data", 404
    return send_file(chart_vehicle_type(df), mimetype="image/png")


@app.route("/chart/zone")
def chart_zone_route():
    df = _filtered_df(request)
    if df.empty: return "no data", 404
    buf = chart_zone(df)
    if not buf: return "no zone data", 404
    return send_file(buf, mimetype="image/png")


# ══════════════════════════════════════════════════════════════
#  MOBILE API  —  used by the Android companion app
# ══════════════════════════════════════════════════════════════
import io as _io, re as _re
from flask import jsonify

# In-memory frame store (set by detector when running alongside Flask)
_latest_frame_jpg = None

def set_latest_frame(frame_bgr):
    """Call this from detection thread to push frames to mobile app."""
    global _latest_frame_jpg
    import cv2 as _cv2
    _, buf = _cv2.imencode(".jpg", frame_bgr, [_cv2.IMWRITE_JPEG_QUALITY, 70])
    _latest_frame_jpg = buf.tobytes()


@app.route("/api/stats")
def api_stats():
    """Latest vehicle counts for mobile dashboard."""
    df = load_all_logs()
    if df.empty:
        return jsonify({"total":0,"car":0,"rick":0,"bus":0,"log":"No data yet."})
    by_type  = df["vehicle_type"].value_counts().to_dict()
    total    = len(df)
    bus_truck = by_type.get("bus",0) + by_type.get("truck",0)
    last_log = df.sort_values("timestamp").iloc[-1]
    log_msg  = f"{last_log['vehicle_type']} detected at {last_log['timestamp']}"
    return jsonify({
        "total": total,
        "car":   by_type.get("car", 0),
        "rick":  by_type.get("rickshaw/motorcycle", 0),
        "bus":   bus_truck,
        "log":   log_msg,
    })


@app.route("/api/frame")
def api_frame():
    """Latest annotated video frame as JPEG (for mobile live feed)."""
    if _latest_frame_jpg is None:
        # Return a 1x1 transparent pixel if no frame available
        import base64
        empty = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00' \
                b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t' \
                b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a' \
                b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e' \
                b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4' \
                b'\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00' \
                b'\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b' \
                b'\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05' \
                b'\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06' \
                b'\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br' \
                b'\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZ' \
                b'cdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94' \
                b'\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa' \
                b'\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7' \
                b'\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3' \
                b'\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8' \
                b'\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00\x00' \
                b'\x00\x1f\xff\xd9'
        return send_file(_io.BytesIO(empty), mimetype="image/jpeg")
    return send_file(_io.BytesIO(_latest_frame_jpg), mimetype="image/jpeg")


@app.route("/api/setting", methods=["POST"])
def api_setting():
    """Update a config.py setting from the mobile app."""
    from flask import request as req
    data = req.get_json(force=True) or {}
    key  = data.get("key","")
    val  = data.get("value")
    try:
        with open("config.py") as f: c = f.read()
        if isinstance(val, bool):
            c = _re.sub(rf"{key}\s*=\s*\w+", f"{key} = {val}", c)
        elif isinstance(val, (int,float)):
            c = _re.sub(rf"{key}\s*=\s*[\d.]+", f"{key} = {val}", c)
        elif isinstance(val, str):
            c = _re.sub(rf'{key}\s*=\s*"[^"]*"', f'{key} = "{val}"', c)
        with open("config.py","w") as f: f.write(c)
        return jsonify({"ok":True})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}), 500


@app.route("/api/trigger", methods=["POST"])
def api_trigger():
    """Placeholder: mobile app can signal intent to start detection."""
    return jsonify({"ok":True, "message":"Start detection from the laptop app."})


@app.route("/android")
def android_ui():
    """Serve the Android companion app HTML."""
    try:
        with open("android_app/index.html") as f:
            return f.read(), 200, {"Content-Type":"text/html"}
    except FileNotFoundError:
        return "Android app not found.", 404


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(config.DATA_FOLDER, exist_ok=True)
    print("\n[DASHBOARD] Starting web dashboard...")
    print("  Open your browser at:  http://localhost:5000\n")
    app.run(debug=False, port=5000)
