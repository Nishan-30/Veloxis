# ================================================================
#  VELOXIS  —  app_windows.py  (v1.0)
#
#  Author  : Nishan, SUST CEE
#  Product : VELOXIS · an app of NextCity Tessera
#  Year    : 2026
#  License : MIT
#
#  Light/Dark mode: uses CTk native theming only — works instantly.
#  All colors from CTk theme system — no hardcoded hex.
# ================================================================

import tkinter as tk
import customtkinter as ctk
import threading, os, sys, datetime, json, re, glob, time, queue, math
import cv2, numpy as np, subprocess
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd

# ── Prefs ──────────────────────────────────────────────────────
PREFS_FILE = "data/user_prefs.json"

def load_prefs() -> dict:
    try:
        with open(PREFS_FILE, encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_prefs(d: dict):
    os.makedirs("data", exist_ok=True)
    p = load_prefs(); p.update(d)
    with open(PREFS_FILE, "w", encoding="utf-8") as f: json.dump(p, f, indent=2)

# ── Map Report Generator ───────────────────────────────────────
def generate_map_report(session_label=None):
    """
    Generate a self-contained HTML map report.
    Embeds OpenStreetMap via Leaflet.js (no API key needed).
    Includes: study site pin, volume summary, key metrics table.
    Saves to data/map_report_<timestamp>.html and opens in browser.
    Returns the path or None on failure.
    """
    import webbrowser, glob as _glob
    p = load_prefs()
    try:
        lat  = float(p.get("loc_lat") or 0)
        lng  = float(p.get("loc_lng") or 0)
    except (ValueError, TypeError):
        lat = lng = 0.0

    site_name = p.get("loc_name","Unknown Site") or "Study Intersection"
    has_coords = (lat != 0 and lng != 0)

    # Load session data
    files = _glob.glob(os.path.join("data","log_*.csv"))
    total = fwd = bwd = 0
    bt    = {}
    phf   = v85 = los = headway = satflow = "—"
    duration_hrs = 0.0
    t_start = t_end = "—"
    speed_note = ""

    try:
        dfs = [d for d in [pd.read_csv(f) for f in files] if not d.empty]
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            if session_label and "session" in df.columns:
                df = df[df["session"]==session_label]
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                t_start = df["timestamp"].min().strftime("%Y-%m-%d %H:%M") \
                    if pd.notna(df["timestamp"].min()) else "—"
                t_end   = df["timestamp"].max().strftime("%Y-%m-%d %H:%M") \
                    if pd.notna(df["timestamp"].max()) else "—"
                dur_sec = (df["timestamp"].max()-df["timestamp"].min()).total_seconds()
                duration_hrs = max(dur_sec/3600, 0.017)
            total = len(df)
            if "vehicle_type" in df.columns:
                bt = df["vehicle_type"].value_counts().to_dict()
            if "direction" in df.columns:
                fwd = df["direction"].str.contains("FWD",na=False).sum()
                bwd = total - fwd
            # Pull metrics from summary CSV if available
            sumfiles = _glob.glob(os.path.join("data","*_summary.csv"))
            if sumfiles:
                sdf = pd.read_csv(sorted(sumfiles)[-1])
                if not sdf.empty:
                    row = sdf.iloc[-1]
                    phf      = f"{float(row.get('phf','0')):.2f}" if row.get('phf') else "—"
                    v85      = f"{float(row.get('speed_85th_kmh','0')):.0f}" if row.get('speed_85th_kmh') else "—"
                    los      = str(row.get('los_letter','—'))
                    headway  = f"{float(row.get('avg_headway_sec','0')):.1f}" if row.get('avg_headway_sec') else "—"
                    satflow  = str(int(float(row.get('saturation_flow_vph',0)))) if row.get('saturation_flow_vph') else "—"
                    # Speed vs limit note
                    try:
                        lim_raw = p.get("speed_limit","50")
                        lim_val = int(str(lim_raw).replace("*",""))
                        v85_val = float(row.get('speed_85th_kmh',0))
                        if v85_val and lim_val:
                            diff = v85_val - lim_val
                            if diff > 10:   speed_note = f"⚠️ V85 exceeds limit by {diff:.0f} km/h"
                            elif diff > 0:  speed_note = f"↑ V85 slightly above posted limit"
                            else:           speed_note = f"✓ V85 within posted speed limit"
                    except: pass
    except Exception: pass

    # Build vehicle rows
    vehicle_rows = ""
    icons = {"car":"🚙","rickshaw":"🛺","cng":"🛺","motorcycle":"🏍",
             "bus":"🚌","truck":"🚛","bicycle":"🚲","easybike":"⚡",
             "battery_rickshaw":"🔋","human_hauler":"🚐","leguna":"🚐",
             "nosimon":"🚜","microbus":"🚐","pickup":"🚚","tempo":"🚐"}
    vol_rows = ""
    for vt, cnt in sorted(bt.items(), key=lambda x: -x[1]):
        icon = icons.get(vt.lower().replace("/","_").replace(" ","_"), "🚗")
        vph  = f"{cnt/duration_hrs:.0f}" if duration_hrs>0 else "—"
        vol_rows += f"<tr><td>{icon} {vt}</td><td>{cnt}</td><td>{vph}</td></tr>\n"

    # Map section — only if coords available
    map_section = ""
    if has_coords:
        map_section = f"""
        <div id="map"></div>
        <script>
          var map = L.map('map').setView([{lat}, {lng}], 17);
          L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{
            attribution:'© OpenStreetMap contributors', maxZoom:19}}).addTo(map);
          var icon = L.divIcon({{
            html:'<div style="font-size:28px;line-height:1">📍</div>',
            className:'',iconAnchor:[14,28]}});
          L.marker([{lat},{lng}],{{icon:icon}}).addTo(map)
            .bindPopup('<b>{site_name}</b><br>Lat {lat:.5f}, Lng {lng:.5f}<br>Total: {total} vehicles')
            .openPopup();
        </script>"""
    else:
        map_section = """<div id="map" style="display:flex;align-items:center;
            justify-content:center;color:#64748b;font-size:14px">
            No GPS coordinates saved — set location in Settings to enable map</div>"""

    los_colours = {"A":"#16a34a","B":"#65a30d","C":"#ca8a04",
                   "D":"#ea580c","E":"#dc2626","F":"#7f1d1d","—":"#64748b"}
    los_col = los_colours.get(los, "#64748b")

    speed_note_html = f'<div class="speed-note">{speed_note}</div>' if speed_note else ""
    road_type = p.get("road_type","—")
    speed_limit = p.get("speed_limit","—")

    # Build approach volume rows for map report
    approach_rows = ""
    try:
        tmc_files = _glob.glob(os.path.join("data","*_tmc.csv"))
        if tmc_files:
            tdf = pd.read_csv(sorted(tmc_files)[-1], index_col=0)
            tdf = tdf.drop(columns=["TOTAL"], errors="ignore")
            if not tdf.empty:
                for arm in tdf.index:
                    arm_total = int(tdf.loc[arm].sum())
                    exits = "  ".join(f"{col}:{int(tdf.loc[arm,col])}"
                                      for col in tdf.columns if tdf.loc[arm,col]>0)
                    approach_rows += f"<tr><td><b>{arm}</b></td><td>{arm_total}</td><td style='font-size:11px;color:#64748b'>{exits}</td></tr>\n"
    except Exception:
        pass

    tmc_section = ""
    if approach_rows:
        tmc_section = f"""
    <div class="panel" style="margin-bottom:20px">
      <h2>Turning Movement Counts — Approach Summary</h2>
      <table>
        <thead><tr><th>Approach</th><th>Total</th><th>Exit breakdown</th></tr></thead>
        <tbody>{approach_rows}</tbody>
      </table>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VELOXIS — Traffic Map Report</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}
header{{background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:20px 32px;
  border-bottom:3px solid #3b82f6;display:flex;align-items:center;gap:18px}}
header h1{{font-size:22px;font-weight:700;color:#60a5fa}}
header .sub{{font-size:13px;color:#94a3b8;margin-top:3px}}
.badge{{background:#3b82f6;color:#fff;font-size:11px;padding:3px 10px;
  border-radius:20px;font-weight:600;margin-left:10px}}
.container{{max-width:1100px;margin:0 auto;padding:24px 20px}}
#map{{height:400px;border-radius:12px;margin-bottom:24px;border:1px solid #1e293b}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px}}
.card{{background:#1e293b;border-radius:10px;padding:14px 16px;border-top:3px solid}}
.card .label{{font-size:10px;color:#64748b;letter-spacing:.05em;text-transform:uppercase;margin-bottom:6px}}
.card .val{{font-size:24px;font-weight:700}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}}
@media(max-width:640px){{.grid2{{grid-template-columns:1fr}}}}
.panel{{background:#1e293b;border-radius:12px;padding:20px;border:1px solid #1e3a5f}}
.panel h2{{font-size:13px;font-weight:600;color:#94a3b8;margin-bottom:14px;
  text-transform:uppercase;letter-spacing:.06em}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;color:#64748b;font-size:11px;padding:6px 8px;
  border-bottom:1px solid #1e3a5f;text-transform:uppercase}}
td{{padding:8px;border-bottom:1px solid #1e3a5f}}
tr:last-child td{{border-bottom:none}}
.los-badge{{display:inline-block;padding:3px 12px;border-radius:20px;
  font-weight:700;font-size:18px;color:#fff;background:{los_col}}}
.speed-note{{margin-top:10px;padding:8px 12px;background:#1e293b;
  border-radius:8px;font-size:12px;color:#94a3b8;border-left:3px solid #3b82f6}}
.meta{{background:#0f172a;border-radius:10px;padding:14px 18px;margin-bottom:20px;
  border:1px solid #1e293b;font-size:12px;color:#64748b;line-height:2}}
footer{{text-align:center;padding:24px;font-size:11px;color:#475569;border-top:1px solid #1e293b;margin-top:12px}}
</style>
</head>
<body>
<header>
  <div style="font-size:32px">🚦</div>
  <div>
    <h1>VELOXIS — Traffic Analysis Report<span class="badge">NextCity Tessera</span></h1>
    <div class="sub">{site_name} &nbsp;·&nbsp; {t_start} → {t_end} &nbsp;·&nbsp; {road_type}</div>
  </div>
</header>
<div class="container">
  <div class="meta">
    📍 <b>Site:</b> {site_name} &nbsp;|&nbsp;
    🗓 <b>Period:</b> {t_start} – {t_end} &nbsp;|&nbsp;
    🛣 <b>Road type:</b> {road_type} &nbsp;|&nbsp;
    🚦 <b>Speed limit:</b> {speed_limit} km/h &nbsp;|&nbsp;
    📐 <b>Coords:</b> {lat:.5f}, {lng:.5f}
  </div>

  <div class="cards">
    <div class="card" style="border-color:#3b82f6">
      <div class="label">Total Vehicles</div>
      <div class="val" style="color:#3b82f6">{total}</div>
    </div>
    <div class="card" style="border-color:#2dd4bf">
      <div class="label">Forward</div>
      <div class="val" style="color:#2dd4bf">{fwd}</div>
    </div>
    <div class="card" style="border-color:#fbbf24">
      <div class="label">Backward</div>
      <div class="val" style="color:#fbbf24">{bwd}</div>
    </div>
    <div class="card" style="border-color:{los_col}">
      <div class="label">LOS</div>
      <div class="val"><span class="los-badge">{los}</span></div>
    </div>
    <div class="card" style="border-color:#a78bfa">
      <div class="label">PHF</div>
      <div class="val" style="color:#a78bfa">{phf}</div>
    </div>
    <div class="card" style="border-color:#34d399">
      <div class="label">V85 (km/h)</div>
      <div class="val" style="color:#34d399">{v85}</div>
    </div>
    <div class="card" style="border-color:#fb923c">
      <div class="label">Avg Headway</div>
      <div class="val" style="color:#fb923c">{headway}s</div>
    </div>
    <div class="card" style="border-color:#60a5fa">
      <div class="label">Sat. Flow</div>
      <div class="val" style="color:#60a5fa">{satflow}</div>
    </div>
  </div>

  {map_section}

  <div class="grid2">
    <div class="panel">
      <h2>Volume by Vehicle Type</h2>
      <table>
        <thead><tr><th>Type</th><th>Count</th><th>Veh/hr</th></tr></thead>
        <tbody>{vol_rows}</tbody>
      </table>
    </div>
    <div class="panel">
      <h2>Intersection Capacity Analysis</h2>
      <table>
        <tr><td>Level of Service</td><td><span class="los-badge">{los}</span></td></tr>
        <tr><td>Peak Hour Factor (PHF)</td><td><b>{phf}</b></td></tr>
        <tr><td>85th Percentile Speed</td><td><b>{v85} km/h</b></td></tr>
        <tr><td>Posted Speed Limit</td><td>{speed_limit} km/h</td></tr>
        <tr><td>Average Headway</td><td>{headway} sec/veh</td></tr>
        <tr><td>Saturation Flow</td><td>{satflow} veh/hr</td></tr>
        <tr><td>Total Volume</td><td>{total} vehicles</td></tr>
        <tr><td>FWD / BWD Split</td><td>{fwd} / {bwd}</td></tr>
      </table>
      {speed_note_html}
    </div>
  </div>
  {tmc_section}
</div>
<footer>
  Generated by VELOXIS v2.0 · Nishan, SUST CEE · NextCity Tessera · {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
</footer>
</body>
</html>"""

    os.makedirs("data", exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("data", f"map_report_{ts}.html")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        webbrowser.open(f"file:///{os.path.abspath(path)}")
        return path
    except Exception as e:
        print(f"[WARN] Map report save failed: {e}")
        return None


def generate_map_report_multi(session_labels):
    """
    Generate one combined HTML report for multiple selected sessions.
    Each session gets its own metrics row in a comparison table.
    The map shows one pin (study site from prefs — same location assumed).
    """
    import webbrowser, glob as _glob
    p = load_prefs()
    try:
        lat = float(p.get("loc_lat") or 0)
        lng = float(p.get("loc_lng") or 0)
    except (ValueError, TypeError):
        lat = lng = 0.0
    site_name  = p.get("loc_name","Study Intersection") or "Study Intersection"
    has_coords = (lat != 0 and lng != 0)
    road_type  = p.get("road_type","—")
    speed_limit = p.get("speed_limit","—")

    # Load all log CSVs then filter per session
    files = _glob.glob(os.path.join("data","log_*.csv"))
    try:
        all_dfs = [d for d in [pd.read_csv(f) for f in files] if not d.empty]
        full_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        if "timestamp" in full_df.columns:
            full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], errors="coerce")
    except Exception:
        full_df = pd.DataFrame()

    # Load summary CSVs
    sumfiles = sorted(_glob.glob(os.path.join("data","*_summary.csv")))
    try:
        sum_df = pd.concat([pd.read_csv(f) for f in sumfiles], ignore_index=True) if sumfiles else pd.DataFrame()
    except Exception:
        sum_df = pd.DataFrame()

    icons = {"car":"🚙","rickshaw":"🛺","cng":"🛺","motorcycle":"🏍",
             "bus":"🚌","truck":"🚛","bicycle":"🚲","easybike":"⚡",
             "battery_rickshaw":"🔋","human_hauler":"🚐","leguna":"🚐",
             "nosimon":"🚜","microbus":"🚐","pickup":"🚚","tempo":"🚐"}

    # Build per-session rows
    session_rows_html = ""
    grand_total = 0
    type_totals = {}

    for lbl in session_labels:
        df_s = full_df[full_df["session"]==lbl] if "session" in full_df.columns else pd.DataFrame()
        cnt  = len(df_s)
        grand_total += cnt

        t_start = t_end = "—"
        if not df_s.empty and "timestamp" in df_s.columns:
            t_start = df_s["timestamp"].min().strftime("%Y-%m-%d %H:%M") if pd.notna(df_s["timestamp"].min()) else "—"
            t_end   = df_s["timestamp"].max().strftime("%Y-%m-%d %H:%M") if pd.notna(df_s["timestamp"].max()) else "—"

        # Vehicle type breakdown for this session
        bt = {}
        if not df_s.empty and "vehicle_type" in df_s.columns:
            bt = df_s["vehicle_type"].value_counts().to_dict()
        for vt, c in bt.items():
            type_totals[vt] = type_totals.get(vt,0) + c

        # Direction
        fwd = bwd = 0
        if not df_s.empty and "direction" in df_s.columns:
            fwd = df_s["direction"].str.contains("FWD",na=False).sum()
            bwd = cnt - fwd

        # Metrics from summary CSV
        phf = v85 = los = hdwy = sat = "—"
        if not sum_df.empty and "session" in sum_df.columns:
            sr = sum_df[sum_df["session"]==lbl]
            if not sr.empty:
                row = sr.iloc[-1]
                phf = f"{float(row.get('phf',0)):.2f}"          if row.get('phf') else "—"
                v85 = f"{float(row.get('speed_85th_kmh',0)):.0f}" if row.get('speed_85th_kmh') else "—"
                los = str(row.get('los_letter','—'))
                hdwy= f"{float(row.get('avg_headway_sec',0)):.1f}" if row.get('avg_headway_sec') else "—"
                sat = str(int(float(row.get('saturation_flow_vph',0)))) if row.get('saturation_flow_vph') else "—"

        type_str = "  ".join(f"{icons.get(k.lower(),'🚗')}{k}:{v}" for k,v in list(bt.items())[:5])
        los_cols = {"A":"#16a34a","B":"#65a30d","C":"#ca8a04","D":"#ea580c","E":"#dc2626","F":"#7f1d1d","—":"#64748b"}
        los_col  = los_cols.get(los,"#64748b")

        session_rows_html += f"""
        <tr>
          <td style="font-size:11px;color:#94a3b8;white-space:nowrap">{lbl[:28]}</td>
          <td style="font-size:11px;color:#64748b">{t_start}</td>
          <td><b>{cnt}</b></td>
          <td>{fwd}</td><td>{bwd}</td>
          <td><span style="background:{los_col};color:#fff;padding:2px 8px;border-radius:12px;font-weight:700">{los}</span></td>
          <td>{phf}</td><td>{v85}</td><td>{hdwy}s</td><td>{sat}</td>
        </tr>
        <tr><td colspan="10" style="padding:2px 8px 8px;font-size:10px;color:#475569">{type_str}</td></tr>
"""

    # Grand total type breakdown
    vol_rows = ""
    dur_total = 0
    for vt, cnt_t in sorted(type_totals.items(), key=lambda x: -x[1]):
        icon = icons.get(vt.lower().replace("/","_").replace(" ","_"),"🚗")
        vol_rows += f"<tr><td>{icon} {vt}</td><td>{cnt_t}</td><td>{round(cnt_t/max(dur_total,1),0) if dur_total else '—'}</td></tr>\n"

    # Map
    map_section = f"""
    <div id="map"></div>
    <script>
      var map = L.map('map').setView([{lat},{lng}],17);
      L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{
        attribution:'© OpenStreetMap contributors',maxZoom:19}}).addTo(map);
      var icon=L.divIcon({{html:'<div style="font-size:28px;line-height:1">📍</div>',className:'',iconAnchor:[14,28]}});
      L.marker([{lat},{lng}],{{icon:icon}}).addTo(map)
        .bindPopup('<b>{site_name}</b><br>{len(session_labels)} sessions · {grand_total} vehicles total')
        .openPopup();
    </script>""" if has_coords else """<div id="map" style="display:flex;align-items:center;
        justify-content:center;color:#64748b;font-size:14px">No GPS coordinates — set location in Settings</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VELOXIS — Multi-Session Report</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}
header{{background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:20px 32px;border-bottom:3px solid #3b82f6;display:flex;align-items:center;gap:18px}}
header h1{{font-size:20px;font-weight:700;color:#60a5fa}}
.badge{{background:#3b82f6;color:#fff;font-size:11px;padding:3px 10px;border-radius:20px;font-weight:600;margin-left:10px}}
.container{{max-width:1200px;margin:0 auto;padding:24px 20px}}
#map{{height:360px;border-radius:12px;margin-bottom:24px;border:1px solid #1e293b}}
.summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:24px}}
.card{{background:#1e293b;border-radius:10px;padding:12px 14px;border-top:3px solid}}
.card .label{{font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}}
.card .val{{font-size:22px;font-weight:700}}
.panel{{background:#1e293b;border-radius:12px;padding:20px;border:1px solid #1e3a5f;margin-bottom:20px}}
.panel h2{{font-size:12px;font-weight:600;color:#94a3b8;margin-bottom:14px;text-transform:uppercase;letter-spacing:.06em}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;color:#64748b;font-size:10px;padding:6px 8px;border-bottom:1px solid #1e3a5f;text-transform:uppercase;white-space:nowrap}}
td{{padding:6px 8px;border-bottom:1px solid #0f172a;vertical-align:middle}}
tr:hover td{{background:#0f172a20}}
footer{{text-align:center;padding:20px;font-size:11px;color:#475569;border-top:1px solid #1e293b;margin-top:12px}}
</style>
</head>
<body>
<header>
  <div style="font-size:30px">🚦</div>
  <div>
    <h1>VELOXIS — Multi-Session Report<span class="badge">{len(session_labels)} sessions</span></h1>
    <div style="font-size:12px;color:#94a3b8;margin-top:3px">{site_name} &nbsp;·&nbsp; {road_type} &nbsp;·&nbsp; Speed limit: {speed_limit} km/h</div>
  </div>
</header>
<div class="container">
  <div class="summary-cards">
    <div class="card" style="border-color:#3b82f6"><div class="label">Total Vehicles</div><div class="val" style="color:#3b82f6">{grand_total}</div></div>
    <div class="card" style="border-color:#2dd4bf"><div class="label">Sessions</div><div class="val" style="color:#2dd4bf">{len(session_labels)}</div></div>
    <div class="card" style="border-color:#a78bfa"><div class="label">Vehicle Types</div><div class="val" style="color:#a78bfa">{len(type_totals)}</div></div>
  </div>

  {map_section}

  <div class="panel">
    <h2>Session Comparison</h2>
    <table>
      <thead><tr>
        <th>Session</th><th>Start</th><th>Total</th>
        <th>FWD</th><th>BWD</th><th>LOS</th>
        <th>PHF</th><th>V85</th><th>Headway</th><th>Sat.Flow</th>
      </tr></thead>
      <tbody>{session_rows_html}</tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Combined Volume by Vehicle Type</h2>
    <table>
      <thead><tr><th>Type</th><th>Count</th><th>Veh/hr</th></tr></thead>
      <tbody>{vol_rows}</tbody>
    </table>
  </div>
</div>
<footer>Generated by VELOXIS v2.0 · NextCity Tessera · {__import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M")}</footer>
</body></html>"""

    os.makedirs("data", exist_ok=True)
    ts   = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("data", f"map_report_multi_{ts}.html")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        webbrowser.open(f"file:///{os.path.abspath(path)}")
        return path
    except Exception as e:
        print(f"[WARN] Multi map report save failed: {e}")
        return None



_prefs = load_prefs()
_THEME = _prefs.get("theme", "dark")
ctk.set_appearance_mode(_THEME)
ctk.set_default_color_theme("blue")

# Accent colours — brighter, higher saturation for premium dark-mode feel
ACC_BLUE   = "#4f8ef7"   # was #3b82f6 — brighter royal blue
ACC_GREEN  = "#3ddba8"   # was #34d399 — vivid emerald
ACC_AMBER  = "#fdc040"   # was #fbbf24 — warmer gold
ACC_RED    = "#ff6b7a"   # was #f87171 — punchier coral-red
ACC_PURPLE = "#b98df7"   # was #a78bfa — richer violet
ACC_TEAL   = "#2fe6d4"   # was #2dd4bf — electric teal
LANE_COLS  = ["#2fe6d4","#fdc040","#ff6b7a","#b98df7","#3ddba8","#fb923c"]

# Chart palette — vivid series colours for matplotlib
CHART_PALETTE = ["#4f8ef7","#2fe6d4","#fdc040","#ff6b7a","#b98df7",
                 "#3ddba8","#fb923c","#f472b6","#38bdf8"]

# Vehicle icon map
V_ICONS = {
    "car":          ("🚙", ACC_TEAL),
    "rickshaw":     ("🛺", ACC_AMBER),
    "CNG/auto":     ("🛺", "#fb923c"),
    "rickshaw/CNG": ("🛺", ACC_AMBER),
    "motorcycle":   ("🏍", ACC_RED),
    "bus":          ("🚌", ACC_BLUE),
    "truck":        ("🚛", ACC_PURPLE),
    "bicycle":      ("🚲", ACC_GREEN),
    "train":        ("🚆", "#60a5fa"),
}


# ================================================================
#  AI COUNTING LINE DETECTOR
# ================================================================
class AILineDetector:
    def __init__(self, n=35):
        self.n=n; self.frames=[]; self.ready=False
        self.line_start=None; self.line_end=None; self.position=0.55
    def feed(self, frame):
        if self.ready: return True
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.frames.append(cv2.resize(gray,(320,180)))
        if len(self.frames)>=self.n: self._analyse(); return True
        return False
    def _analyse(self):
        vx_list, vy_list = [], []
        h_frame, w_frame = self.frames[0].shape[:2]

        # Only analyse bottom 60% of frame — ignore sky and background
        # Sky has more movement from camera shake, not from vehicles
        road_top = int(h_frame * 0.35)

        for i in range(0, len(self.frames)-1, 2):
            f1 = self.frames[i][road_top:, :]
            f2 = self.frames[i+1][road_top:, :]
            flow = cv2.calcOpticalFlowFarneback(
                f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            # Only strong motion — likely vehicles not background drift
            thresh = np.percentile(mag, 80)
            mask = mag > thresh
            if mask.sum() < 50:   # too few moving pixels — skip
                continue
            vx_list.append(float(flow[...,0][mask].mean()))
            vy_list.append(float(flow[...,1][mask].mean()))

        if not vx_list:
            # Fallback: horizontal line at 60% of frame
            self.line_start = (0.0, 0.60)
            self.line_end   = (1.0, 0.60)
            self.ready = True
            return

        vx, vy = np.mean(vx_list), np.mean(vy_list)

        # Flow perpendicular = counting line direction
        perp = np.degrees(np.arctan2(vy, vx)) + 90
        rad  = np.radians(perp)
        dx, dy = np.cos(rad), np.sin(rad)

        # Centre of counting line: 60% down, middle horizontally
        cx = w_frame * 0.5
        cy = h_frame * 0.60   # well into road area, not sky
        t  = max(w_frame, h_frame) * 1.5

        # Clamp to frame bounds (fractions 0-1)
        x1 = max(0.0, min(1.0, (cx - dx*t) / w_frame))
        y1 = max(0.0, min(1.0, (cy - dy*t) / h_frame))
        x2 = max(0.0, min(1.0, (cx + dx*t) / w_frame))
        y2 = max(0.0, min(1.0, (cy + dy*t) / h_frame))

        # Sanity check: if line is mostly in top third, force it down
        mid_y = (y1 + y2) / 2
        if mid_y < 0.4:
            self.line_start = (0.0, 0.60)
            self.line_end   = (1.0, 0.60)
        else:
            self.line_start = (x1, y1)
            self.line_end   = (x2, y2)
        self.ready = True
    def get_line_px(self,w,h):
        if self.line_start:
            return (int(self.line_start[0]*w),int(self.line_start[1]*h)),\
                   (int(self.line_end[0]*w),  int(self.line_end[1]*h))
        ly=int(h*self.position); return (0,ly),(w,ly)
    def progress(self): return len(self.frames)/self.n


# ================================================================
#  DETECTION THREAD
# ================================================================
class DetectionThread(threading.Thread):
    def __init__(self, source, mode, on_status, on_done,
                 on_progress=None, use_ai=True, conf_ref=None,
                 manual_line=None):
        super().__init__(daemon=True)
        self.source=source; self.mode=mode
        self.on_status=on_status; self.on_done=on_done
        self.on_progress=on_progress; self.use_ai=use_ai
        self.conf_ref=conf_ref; self.manual_line=manual_line
        self._stop=threading.Event()
        # Only use AI if no manual line set
        self.ai=AILineDetector(35) if (use_ai and manual_line is None) else None
        self.calibrating=(use_ai and manual_line is None)
        self.frame_q=queue.Queue(maxsize=2)
        self.fps=0.0; self._ft=time.time(); self._ff=0

    def stop(self): self._stop.set()

    def _push(self, frame, summary):
        try: self.frame_q.put_nowait((frame,summary))
        except queue.Full:
            try: self.frame_q.get_nowait()
            except queue.Empty: pass
            try: self.frame_q.put_nowait((frame,summary))
            except queue.Full: pass
        self._ff+=1
        now=time.time()
        if now-self._ft>=1.0:
            self.fps=self._ff/(now-self._ft); self._ff=0; self._ft=now

    def run(self):
        try:
            import config
            from detector import VehicleDetector
        except ImportError as e:
            self.on_status(f"ERROR: {e}"); return
        self.on_status("Loading YOLO model…")
        lbl=f"{'live' if self.mode=='live' else 'file'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        self._session_label = lbl   # stored so UI can pass to map report

        # Check model file before creating detector — show warning in UI if fallback
        import config as _cfg
        _model = _cfg.YOLO_MODEL
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _model_exists = os.path.exists(_model) or os.path.exists(os.path.join(_script_dir, _model))
        _fallbacks = ["bd_vehicles_best.pt","yolo11s.pt","yolov8s.pt","yolov8n.pt"]
        if not _model_exists:
            _found = next((f for f in _fallbacks
                           if os.path.exists(f) or os.path.exists(os.path.join(_script_dir,f))), None)
            if _found:
                self.on_status(f"⚠️  {_model} not found — using fallback: {_found}  (check Settings)")
            else:
                self.on_status(f"⚠️  No model file found — downloading yolo11s.pt (internet required)")

        det=VehicleDetector(session_label=lbl)

        # Apply manual line — takes priority over AI
        if self.manual_line is not None:
            det.manual_line_a = max(0.05, min(0.95, self.manual_line))
            self.on_status(f"Manual line at {int(self.manual_line*100)}% — Detecting…")

        cap=(cv2.VideoCapture(self.source,cv2.CAP_DSHOW)
             if isinstance(self.source,int) else cv2.VideoCapture(self.source))
        if not cap.isOpened(): self.on_status("Cannot open source."); cap.release(); return
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        # Auto-detect video FPS and update config
        detected_fps = cap.get(cv2.CAP_PROP_FPS)
        if detected_fps and 5 < detected_fps < 120:
            import config as cfg
            cfg.VIDEO_FPS = detected_fps
            self.on_status(f"Video FPS auto-detected: {detected_fps:.1f}")

        fn=0
        _t_last_frame = time.time()
        _last_progress_pct = -1   # dedup guard — only fire on integer % change (max 101 calls)
        try:
            while not self._stop.is_set():
                ret,frame=cap.read()
                if not ret:
                    if self.mode=="live": time.sleep(0.03); continue
                    break
                fn+=1
                if self.conf_ref:
                    import config as cfg; cfg.CONFIDENCE=self.conf_ref[0]
    
                # Live mode: if processing is slower than camera, drop frames to stay current
                # This prevents counting line misses from accumulated lag
                if self.mode=="live":
                    _now = time.time()
                    _elapsed = _now - _t_last_frame
                    # If we're more than 2 frames behind (~67ms at 30fps), drain buffer
                    if _elapsed < 0.02 and not self.frame_q.empty():
                        continue  # skip this frame, read next
                    _t_last_frame = _now
    
                # AI calibration — only if no manual line
                if self.ai and self.calibrating and det.manual_line_a is None:
                    done=self.ai.feed(frame)
                    if not done:
                        pct=int(self.ai.progress()*100)
                        self.on_status(f"AI analysing traffic flow… {pct}%")
                        h,w=frame.shape[:2]
                        ov=frame.copy()
                        cv2.rectangle(ov,(0,0),(w,h),(10,14,22),cv2.FILLED)
                        cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
                        bw=int(w*0.55); bx=(w-bw)//2; by=h//2
                        cv2.rectangle(frame,(bx,by-10),(bx+bw,by+10),(30,35,50),-1)
                        cv2.rectangle(frame,(bx,by-10),(bx+int(bw*pct/100),by+10),(45,180,140),-1)
                        cv2.putText(frame,f"AI calibrating…  {pct}%",
                                    (bx,by-18),cv2.FONT_HERSHEY_SIMPLEX,0.62,(190,190,200),1)
                        self._push(frame,{}); continue
                    else:
                        self.calibrating=False
                        if self.ai.line_start and det.manual_line_a is None:
                            det.ai_line_start=self.ai.line_start
                            det.ai_line_end=self.ai.line_end
                        self.on_status("Detecting…")
                ann,summary=det.process_frame(frame)
                cv2.putText(ann,f"FPS:{self.fps:.1f}",
                            (ann.shape[1]-88,18),cv2.FONT_HERSHEY_SIMPLEX,0.48,(57,197,187),1)
                self._push(ann,summary)
                if self.on_progress and self.mode=="file":
                    _pct = int(fn / total * 100)
                    if _pct != _last_progress_pct:   # fire only on integer change
                        _last_progress_pct = _pct
                        self.on_progress(_pct)
                    # Throttle to ~real-time playback based on video FPS
                    # This prevents the analysis thread racing ahead of the display
                    # while keeping live detection at full speed
                    _target_spf = 1.0 / max(detected_fps, 10)  # seconds per frame
                    time.sleep(max(0.005, _target_spf * 0.4))   # 40% of frame time = sustainable
        except Exception as _loop_exc:
            # Unhandled exception in detection loop — log it, fall through to cleanup
            import traceback
            print(f"[ERROR] Detection loop crashed: {_loop_exc}")
            traceback.print_exc()
            self.on_status(f"⚠️  Session interrupted — saving data…")
        finally:
            cap.release()
        # Save session summary CSV with all metrics
        det.save_session_summary()
        self.on_done({
            "total_unique":   len(det.counted_ids),
            "by_type":        det.total_counts,
            "phf":            det.phf,
            "peak_rate":      det.peak_rate,
            "avg_headway":    det.avg_headway_sec,
            "saturation":     det.saturation_flow,
            "saturation_flow":det.saturation_flow,
            "speed_85th":     det.speed_85th,
            "speed_mean":     det.speed_mean,
            "safety_events":  det.safety_events,
            "los_letter":     det.los_letter,
            "avg_delay_sec":  det.avg_delay_sec,
            "turning_counts": det.turning_counts,
            "approach_counts":det.approach_counts,
            "session_label":  lbl,
        })
        self.on_status("Session complete ✓")


# ================================================================
#  REUSABLE WIDGETS  (all use CTk native theming)
# ================================================================

class StatCard(ctk.CTkFrame):
    """Premium stat card — thick accent bar, glowing value."""
    def __init__(self, master, label, value="—", accent=ACC_BLUE, icon="", **kw):
        super().__init__(master, corner_radius=14, border_width=1, **kw)
        self.grid_columnconfigure(0, weight=1)
        # Thick top accent bar
        ctk.CTkFrame(self, fg_color=accent, height=4, corner_radius=0
                    ).grid(row=0, column=0, sticky="ew")
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=1, column=0, padx=14, pady=(10, 0), sticky="ew")
        ctk.CTkLabel(top, text=icon, font=("Segoe UI", 14)
                    ).pack(side="left", padx=(0, 5))
        ctk.CTkLabel(top, text=label.upper(),
                     font=("Segoe UI", 8, "bold"),
                     text_color="#64748b").pack(side="left")
        self._val = ctk.CTkLabel(self, text=str(value),
                                 font=("Segoe UI", 28, "bold"),
                                 text_color=accent)
        self._val.grid(row=2, column=0, padx=14, pady=(2, 14), sticky="w")

    def set(self, v): self._val.configure(text=str(v))


class ClickableVideoCanvas(ctk.CTkLabel):
    """
    Video display with click-to-set counting line.
    - Click anywhere on video → sets horizontal counting line at that Y position
    - Drag to move the line
    - Right-click → clear the line (revert to AI/default)
    """
    def __init__(self, master, on_line_set=None,
                 placeholder="Browse a video → then click on frame to set counting line", **kw):
        super().__init__(master, text=placeholder, corner_radius=14,
                         font=("Segoe UI",13), **kw)
        self._img        = None
        self._line_frac  = None
        self._frame_orig = None
        self.on_line_set = on_line_set
        # Stored after each _render so _set_line_at can compute correct fraction
        self._nh = None   # scaled frame height (pixels)
        self._nw = None   # scaled frame width  (pixels)
        self.bind("<Button-1>",   self._on_click)
        self.bind("<B1-Motion>",  self._on_drag)
        self.bind("<Button-3>",   self._on_right)   # right-click = clear
        self.configure(cursor="crosshair")

    def update_frame(self, frame: np.ndarray):
        self._frame_orig = frame.copy()
        self._render(frame)

    def _render(self, frame):
        h, w = frame.shape[:2]
        ww = max(self.winfo_width(), 640)
        wh = max(self.winfo_height(), 360)
        scale = min(ww/w, wh/h, 1.0)
        nw, nh = int(w*scale), int(h*scale)
        disp = cv2.resize(frame, (nw, nh))

        if self._line_frac is not None:
            ly = int(nh * self._line_frac)
            # Draw line with handles
            cv2.line(disp, (0, ly), (nw, ly), (57,197,187), 2)
            # Left handle circle
            cv2.circle(disp, (20, ly), 8, (57,197,187), -1)
            # Right handle circle
            cv2.circle(disp, (nw-20, ly), 8, (57,197,187), -1)
            # Label
            cv2.rectangle(disp, (4, max(ly-22,0)), (260, max(ly-2,20)), (10,30,40), -1)
            cv2.putText(disp, f"Counting line ({int(self._line_frac*100)}%) - drag to move",
                        (8, max(ly-6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (57,197,187), 1)
        else:
            # Show hint overlay
            ov = disp.copy()
            cv2.rectangle(ov, (0, nh//2-22), (nw, nh//2+22), (10,20,30), -1)
            cv2.addWeighted(ov, 0.6, disp, 0.4, 0, disp)
            cv2.putText(disp, "Click anywhere on video to set counting line",
                        (nw//2-220, nh//2+7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (57,197,187), 1)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._img = ctk.CTkImage(light_image=img, dark_image=img, size=(nw,nh))
        self.configure(image=self._img, text="")
        # Store scaled dimensions so _set_line_at can compute correct fraction
        self._nh = nh
        self._nw = nw

    def _on_click(self, e):
        self._set_line_at(e.y)

    def _on_drag(self, e):
        self._set_line_at(e.y)

    def _on_right(self, e):
        """Right-click clears the manual line."""
        self._line_frac = None
        if self._frame_orig is not None:
            self._render(self._frame_orig)
        if self.on_line_set:
            self.on_line_set(None)

    def _set_line_at(self, y_px):
        # Use stored scaled frame height (nh) — NOT winfo_height() (widget height).
        # When the video frame doesn't fill the full widget vertically, dividing by
        # winfo_height causes the drawn line to appear above where the user clicked.
        nh = self._nh if self._nh else self.winfo_height()
        if nh <= 0: return
        # Clamp click to within the frame area, then compute fraction of frame height
        frac = max(0.05, min(0.95, y_px / nh))
        self._line_frac = frac
        if self._frame_orig is not None:
            self._render(self._frame_orig)
        if self.on_line_set:
            self.on_line_set(frac)

    def get_line_frac(self):
        return self._line_frac

    def clear(self):
        self._line_frac = None


class NavBtn(ctk.CTkButton):
    def __init__(self, master, icon, label, cmd, **kw):
        super().__init__(master, text=f"  {icon}   {label}", anchor="w",
                         fg_color="transparent", font=("Segoe UI", 12),
                         height=42, corner_radius=10, command=cmd, **kw)

    def set_active(self, v):
        self.configure(
            fg_color=(["#1a3a6e", "#dbeafe"][ctk.get_appearance_mode() == "Light"] if v
                      else "transparent"),
            text_color=(ACC_BLUE if v else ("#94a3b8", "#475569")),
            font=("Segoe UI", 12, "bold" if v else "normal"),
            border_width=1 if v else 0,
            border_color=ACC_BLUE)   # always a real colour — border_width=0 hides it anyway


class SLabel(ctk.CTkLabel):
    def __init__(self, master, text, **kw):
        super().__init__(master, text=text.upper(),
                         font=("Segoe UI", 9, "bold"),
                         text_color=("#4f8ef7", "#64748b"), **kw)


class StatusBar(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, height=30, corner_radius=0, **kw)
        self.grid_columnconfigure(1, weight=1)
        # Left pill — status dot + message
        pill = ctk.CTkFrame(self, fg_color="transparent")
        pill.grid(row=0, column=0, padx=(10, 0), sticky="w")
        self._dot = ctk.CTkLabel(pill, text="●", font=("Segoe UI", 10), width=18)
        self._dot.pack(side="left")
        self._msg = ctk.CTkLabel(pill, text="Ready", font=("Segoe UI", 11))
        self._msg.pack(side="left", padx=(2, 0))
        # Right — branding
        p = load_prefs()
        name = p.get("author_name", "Nishan")
        inst = p.get("institution", "SUST · CEE")
        self._right = ctk.CTkLabel(self,
            text=f"VELOXIS  ·  {name}, {inst}  ·  NextCity Tessera  ·  © 2026",
            font=("Segoe UI", 10), text_color="#4f8ef7")
        self._right.grid(row=0, column=2, padx=12, sticky="e")

    def set(self, msg, state="idle"):
        colours = {"idle": None, "running": ACC_GREEN,
                   "warn": ACC_AMBER, "error": ACC_RED}
        c = colours.get(state)
        if c:
            self._dot.configure(text_color=c)
        else:
            self._dot.configure(
                text_color=["#1f2937", "#94a3b8"][ctk.get_appearance_mode() == "Light"])
        self._msg.configure(text=msg)


class Page(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0)
        self.grid_columnconfigure(0, weight=1)

    def page_header(self, icon, title, subtitle):
        hf = ctk.CTkFrame(self, fg_color="transparent")
        hf.grid(row=0, column=0, padx=0, pady=0, sticky="ew")
        hf.grid_columnconfigure(1, weight=1)
        # Gradient accent top bar (4px)
        ctk.CTkFrame(hf, height=4, corner_radius=0,
                     fg_color=(ACC_BLUE, "#2563eb")
                    ).grid(row=0, column=0, columnspan=3, sticky="ew")
        inner = ctk.CTkFrame(hf, fg_color="transparent")
        inner.grid(row=1, column=0, columnspan=3, padx=32, pady=(16, 14), sticky="ew")
        inner.grid_columnconfigure(1, weight=1)
        # Icon badge — brighter, slightly larger
        ic = ctk.CTkFrame(inner, width=50, height=50, corner_radius=14,
                          fg_color=(ACC_BLUE, "#1a3a6e"))
        ic.grid(row=0, column=0, rowspan=2, padx=(0, 18))
        ic.grid_propagate(False)
        ctk.CTkLabel(ic, text=icon, font=("Segoe UI", 22)
                    ).place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(inner, text=title, font=("Segoe UI", 19, "bold")
                    ).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(inner, text=subtitle, font=("Segoe UI", 11),
                     text_color="#64748b"
                    ).grid(row=1, column=1, sticky="w")
        # Bottom separator
        ctk.CTkFrame(hf, height=1, corner_radius=0
                    ).grid(row=2, column=0, columnspan=3, sticky="ew")


class DetachedWindow(tk.Toplevel):
    def __init__(self,master):
        super().__init__(master)
        self.title("TrafficCounter BD — Live Feed")
        self.geometry("960x580"); self._img=None; self._frame=None
        theme=ctk.get_appearance_mode()
        bg="#0e1117" if theme=="Dark" else "#f0f4f8"
        self.configure(bg=bg)
        self.protocol("WM_DELETE_WINDOW",self._close); self._closed=False
        bar=tk.Frame(self,bg="#161b27" if theme=="Dark" else "#fff",height=38)
        bar.pack(fill="x")
        tk.Label(bar,text="  📹  Live Camera Feed",bg=bar["bg"],
                 fg="#e8eaf0" if theme=="Dark" else "#0f172a",
                 font=("Segoe UI",11,"bold")).pack(side="left",pady=8)
        self.fps_l=tk.Label(bar,text="FPS: —",bg=bar["bg"],
                             fg=ACC_TEAL,font=("Consolas",10))
        self.fps_l.pack(side="right",padx=12)
        self.cnt_l=tk.Label(bar,text="Total: 0",bg=bar["bg"],
                             fg=ACC_GREEN,font=("Segoe UI",10,"bold"))
        self.cnt_l.pack(side="right",padx=6)
        tk.Button(bar,text="📸 Snapshot",bg="#1e2535" if theme=="Dark" else "#e8edf2",
                  fg=ACC_BLUE,relief="flat",font=("Segoe UI",10),
                  command=self._snap).pack(side="right",padx=6,pady=5)
        self.lbl=tk.Label(self,bg="#0a0d14" if theme=="Dark" else "#f8fafc",
                          text="Waiting…",fg="#64748b",font=("Segoe UI",12))
        self.lbl.pack(fill="both",expand=True)
    def update_frame(self,frame,summary):
        if self._closed: return
        self._frame=frame.copy()
        h,w=frame.shape[:2]
        lw=max(self.lbl.winfo_width(),640); lh=max(self.lbl.winfo_height(),360)
        sc=min(lw/w,lh/h,1.0); nw,nh=int(w*sc),int(h*sc)
        rgb=cv2.cvtColor(cv2.resize(frame,(nw,nh)),cv2.COLOR_BGR2RGB)
        self._img=ImageTk.PhotoImage(Image.fromarray(rgb))
        self.lbl.configure(image=self._img,text="")
        self.cnt_l.configure(text=f"Total: {summary.get('total_unique',0)}")
    def set_fps(self,fps): self.fps_l.configure(text=f"FPS: {fps:.1f}")
    def _snap(self):
        if self._frame is None: return
        os.makedirs("data/snapshots",exist_ok=True)
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"data/snapshots/snap_{ts}.jpg",self._frame)
        self.cnt_l.configure(text="Snapshot saved!")
    def _close(self): self._closed=True; self.destroy()
    @property
    def closed(self): return self._closed


# ================================================================
#  HOME PAGE
# ================================================================
class HomePage(Page):
    def __init__(self,master):
        super().__init__(master)
        self.grid_rowconfigure(4,weight=1)
        self.page_header("🏠","Dashboard","Session overview & live stats")
        # Row 1 — primary counts
        r1=ctk.CTkFrame(self,fg_color="transparent")
        r1.grid(row=1,column=0,padx=32,sticky="ew")
        r1.grid_columnconfigure((0,1,2,3),weight=1)
        self.ct =StatCard(r1,"Total",   "—",ACC_BLUE,  "🚗")
        self.cc =StatCard(r1,"Cars",    "—",ACC_TEAL,  "🚙")
        self.cr =StatCard(r1,"Rickshaws","—",ACC_AMBER,"🛺")
        self.cm =StatCard(r1,"Motorcycles","—",ACC_RED,"🏍")
        for i,c in enumerate([self.ct,self.cc,self.cr,self.cm]):
            c.grid(row=0,column=i,padx=(0 if i==0 else 10,0),sticky="ew")
        # Row 2 — secondary counts
        r2=ctk.CTkFrame(self,fg_color="transparent")
        r2.grid(row=2,column=0,padx=32,pady=(10,0),sticky="ew")
        r2.grid_columnconfigure((0,1,2,3),weight=1)
        self.cbus =StatCard(r2,"Buses",    "—",ACC_BLUE,  "🚌")
        self.ctrk =StatCard(r2,"Trucks",   "—",ACC_PURPLE,"🚛")
        self.cbike=StatCard(r2,"Bicycles", "—",ACC_GREEN, "🚲")
        self.csess=StatCard(r2,"Sessions", "—","#94a3b8",  "📁")
        for i,c in enumerate([self.cbus,self.ctrk,self.cbike,self.csess]):
            c.grid(row=0,column=i,padx=(0 if i==0 else 10,0),sticky="ew")
        # Session log header row with buttons
        log_hdr=ctk.CTkFrame(self,fg_color="transparent")
        log_hdr.grid(row=3,column=0,padx=32,pady=(18,4),sticky="ew")
        log_hdr.grid_columnconfigure(0,weight=1)
        SLabel(log_hdr,"Session Log").grid(row=0,column=0,sticky="w")
        ctk.CTkButton(log_hdr,text="🗺  Map Report",width=130,height=30,
            fg_color=ACC_BLUE,hover_color="#2563eb",
            font=("Segoe UI",11,"bold"),corner_radius=8,
            command=self._map_report_dialog
        ).grid(row=0,column=1,padx=(0,8))
        ctk.CTkButton(log_hdr,text="🗑  Clear Log",width=100,height=30,
            fg_color="transparent",border_width=1,
            font=("Segoe UI",11),corner_radius=8,
            command=self._clear_log
        ).grid(row=0,column=2)
        self.log=ctk.CTkTextbox(self,font=("Consolas",11),corner_radius=12,
                                 border_width=1)
        self.log.grid(row=4,column=0,padx=32,pady=(0,24),sticky="nsew")
        self._load_stats()

    def _get_df(self):
        files=glob.glob(os.path.join("data","log_*.csv"))
        if not files: return None
        dfs=[d for d in [pd.read_csv(f) for f in files] if not d.empty]
        return pd.concat(dfs,ignore_index=True) if dfs else None

    def _load_stats(self):
        df=self._get_df()
        # Always clear first — prevents duplicate entries on refresh
        self.log.delete("1.0","end")
        if df is None:
            self.log.insert("end","No sessions yet. Run detection to start.\n")
            return
        try:
            bt=df["vehicle_type"].value_counts().to_dict() if "vehicle_type" in df.columns else {}
            bt_low = {k.lower(): v for k, v in bt.items()}
            self.ct.set(len(df))
            self.cc.set(bt_low.get("car",0))
            self.cr.set(bt_low.get("rickshaw",0) or bt_low.get("rickshaw/cng",0))
            self.cm.set(bt_low.get("motorcycle",0))
            self.cbus.set(bt_low.get("bus",0))
            self.ctrk.set(bt_low.get("truck",0))
            self.cbike.set(bt_low.get("bicycle",0) or bt_low.get("bike",0))
            nsess = df["session"].nunique() if "session" in df.columns else 0
            self.csess.set(nsess)
            # Load recent session history into log
            self.log.insert("end","── Previous Sessions ──────────────────────\n")
            if "session" in df.columns and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                for sess, grp in df.groupby("session"):
                    t0 = grp["timestamp"].min()
                    date_str = t0.strftime("%Y-%m-%d %H:%M") if pd.notna(t0) else "unknown"
                    vt = grp["vehicle_type"].value_counts().to_dict() if "vehicle_type" in grp.columns else {}
                    top = ", ".join(f"{k}:{v}" for k,v in list(vt.items())[:4])
                    self.log.insert("end",f"[{date_str}]  {sess}  — {len(grp)} vehicles  ({top})\n")
            self.log.see("end")
        except Exception as e:
            self.log.insert("end",f"Could not load history: {e}\n")

    def update_stats(self,s):
        bt=s.get("by_type",{})
        bt_low = {k.lower(): v for k, v in bt.items()}
        self.ct.set(s.get("total_unique",0))
        self.cc.set(bt_low.get("car",0))
        self.cr.set(bt_low.get("rickshaw",0) or bt_low.get("rickshaw/cng",0))
        self.cm.set(bt_low.get("motorcycle",0))
        self.cbus.set(bt_low.get("bus",0))
        self.ctrk.set(bt_low.get("truck",0))
        self.cbike.set(bt_low.get("bicycle",0) or bt_low.get("bike",0))

    def log_msg(self,msg):
        ts=datetime.datetime.now().strftime("%H:%M:%S")
        self.log.insert("end",f"[{ts}]  {msg}\n"); self.log.see("end")

    def _clear_log(self):
        self.log.delete("1.0","end")
        self.log.insert("end","Log cleared.\n")

    def _map_report_dialog(self):
        """Session selector dialog for map report — single, multi, or all sessions."""
        try:
            files = glob.glob(os.path.join("data","log_*.csv"))
            dfs = [d for d in [pd.read_csv(f) for f in files] if not d.empty]
            if not dfs: generate_map_report(); return
            df = pd.concat(dfs, ignore_index=True)
            if "session" not in df.columns: generate_map_report(); return
            sessions = sorted(df["session"].dropna().unique().tolist(), reverse=True)
        except Exception: generate_map_report(); return

        dlg = tk.Toplevel(self)
        dlg.title("Map Report — Select Sessions")
        dlg.geometry("480x520"); dlg.resizable(False, True)
        dlg.configure(bg="#0f172a"); dlg.grab_set()

        tk.Label(dlg,text="Map Report — Select Sessions",bg="#0f172a",fg="#60a5fa",
                 font=("Segoe UI",14,"bold")).pack(padx=24,pady=(18,4),anchor="w")
        tk.Label(dlg,text="Tick sessions to include. Multiple = combined report.",
                 bg="#0f172a",fg="#94a3b8",font=("Segoe UI",11)).pack(padx=24,pady=(0,10),anchor="w")

        frame=tk.Frame(dlg,bg="#1e293b"); frame.pack(fill="both",expand=True,padx=24,pady=(0,8))
        canv=tk.Canvas(frame,bg="#1e293b",highlightthickness=0)
        vsb=tk.Scrollbar(frame,orient="vertical",command=canv.yview)
        canv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right",fill="y"); canv.pack(side="left",fill="both",expand=True)
        inner=tk.Frame(canv,bg="#1e293b"); canv.create_window((0,0),window=inner,anchor="nw")
        inner.bind("<Configure>",lambda e: canv.configure(scrollregion=canv.bbox("all")))

        chk_vars={}
        all_var=tk.BooleanVar(value=False)
        def _toggle_all():
            v=all_var.get()
            for var in chk_vars.values(): var.set(v)
        tk.Checkbutton(inner,text="  ALL sessions combined",variable=all_var,
                       command=_toggle_all,bg="#1e293b",fg=ACC_BLUE,selectcolor="#0f172a",
                       activebackground="#1e293b",activeforeground=ACC_BLUE,
                       font=("Segoe UI",11,"bold"),cursor="hand2").pack(anchor="w",padx=12,pady=(10,6))
        tk.Frame(inner,bg="#334155",height=1).pack(fill="x",padx=12,pady=(0,6))

        for sess in sessions:
            grp=df[df["session"]==sess]; cnt=len(grp)
            if "timestamp" in grp.columns:
                grp_ts=pd.to_datetime(grp["timestamp"],errors="coerce")
                date_str=grp_ts.min().strftime("%Y-%m-%d %H:%M") if not grp_ts.isna().all() else "?"
            else: date_str="?"
            var=tk.BooleanVar(value=False); chk_vars[sess]=var
            rf=tk.Frame(inner,bg="#1e293b"); rf.pack(fill="x",padx=12,pady=2)
            tk.Checkbutton(rf,variable=var,bg="#1e293b",activebackground="#1e293b",
                           selectcolor="#0f172a").pack(side="left")
            tk.Label(rf,text=f"{sess[:32]}",bg="#1e293b",fg="#e2e8f0",
                     font=("Segoe UI",10,"bold")).pack(side="left",padx=(2,6))
            tk.Label(rf,text=f"{date_str}  ·  {cnt} veh",bg="#1e293b",fg="#64748b",
                     font=("Segoe UI",9)).pack(side="left")

        btn_f=tk.Frame(dlg,bg="#0f172a"); btn_f.pack(fill="x",padx=24,pady=(4,18))
        def _generate():
            selected=[s for s,v in chk_vars.items() if v.get()]
            dlg.destroy()
            if not selected: generate_map_report(None)
            elif len(selected)==1: generate_map_report(selected[0])
            else: generate_map_report_multi(selected)
        tk.Button(btn_f,text="Generate Report",bg=ACC_BLUE,fg="white",
                  font=("Segoe UI",12,"bold"),relief="flat",padx=20,pady=8,
                  cursor="hand2",command=_generate).pack(side="left")
        tk.Button(btn_f,text="Cancel",bg="#1e293b",fg="#94a3b8",
                  font=("Segoe UI",11),relief="flat",padx=14,pady=8,
                  cursor="hand2",command=dlg.destroy).pack(side="left",padx=(10,0))



# ================================================================
#  LIVE PAGE
# ================================================================
class LivePage(Page):
    def __init__(self,master,status_bar=None,home_page=None):
        super().__init__(master)
        self.thread=None; self.status_bar=status_bar; self.home_page=home_page
        self._detached=None; self._t0=None; self._last_frame=None
        self._conf_ref=[0.40]
        self.grid_rowconfigure(3,weight=1)
        self.page_header("📹","Live Detection","Real-time · AI line · any camera angle")

        # ── Source card ───────────────────────────────────────
        src=ctk.CTkFrame(self,corner_radius=14,border_width=1)
        src.grid(row=1,column=0,padx=32,pady=(0,10),sticky="ew")
        src.grid_columnconfigure((0,1,2,3),weight=1)
        SLabel(src,"Camera Source").grid(row=0,column=0,columnspan=4,padx=18,pady=(14,10),sticky="w")

        self.src_var=tk.StringVar(value=load_prefs().get("last_source","webcam"))
        opts=[("webcam","💻","Laptop\nWebcam"),("usb","🔌","USB\nCamera"),
              ("droidcam","📱","DroidCam\n(WiFi)"),("custom","🔗","Custom\nURL")]
        self.opt_btns={}
        _inactive_border = "#2d3748"
        for col,(key,icon,label) in enumerate(opts):
            is_active = self.src_var.get()==key
            f=ctk.CTkFrame(src,corner_radius=8,border_width=2,
                            border_color=(ACC_BLUE if is_active else _inactive_border))
            f.grid(row=1,column=col,padx=(10 if col==0 else 4,10 if col==3 else 4),
                   pady=(0,8),sticky="ew")
            ctk.CTkRadioButton(f,text="",variable=self.src_var,value=key,
                               fg_color=ACC_BLUE,width=16,
                               command=self._src_ch).place(relx=0.88,rely=0.1)
            # Compact: icon small, label one line
            ctk.CTkLabel(f,text=icon,font=("Segoe UI",16)).pack(pady=(8,1))
            ctk.CTkLabel(f,text=label.replace("\n"," "),
                         font=("Segoe UI",9),justify="center").pack(pady=(0,8))
            self.opt_btns[key]=f

        # Detail panes
        self.det=ctk.CTkFrame(src,fg_color="transparent")
        self.det.grid(row=2,column=0,columnspan=4,padx=16,pady=(0,8),sticky="ew")

        self.wp=ctk.CTkFrame(self.det,fg_color="transparent")
        ctk.CTkLabel(self.wp,text="Index:",font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.cam_idx=ctk.CTkComboBox(self.wp,values=["0","1","2","3"],width=72)
        self.cam_idx.set("0"); self.cam_idx.pack(side="left",padx=(0,10))
        self.scan_btn=ctk.CTkButton(self.wp,text="🔍  Auto-scan cameras",width=160,height=30,
            fg_color="transparent",border_width=1,font=("Segoe UI",12),command=self._scan)
        self.scan_btn.pack(side="left",padx=(0,8))
        self.scan_lbl=ctk.CTkLabel(self.wp,text="",font=("Segoe UI",11))
        self.scan_lbl.pack(side="left")

        self.dp2=ctk.CTkFrame(self.det,fg_color="transparent")
        ctk.CTkLabel(self.dp2,text="Phone IP:",font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.ip_var=tk.StringVar(value=load_prefs().get("droidcam_ip",""))
        self.ip_e=ctk.CTkEntry(self.dp2,textvariable=self.ip_var,
            placeholder_text="192.168.1.5",width=165)
        self.ip_e.pack(side="left",padx=(0,8))
        ctk.CTkLabel(self.dp2,text="Port:",font=("Segoe UI",12)).pack(side="left",padx=(0,4))
        self.port_var=tk.StringVar(value=load_prefs().get("droidcam_port","4747"))
        ctk.CTkEntry(self.dp2,textvariable=self.port_var,width=62
                    ).pack(side="left",padx=(0,10))
        self.droid_test=ctk.CTkButton(self.dp2,text="🔗  Test",width=90,height=30,
            fg_color="transparent",border_width=1,font=("Segoe UI",12),command=self._test_droid)
        self.droid_test.pack(side="left",padx=(0,8))
        self.droid_lbl=ctk.CTkLabel(self.dp2,text="",font=("Segoe UI",11))
        self.droid_lbl.pack(side="left")

        self.up=ctk.CTkFrame(self.det,fg_color="transparent")
        ctk.CTkLabel(self.up,text="URL:",font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.url_var=tk.StringVar()
        ctk.CTkEntry(self.up,textvariable=self.url_var,
            placeholder_text="rtsp://...  or  http://...",width=420).pack(side="left")

        # Options
        opt=ctk.CTkFrame(src,fg_color="transparent")
        opt.grid(row=3,column=0,columnspan=4,padx=18,pady=(0,14),sticky="ew")
        self.ai_var=tk.BooleanVar(value=True)
        ctk.CTkSwitch(opt,text="",variable=self.ai_var,
                      button_color=ACC_TEAL,progress_color=ACC_TEAL,
                      width=44,height=22).pack(side="left",padx=(0,8))
        ctk.CTkLabel(opt,text="AI auto-detect counting line",
                     font=("Segoe UI",12,"bold"),
                     text_color=(ACC_TEAL,"#0f766e")).pack(side="left",padx=(0,24))
        ctk.CTkLabel(opt,text="Confidence:",font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.conf_sl=ctk.CTkSlider(opt,from_=0.1,to=0.9,width=130,
            button_color=ACC_BLUE,progress_color=ACC_BLUE,
            command=lambda v:[self._conf_ref.__setitem__(0,float(v)),
                               self.conf_lbl.configure(text=f"{int(float(v)*100)}%"),
                               self.conf_hint.configure(text=self._conf_hint(float(v)))])
        self.conf_sl.set(0.40); self.conf_sl.pack(side="left",padx=(0,4))
        self.conf_lbl=ctk.CTkLabel(opt,text="40%",font=("Segoe UI",11,"bold"),
                                    width=36,text_color=(ACC_BLUE,"#1d4ed8"))
        self.conf_lbl.pack(side="left")
        self.conf_hint=ctk.CTkLabel(opt,text="· crowded road",
                                     font=("Segoe UI",9),text_color="#64748b")
        self.conf_hint.pack(side="left",padx=(4,0))
        self._src_ch()

        # ── Buttons ───────────────────────────────────────────
        br=ctk.CTkFrame(self,fg_color="transparent")
        br.grid(row=2,column=0,padx=32,pady=(0,10),sticky="w")
        self.start_btn=ctk.CTkButton(br,text="▶  Start Detection",
            width=180,height=44,font=("Segoe UI",14,"bold"),corner_radius=10,
            command=self._start)
        self.start_btn.pack(side="left",padx=(0,10))
        self.stop_btn=ctk.CTkButton(br,text="■  Stop",
            width=110,height=44,font=("Segoe UI",14,"bold"),corner_radius=10,
            fg_color="#7f1d1d",hover_color="#991b1b",state="disabled",command=self._stop)
        self.stop_btn.pack(side="left",padx=(0,10))
        self.detach_btn=ctk.CTkButton(br,text="⧉  Pop-out",
            width=120,height=44,font=("Segoe UI",13),corner_radius=10,
            fg_color="transparent",border_width=1,command=self._detach)
        self.detach_btn.pack(side="left",padx=(0,10))
        self.snap_btn=ctk.CTkButton(br,text="📸  Snapshot",
            width=130,height=44,font=("Segoe UI",13),corner_radius=10,
            fg_color="transparent",border_width=1,command=self._snap)
        self.snap_btn.pack(side="left",padx=(0,8))
        self.map_rpt_btn=ctk.CTkButton(br,text="🗺  Map Report",
            width=130,height=44,font=("Segoe UI",13),corner_radius=10,
            fg_color=ACC_BLUE,hover_color="#2563eb",
            command=lambda: generate_map_report(
                getattr(self,"_current_session_label",None)))
        self.map_rpt_btn.pack(side="left",padx=(0,14))
        self.timer_lbl=ctk.CTkLabel(br,text="00:00:00",font=("Consolas",13))
        self.timer_lbl.pack(side="left")

        # ── Compact stats strip (one row) ────────────────────
        self.grid_rowconfigure(3,weight=0)
        self.grid_rowconfigure(4,weight=1)

        strip=ctk.CTkFrame(self,fg_color="transparent",height=72)
        strip.grid(row=3,column=0,padx=32,pady=(0,6),sticky="ew")
        strip.grid_propagate(False)

        self.lv_cards={}
        items=[
            ("total",    "Total",   "🚗", ACC_BLUE,   ("#1d4ed8","#3b82f6")),
            ("car",      "Cars",    "🚙", ACC_TEAL,   ("#0f766e","#2dd4bf")),
            ("cng",      "CNG",     "🛺", "#fb923c",  ("#c2410c","#fb923c")),
            ("rickshaw", "Rick.",   "🛺", ACC_AMBER,  ("#b45309","#fbbf24")),
            ("motorcycle","Moto",   "🏍", ACC_RED,    ("#b91c1c","#f87171")),
            ("bus",      "Bus",     "🚌", "#60a5fa",  ("#1d4ed8","#60a5fa")),
            ("truck",    "Truck",   "🚛", ACC_PURPLE, ("#6d28d9","#a78bfa")),
            ("bicycle",  "Bike",   "🚲", ACC_GREEN,  ("#047857","#34d399")),
            ("_live",    "Live",   "📍", "#60a5fa",  ("#1d4ed8","#60a5fa")),
            ("_occ",     "Occ%",   "📊", "#f97316",  ("#c2410c","#f97316")),
            ("_queue",   "Queue",  "🚦", "#818cf8",  ("#4c1d95","#818cf8")),
            ("_rate",    "Rate/hr","📈", "#fbbf24",  ("#b45309","#fbbf24")),
            ("_v85",     "V85",    "🚀", "#34d399",  ("#047857","#34d399")),
            ("_limit",   "Limit",  "🛑", "#f87171",  ("#b91c1c","#f87171")),
            ("_los",     "LOS",    "🏁", "#a78bfa",  ("#6d28d9","#a78bfa")),
            ("_person",  "People", "🚶", "#a78bfa",  ("#6d28d9","#a78bfa")),
            ("_safety",  "Safety", "⚠️", "#f87171",  ("#b91c1c","#f87171")),
        ]
        n_cols = len(items)
        strip.grid_columnconfigure(list(range(n_cols)), weight=1)
        strip.grid_rowconfigure(0, weight=1)

        for col,(key,lbl,icon,acc,theme_color) in enumerate(items):
            f=ctk.CTkFrame(strip,corner_radius=8,border_width=1)
            f.grid(row=0,column=col,padx=(0 if col==0 else 2,0),sticky="nsew")
            f.grid_propagate(False)
            f.grid_rowconfigure(1,weight=1)
            f.grid_rowconfigure(2,weight=1)
            f.grid_columnconfigure(0,weight=1)
            # Top accent bar
            ctk.CTkFrame(f,fg_color=acc,height=3,corner_radius=0
                        ).grid(row=0,column=0,sticky="ew")
            # Label — small, always readable in both themes
            ctk.CTkLabel(f,
                         text=f"{icon} {lbl}",
                         font=("Segoe UI",7),
                         text_color=("#475569","#94a3b8"),
                        ).grid(row=1,column=0,padx=2,pady=(3,0),sticky="sew")
            # Value — theme-aware accent (dark=bright, light=dark shade)
            val_lbl=ctk.CTkLabel(f,
                                  text="0",
                                  font=("Segoe UI",11,"bold"),
                                  text_color=theme_color)
            val_lbl.grid(row=2,column=0,padx=2,pady=(0,4),sticky="new")
            self.lv_cards[key]=val_lbl

        self._live_manual_line = None
        self.video=ClickableVideoCanvas(self,
            on_line_set=self._live_line_set,
            placeholder="Camera feed will appear here · Click to set counting line")
        self.video.grid(row=4,column=0,padx=32,pady=(0,24),sticky="nsew")

    def _src_ch(self):
        v=self.src_var.get()
        _inactive = "#2d3748"
        for k,f in self.opt_btns.items():
            f.configure(border_color=(ACC_BLUE if k==v else _inactive))
        for p in [self.wp,self.dp2,self.up]: p.pack_forget()
        if v in ("webcam","usb"):
            self.wp.pack(fill="x")
            self.cam_idx.set("1" if v=="usb" else "0")
        elif v=="droidcam": self.dp2.pack(fill="x")
        elif v=="custom": self.up.pack(fill="x")

    def _scan(self):
        self.scan_lbl.configure(text="Scanning…",text_color=ACC_AMBER); self.update()
        found=[]
        for i in range(4):
            cap=cv2.VideoCapture(i,cv2.CAP_DSHOW)
            if cap.isOpened():
                ret,_=cap.read()
                if ret: found.append(str(i))
                cap.release()
        if found:
            self.scan_lbl.configure(text=f"✓ Found: {', '.join(found)}",text_color=ACC_GREEN)
            self.cam_idx.set(found[0])
        else: self.scan_lbl.configure(text="✗ None found",text_color=ACC_RED)

    def _test_droid(self):
        ip=self.ip_var.get().strip() or "192.168.1.5"
        port=self.port_var.get().strip() or "4747"
        self.droid_lbl.configure(text="Testing…",text_color=ACC_AMBER); self.update()
        for suf in ["/video","/mjpegfeed","/videofeed"]:
            url=f"http://{ip}:{port}{suf}"
            cap=cv2.VideoCapture(url)
            if cap.isOpened():
                ret,_=cap.read(); cap.release()
                if ret:
                    self.droid_lbl.configure(text="✓ Connected!",text_color=ACC_GREEN)
                    self._droid_url=url
                    save_prefs({"droidcam_ip":ip,"droidcam_port":port}); return
        self.droid_lbl.configure(text="✗ Cannot connect — check IP & WiFi",text_color=ACC_RED)

    def _get_src(self):
        v=self.src_var.get()
        if v in ("webcam","usb"): return int(self.cam_idx.get() or "0")
        if v=="droidcam":
            if hasattr(self,"_droid_url"): return self._droid_url
            ip=self.ip_var.get().strip() or "192.168.1.5"
            port=self.port_var.get().strip() or "4747"
            return f"http://{ip}:{port}/video"
        return self.url_var.get().strip()

    def _start(self):
        src=self._get_src()
        if not src and src!=0: return
        save_prefs({"last_source":self.src_var.get()})
        self._t0=time.time(); self._conf_ref[0]=self.conf_sl.get()
        self._current_session_label = None   # will be set when thread fires on_done
        if self.status_bar: self.status_bar.set("Detection running…","running")
        # Use manual line if user clicked on video
        manual = getattr(self,'_live_manual_line',None)
        self.thread=DetectionThread(src,"live",
            on_status=lambda m: self.after(0,lambda mm=m: self._sts(mm)),
            on_done  =lambda s: self.after(0,lambda ss=s: self._done(ss)),
            use_ai=self.ai_var.get() and manual is None,
            conf_ref=self._conf_ref,
            manual_line=manual)
        self.thread.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        # Fetch road speed limit in background (non-blocking)
        self._road_speed_limit = "—"
        threading.Thread(target=self._fetch_speed_limit, daemon=True).start()
        self._poll(); self._tick()

    def _poll(self):
        # Process frames from queue
        try:
            frame,summary=self.thread.frame_q.get_nowait()
            self._last_frame=frame; self.video.update_frame(frame)
            if self._detached and not self._detached.closed:
                self._detached.update_frame(frame,summary)
                self._detached.set_fps(self.thread.fps)
            if summary:
                bt=summary.get("by_type",{})
                # Normalise keys to lowercase for robust matching
                # Model may output "Car", "CNG", "Rickshaw" etc with different casing
                bt_low = {k.lower(): v for k, v in bt.items()}
                total=summary.get("total_unique",0)
                self.lv_cards["total"].configure(text=str(total))
                # Aliases: card_key -> list of possible model class names (all lowercase)
                _card_aliases = {
                    "car":        ["car"],
                    "cng":        ["cng", "cng/auto", "auto", "auto_rickshaw", "autorickshaw"],
                    "rickshaw":   ["rickshaw", "rickshaw/cng", "rick"],
                    "motorcycle": ["motorcycle", "moto", "bike"],
                    "bus":        ["bus"],
                    "truck":      ["truck"],
                    "bicycle":    ["bicycle", "bike", "cycle"],
                }
                for k, aliases in _card_aliases.items():
                    val = sum(bt_low.get(a, 0) for a in aliases)
                    self.lv_cards[k].configure(text=str(val))
                self.lv_cards["_live"].configure(text=str(summary.get("live_vehicles",0)))
                self.lv_cards["_occ"].configure(text=f"{summary.get('occupancy_pct',0):.0f}%")
                self.lv_cards["_queue"].configure(text=str(summary.get("queue_length",0)))
                self.lv_cards["_rate"].configure(text=str(summary.get("current_rate",0)))
                v85 = summary.get("speed_85th", 0)
                self.lv_cards["_v85"].configure(
                    text=f"{v85:.0f}" if v85 else "—")
                # Speed limit from cached Overpass fetch
                lim = getattr(self, "_road_speed_limit", "—")
                self.lv_cards["_limit"].configure(text=str(lim))
                los = summary.get("los_letter", "—")
                self.lv_cards["_los"].configure(text=str(los))
                self.lv_cards["_person"].configure(text=str(summary.get("person_count",0)))
                self.lv_cards["_safety"].configure(text=str(summary.get("safety_events",0)))
                if total and self.home_page:
                    self.home_page.update_stats(summary)
        except queue.Empty: pass
        # Keep polling as long as thread is alive OR queue still has frames
        if self.thread and (self.thread.is_alive() or not self.thread.frame_q.empty()):
            self.after(33,self._poll)

    def _tick(self):
        if self.thread and self.thread.is_alive() and self._t0:
            e=int(time.time()-self._t0)
            self.timer_lbl.configure(text=f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}")
            self.after(1000,self._tick)

    def _sts(self,m):
        if self.status_bar:
            self.status_bar.set(m,"error" if "ERROR" in m or "Cannot" in m else "running")

    def _done(self,s):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        total = s.get("total_unique", 0)
        phf   = s.get("phf", 0)
        peak  = s.get("peak_rate", 0)
        v85   = s.get("speed_85th", 0)
        sat   = s.get("saturation", 0)
        los   = s.get("los_letter", "—")
        delay = s.get("avg_delay_sec", 0)
        # Store session label so map report button filters to this session only
        self._current_session_label = s.get("session_label")

        # Speed comparison summary
        lim_raw = getattr(self, "_road_speed_limit", "—")
        speed_note = ""
        try:
            lim_val = int(str(lim_raw).replace("*",""))
            if v85 and lim_val:
                diff = v85 - lim_val
                if diff > 10:
                    speed_note = f"  ⚠ V85 exceeds limit by {diff:.0f}km/h"
                elif diff > 0:
                    speed_note = f"  ↑ V85 slightly above limit"
                else:
                    speed_note = f"  ✓ V85 within speed limit"
        except: pass

        if self.status_bar:
            self.status_bar.set(
                f"Session complete ✓  —  {total} vehicles  |  "
                f"PHF:{phf:.2f}  Peak:{peak}v/hr  LOS:{los}({delay:.0f}s)  "
                f"V85:{v85}km/h / Limit:{lim_raw}km/h{speed_note}",
                "idle")
        if self.home_page:
            self.home_page.update_stats(s)
            msg = (f"Live session — {total} vehicles  |  "
                   f"PHF:{phf:.2f}  Peak:{peak}v/hr  LOS:{los}  "
                   f"V85:{v85}km/h  Limit:{lim_raw}km/h{speed_note}  "
                   f"SatFlow:{sat}v/hr")
            self.home_page.log_msg(msg)

    def _stop(self):
        if self.thread:
            self.thread.stop()
            # _done callback fires from thread.run() after cap.release()
            # Just update UI state here — don't reset counters
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if self.status_bar: self.status_bar.set("Stopping — saving session…","warn")

    def _detach(self):
        if self._detached and not self._detached.closed:
            self._detached.destroy(); self._detached=None
            self.detach_btn.configure(text="⧉  Pop-out")
        else:
            self._detached=DetachedWindow(self)
            self.detach_btn.configure(text="✕  Close pop-out")

    @staticmethod
    def _conf_hint(v):
        if v < 0.30: return "night/dark"
        if v < 0.40: return "crowded road"
        if v < 0.55: return "daylight clear"
        if v < 0.70: return "strict"
        return "very strict"

    def _fetch_speed_limit(self):
        """
        Fetch road speed limit from OpenStreetMap Overpass API.
        Uses study location from Settings. Runs in background thread.
        No API key needed — completely free.
        """
        try:
            p = load_prefs()
            lat = float(p.get("loc_lat") or 0)
            lng = float(p.get("loc_lng") or 0)
            if lat == 0 or lng == 0:
                self.after(0, lambda: self.lv_cards["_limit"].configure(text="Set loc"))
                return

            # Overpass API query — finds roads within 50m of study point
            # Gets maxspeed tag from nearest highway
            import urllib.request, json as _json
            delta = 0.0005   # ~50m radius
            query = (
                f"[out:json][timeout:8];"
                f"way[highway][maxspeed]"
                f"({lat-delta},{lng-delta},{lat+delta},{lng+delta});"
                f"out tags 1;"
            )
            url = "https://overpass-api.de/api/interpreter"
            req = urllib.request.Request(
                url,
                data=query.encode(),
                headers={"Content-Type": "text/plain",
                         "User-Agent": "VELOXIS/2.0 traffic-research"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = _json.loads(r.read())

            elements = data.get("elements", [])
            if elements:
                raw = elements[0].get("tags", {}).get("maxspeed", "")
                # Parse: "50", "50 mph", "50 km/h" → integer km/h
                num = "".join(c for c in raw if c.isdigit())
                if num:
                    limit = int(num)
                    # Convert mph → km/h if needed
                    if "mph" in raw.lower():
                        limit = round(limit * 1.609)
                    self.after(0, lambda lim=limit: [
                        setattr(self, "_road_speed_limit", f"{lim}"),
                        self.lv_cards["_limit"].configure(text=f"{lim}"),
                        self.status_bar and self.status_bar.set(
                            f"Road speed limit: {lim} km/h (OSM)", "idle")
                    ])
                    return

            # No maxspeed tag found — try wider radius (150m)
            delta2 = 0.0015
            query2 = (
                f"[out:json][timeout:8];"
                f"way[highway]"
                f"({lat-delta2},{lng-delta2},{lat+delta2},{lng+delta2});"
                f"out tags 3;"
            )
            req2 = urllib.request.Request(
                url, data=query2.encode(),
                headers={"Content-Type": "text/plain",
                         "User-Agent": "VELOXIS/2.0 traffic-research"})
            with urllib.request.urlopen(req2, timeout=10) as r2:
                data2 = _json.loads(r2.read())

            for el in data2.get("elements", []):
                raw = el.get("tags", {}).get("maxspeed", "")
                num = "".join(c for c in raw if c.isdigit())
                if num:
                    limit = int(num)
                    if "mph" in raw.lower(): limit = round(limit * 1.609)
                    self.after(0, lambda lim=limit: [
                        setattr(self, "_road_speed_limit", f"{lim}"),
                        self.lv_cards["_limit"].configure(text=f"{lim}")])
                    return

            # No speed limit in OSM — show default fallback
            self.after(0, lambda: [
                setattr(self, "_road_speed_limit", "50*"),
                self.lv_cards["_limit"].configure(text="50*")])

        except Exception as e:
            self.after(0, lambda: [
                setattr(self, "_road_speed_limit", "—"),
                self.lv_cards["_limit"].configure(text="—")])

    def _live_line_set(self, frac):
        self._live_manual_line = frac
        if frac is not None:
            self.ai_var.set(False)
            if self.status_bar:
                self.status_bar.set(
                    f"Line set at {int(frac*100)}% — right-click to clear — click Start","idle")
        else:
            if self.status_bar:
                self.status_bar.set("Line cleared","idle")

    def _snap(self):
        if self._last_frame is None: return
        os.makedirs("data/snapshots",exist_ok=True)
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p=f"data/snapshots/snap_{ts}.jpg"
        cv2.imwrite(p,self._last_frame)
        if self.status_bar: self.status_bar.set(f"Snapshot saved: {p}","idle")
# ================================================================
class FilePage(Page):
    def __init__(self,master,status_bar=None,home_page=None):
        super().__init__(master); self.thread=None; self._last_frame=None
        self.status_bar=status_bar; self.home_page=home_page
        self._cap_preview=None; self._total_frames=1
        self._road_speed_limit = "—"   # init here — avoids getattr(None) race in _done
        self.grid_rowconfigure(4,weight=1)
        self.page_header("🎬","File Detection","Analyse a recorded road video")

        pick=ctk.CTkFrame(self,corner_radius=14,border_width=1)
        pick.grid(row=1,column=0,padx=32,pady=(0,10),sticky="ew")
        pick.grid_columnconfigure(0,weight=1)
        SLabel(pick,"Video File").grid(row=0,column=0,columnspan=3,padx=18,pady=(14,8),sticky="w")
        pr=ctk.CTkFrame(pick,fg_color="transparent")
        pr.grid(row=1,column=0,columnspan=3,padx=16,pady=(0,8),sticky="ew")
        pr.grid_columnconfigure(0,weight=1)
        self.pv=tk.StringVar()
        ctk.CTkEntry(pr,textvariable=self.pv,placeholder_text="No video selected…"
                    ).grid(row=0,column=0,sticky="ew",padx=(0,10))
        ctk.CTkButton(pr,text="📂  Browse",width=120,height=36,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            command=self._browse).grid(row=0,column=1)

        # ── Video seek slider ──────────────────────────────────
        seek_row=ctk.CTkFrame(pick,fg_color="transparent")
        seek_row.grid(row=2,column=0,columnspan=3,padx=16,pady=(0,8),sticky="ew")
        seek_row.grid_columnconfigure(1,weight=1)
        ctk.CTkLabel(seek_row,text="Preview frame:",font=("Segoe UI",11)).grid(row=0,column=0,padx=(0,8))
        self.seek_var=tk.IntVar(value=0)
        self.seek_slider=ctk.CTkSlider(seek_row,from_=0,to=100,
            variable=self.seek_var,command=self._seek_preview,
            button_color=ACC_TEAL,progress_color=ACC_TEAL,state="disabled")
        self.seek_slider.grid(row=0,column=1,sticky="ew",padx=(0,8))
        self.seek_lbl=ctk.CTkLabel(seek_row,text="0 / 0",font=("Segoe UI",11),width=80)
        self.seek_lbl.grid(row=0,column=2)

        ai_r=ctk.CTkFrame(pick,fg_color="transparent")
        ai_r.grid(row=3,column=0,columnspan=3,padx=16,pady=(0,14),sticky="w")
        self.ai_var=tk.BooleanVar(value=False)   # default OFF — user clicks on video
        ctk.CTkSwitch(ai_r,text="",variable=self.ai_var,
                      button_color=ACC_TEAL,progress_color=ACC_TEAL,
                      width=44,height=22).pack(side="left",padx=(0,8))
        ctk.CTkLabel(ai_r,text="AI auto-detect line",
                     font=("Segoe UI",12,"bold"),
                     text_color=(ACC_TEAL,"#0f766e")).pack(side="left",padx=(0,12))
        ctk.CTkLabel(ai_r,
                     text="← OFF: Click on the video frame below to draw your counting line  |  Right-click to clear",
                     font=("Segoe UI",11),text_color="#64748b").pack(side="left")

        br=ctk.CTkFrame(self,fg_color="transparent")
        br.grid(row=2,column=0,padx=32,pady=(0,8),sticky="w")
        self.run_btn=ctk.CTkButton(br,text="▶  Analyse Video",
            width=180,height=44,font=("Segoe UI",14,"bold"),corner_radius=10,
            state="disabled",command=self._run)
        self.run_btn.pack(side="left",padx=(0,10))
        self.stop_btn=ctk.CTkButton(br,text="■  Stop",
            width=110,height=44,font=("Segoe UI",14,"bold"),corner_radius=10,
            fg_color="#7f1d1d",hover_color="#991b1b",state="disabled",command=self._stop)
        self.stop_btn.pack(side="left",padx=(0,10))
        ctk.CTkButton(br,text="📸  Snapshot",width=130,height=44,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            command=self._snap).pack(side="left",padx=(0,10))
        ctk.CTkButton(br,text="📊  Export CSV",width=130,height=44,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            command=self._export).pack(side="left",padx=(0,6))
        ctk.CTkButton(br,text="🏙  Vissim Export",width=140,height=44,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            text_color=(ACC_TEAL,"#0f766e"),command=self._export_vissim).pack(side="left",padx=(0,6))
        self.map_btn=ctk.CTkButton(br,text="🗺  Map Report",width=130,height=44,
            fg_color=ACC_BLUE,hover_color="#2563eb",font=("Segoe UI",13,"bold"),
            state="disabled",command=self._map_report)
        self.map_btn.pack(side="left",padx=(0,6))
        self.tmc_btn=ctk.CTkButton(br,text="📋  TMC Export",width=130,height=44,
            fg_color="transparent",border_width=1,
            text_color=(ACC_PURPLE,"#6d28d9"),
            font=("Segoe UI",13),state="disabled",command=self._export_tmc)
        self.tmc_btn.pack(side="left",padx=(0,10))
        self.prog_lbl=ctk.CTkLabel(br,text="",font=("Segoe UI",12))
        self.prog_lbl.pack(side="left")

        self.prog=ctk.CTkProgressBar(self,height=6,corner_radius=3,
                                      progress_color=ACC_BLUE)
        self.prog.set(0); self.prog.grid(row=3,column=0,padx=32,pady=(0,6),sticky="ew")

        mid=ctk.CTkFrame(self,fg_color="transparent")
        mid.grid(row=4,column=0,padx=32,pady=(0,24),sticky="nsew")
        mid.grid_columnconfigure(0,weight=3); mid.grid_columnconfigure(1,weight=1)
        mid.grid_rowconfigure(0,weight=1)
        self.video=ClickableVideoCanvas(mid, on_line_set=self._on_line_set)
        self.video.grid(row=0,column=0,padx=(0,12),sticky="nsew")
        self._manual_line_frac = None

        res=ctk.CTkScrollableFrame(mid,corner_radius=14,border_width=1,width=180)
        res.grid(row=0,column=1,sticky="nsew")
        SLabel(res,"Results").pack(anchor="w",padx=14,pady=(14,6))
        self.rcards={}
        for key,label,icon,acc in [
            ("total","Total","🚗",ACC_BLUE),("car","Cars","🚙",ACC_TEAL),
            ("rickshaw","Rickshaws","🛺",ACC_AMBER),
            ("CNG/auto","CNGs","🛺","#fb923c"),
            ("motorcycle","Motorcycles","🏍",ACC_RED),
            ("bus","Buses","🚌","#60a5fa"),("truck","Trucks","🚛",ACC_PURPLE),
            ("bicycle","Bicycles","🚲",ACC_GREEN)]:
            c=StatCard(res,label,"—",acc,icon)
            c.pack(fill="x",padx=10,pady=4)
            self.rcards[key]=c
        ctk.CTkButton(res,text="📊 Export",height=36,
            fg_color="transparent",border_width=1,font=("Segoe UI",12),
            command=self._export).pack(fill="x",padx=10,pady=(4,14))

    def _seek_preview(self, val):
        """Show frame at slider position without running detection."""
        if self._cap_preview is None: return
        idx = int(float(val))
        self._cap_preview.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap_preview.read()
        if ret:
            self.video.update_frame(frame)
            self.seek_lbl.configure(text=f"{idx} / {self._total_frames}")

    def _browse(self):
        from tkinter import filedialog
        p=filedialog.askopenfilename(title="Open Video",initialdir="videos",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv *.wmv"),("All","*.*")])
        if not p: return
        self.pv.set(p)
        self.run_btn.configure(state="normal")
        # Load video for seek preview
        if self._cap_preview: self._cap_preview.release()
        self._cap_preview = cv2.VideoCapture(p)
        self._total_frames = max(int(self._cap_preview.get(cv2.CAP_PROP_FRAME_COUNT))-1, 1)
        self.seek_slider.configure(to=self._total_frames, state="normal")
        self.seek_var.set(0)
        # Show first frame as preview
        self._cap_preview.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap_preview.read()
        if ret:
            self.video.update_frame(frame)
            self.seek_lbl.configure(text=f"0 / {self._total_frames}")

    def _on_line_set(self, frac):
        self._manual_line_frac = frac
        if frac is not None:
            # Auto-disable AI when user manually sets line
            self.ai_var.set(False)
            if self.status_bar:
                self.status_bar.set(
                    f"Manual line set at {int(frac*100)}% — right-click to clear — press Analyse Video",
                    "idle")
        else:
            # Line cleared — re-enable AI option
            if self.status_bar:
                self.status_bar.set("Line cleared — AI will auto-detect, or click again to set manually", "idle")

    def _run(self):
        p=self.pv.get().strip()
        if not p: return
        self.prog.set(0)
        if self.status_bar: self.status_bar.set("Analysing…","running")
        manual_line = getattr(self, '_manual_line_frac', None)
        self.thread=DetectionThread(p,"file",
            on_status=lambda m: self.after(0,lambda mm=m:
                [self.prog_lbl.configure(text=mm),
                 self.status_bar and self.status_bar.set(mm,"running")]),
            on_done  =lambda s: self.after(0,lambda ss=s: self._done(ss)),
            on_progress=lambda v: self.after(0,lambda vv=v:
                [self.prog.set(vv/100),self.prog_lbl.configure(text=f"{vv}%")]),
            use_ai=self.ai_var.get() and manual_line is None,
            manual_line=manual_line)
        self.thread.start()
        self.run_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")
        self._poll()

    def _poll(self):
        try:
            frame, summary = self.thread.frame_q.get_nowait()
            self._last_frame = frame
            self.video.update_frame(frame)
            # Update right-panel cards live during analysis
            if summary:
                bt = summary.get("by_type", {})
                self.rcards["total"].set(summary.get("total_unique", 0))
                for k in ["car","rickshaw","CNG/auto","motorcycle","bus","truck","bicycle"]:
                    v = bt.get(k, 0)
                    if v: self.rcards[k].set(v)
        except queue.Empty:
            pass
        # Keep polling while thread alive OR queue still has frames to drain
        if self.thread and (self.thread.is_alive() or not self.thread.frame_q.empty()):
            self.after(33, self._poll)

    def _stop(self):
        if self.thread: self.thread.stop()
        self.run_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")

    def _done(self,s):
        self.run_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
        self.prog.set(1)
        self.map_btn.configure(state="normal")
        self.tmc_btn.configure(state="normal" if s.get("turning_counts") else "disabled")
        bt=s.get("by_type",{})
        bt_low = {k.lower(): v for k, v in bt.items()}
        self.rcards["total"].set(s.get("total_unique",0))
        _rcard_aliases = {
            "car":       ["car"],
            "rickshaw":  ["rickshaw","rickshaw/cng","rick"],
            "CNG/auto":  ["cng","cng/auto","auto","auto_rickshaw"],
            "motorcycle":["motorcycle","moto","bike"],
            "bus":       ["bus"],
            "truck":     ["truck"],
            "bicycle":   ["bicycle","bike","cycle"],
        }
        for k, aliases in _rcard_aliases.items():
            val = sum(bt_low.get(a, 0) for a in aliases)
            if k in self.rcards: self.rcards[k].set(val)
        # Store session label for map report
        self._current_session_label = s.get("session_label")
        self._last_session = s
        v85  = s.get("speed_85th", 0)
        phf  = s.get("phf", 0)
        los  = s.get("los_letter", "—")
        hdwy = s.get("avg_headway_sec", 0)
        sat  = s.get("saturation_flow", 0)
        # Always fetch fresh speed limit in bg — _road_speed_limit initialised to "—" in __init__
        threading.Thread(target=self._fetch_and_compare, args=(v85,), daemon=True).start()
        if self.home_page:
            self.home_page.update_stats(s)
            self.home_page.log_msg(
                f"File — {s.get('total_unique',0)} veh  "
                f"PHF:{phf:.2f}  LOS:{los}  V85:{v85}km/h  "
                f"Hdwy:{hdwy:.1f}s  SatFlow:{sat}v/hr")

    def _map_report(self):
        path = generate_map_report(getattr(self,"_current_session_label",None))
        if path and self.status_bar:
            self.status_bar.set(f"Map report saved: {path}","idle")

    def _export_tmc(self):
        from tkinter import filedialog, messagebox
        path=filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")],initialfile="tmc_matrix.csv")
        if not path: return
        try:
            files=glob.glob(os.path.join("data","*_tmc.csv"))
            if not files:
                messagebox.showinfo("No TMC data",
                    "No TMC file found.\nEnable Zones in Lane Drawing and run detection.")
                return
            import shutil
            shutil.copy(sorted(files)[-1], path)
            # Also copy detail file if present
            detail=sorted(files)[-1].replace("_tmc.csv","_tmc_detail.csv")
            if os.path.exists(detail):
                shutil.copy(detail, path.replace(".csv","_detail.csv"))
            messagebox.showinfo("TMC Exported ✓",
                f"TMC matrix saved to:\n{path}\n\n"
                "Import into Synchro:\n"
                "  Volume → Import → CSV\n\n"
                "Use for HCM signal timing worksheets.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _fetch_and_compare(self, v85):
        """Background fetch of speed limit then show comparison (FilePage). Thread-safe."""
        lim_result = "—"
        try:
            import urllib.request, json as _json
            p = load_prefs()
            lat = float(p.get("loc_lat") or 0)
            lng = float(p.get("loc_lng") or 0)
            if lat != 0 and lng != 0:
                delta = 0.0005
                query = (f"[out:json][timeout:8];"
                         f"way[highway][maxspeed]"
                         f"({lat-delta},{lng-delta},{lat+delta},{lng+delta});"
                         f"out tags 1;")
                req = urllib.request.Request(
                    "https://overpass-api.de/api/interpreter",
                    data=query.encode(),
                    headers={"Content-Type": "text/plain",
                             "User-Agent": "VELOXIS/2.0 traffic-research"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = _json.loads(r.read())
                elements = data.get("elements", [])
                if elements:
                    raw = elements[0].get("tags", {}).get("maxspeed", "")
                    num = "".join(c for c in raw if c.isdigit())
                    if num:
                        lim = int(num)
                        if "mph" in raw.lower(): lim = round(lim * 1.609)
                        lim_result = str(lim)
                    else:
                        lim_result = "50*"
                else:
                    lim_result = "50*"
        except Exception:
            lim_result = "—"
        # All UI writes happen on the main thread via self.after()
        self.after(0, lambda r=lim_result: [
            setattr(self, "_road_speed_limit", r),
            self._show_speed_comparison(v85, r)
        ])

    def _show_speed_comparison(self, v85, lim_raw):
        """Show V85 vs speed limit in status bar."""
        note = ""
        try:
            lim_val = int(str(lim_raw).replace("*",""))
            if v85 and lim_val:
                diff = v85 - lim_val
                if diff > 10:   note = f"  ⚠ V85 exceeds limit by {diff:.0f}km/h"
                elif diff > 0:  note = f"  ↑ V85 slightly above limit"
                else:           note = f"  ✓ V85 within speed limit"
        except: pass
        if self.status_bar:
            self.status_bar.set(
                f"Analysis complete ✓  —  V85:{v85}km/h / Limit:{lim_raw}km/h{note}", "idle")

    def _snap(self):
        if self._last_frame is None: return
        os.makedirs("data/snapshots",exist_ok=True)
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"data/snapshots/snap_{ts}.jpg",self._last_frame)
        if self.status_bar: self.status_bar.set("Snapshot saved","idle")

    def _export(self):
        from tkinter import filedialog, messagebox
        path=filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV","*.csv")],initialfile="traffic_report.csv")
        if not path: return
        try:
            files=glob.glob(os.path.join("data","log_*.csv"))
            dfs=[d for d in [pd.read_csv(f) for f in files] if not d.empty]
            if not dfs: messagebox.showinfo("No data","Run detection first."); return
            pd.concat(dfs,ignore_index=True).to_csv(path,index=False)
            messagebox.showinfo("Exported ✓",f"Saved to:\n{path}")
        except Exception as e: messagebox.showerror("Error",str(e))

    def _export_vissim(self):
        """Export session data in PTV Vissim / Aimsun compatible format."""
        from tkinter import filedialog, messagebox
        try:
            files=glob.glob(os.path.join("data","log_*.csv"))
            dfs=[d for d in [pd.read_csv(f) for f in files] if not d.empty]
            if not dfs: messagebox.showinfo("No data","Run detection first."); return
            df=pd.concat(dfs,ignore_index=True)
            if "timestamp" in df.columns:
                df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
            duration_hrs=max(
                (df["timestamp"].max()-df["timestamp"].min()).total_seconds()/3600
                if "timestamp" in df.columns else 1, 0.0167)
            rows=[]
            for vtype in df["vehicle_type"].dropna().unique():
                sub=df[df["vehicle_type"]==vtype]
                count=len(sub)
                vol_hr=round(count/duration_hrs,1)
                fwd=len(sub[sub["direction"].str.contains("FWD|Forward|→",na=False,regex=True)]) \
                    if "direction" in sub.columns else 0
                bwd=count-fwd
                rows.append({"VehicleType":vtype,"TotalCount":count,
                             "Volume_veh_per_hour":vol_hr,
                             "Forward_count":fwd,"Backward_count":bwd,
                             "Duration_hours":round(duration_hrs,3)})
            out=pd.DataFrame(rows)
            path=filedialog.asksaveasfilename(defaultextension=".csv",
                filetypes=[("CSV","*.csv")],initialfile="vissim_input.csv")
            if not path: return
            out.to_csv(path,index=False)
            messagebox.showinfo("Vissim Export ✓",
                f"Saved to:\n{path}\n\n"
                "Import into PTV Vissim:\n"
                "  Traffic Demand → Vehicle Inputs → Import CSV\n\n"
                "Import into Aimsun:\n"
                "  Traffic State → Origin/Destination → Import")
        except Exception as e: messagebox.showerror("Error",str(e))


# ================================================================
#  CALIBRATE SPEED PAGE  (Homography calibration)
# ================================================================
class CalibratePage(Page):
    """
    4-point homography calibration for accurate speed estimation.
    User loads a video frame, clicks 4 road points, enters real distances.
    Saves homography.npy for detector to use automatically.
    """
    def __init__(self,master,status_bar=None):
        super().__init__(master)
        self.status_bar=status_bar
        self._cap=None; self._frame=None; self._pts=[]; self._photo=None
        self.grid_rowconfigure(3,weight=1)
        self.page_header("📐","Speed Calibration",
            "Click 4 road points → enter real distances → accurate speed for any camera angle")

        # Instructions
        inst=ctk.CTkFrame(self,corner_radius=12,border_width=1)
        inst.grid(row=1,column=0,padx=32,pady=(0,8),sticky="ew")
        ctk.CTkLabel(inst,justify="left",font=("Segoe UI",12),
            text=(
                "HOW TO CALIBRATE:\n"
                "1. Load a video frame where road markings are visible\n"
                "2. Click 4 corners of a known rectangle on the road "
                "(e.g. lane markings, road edge)\n"
                "3. Enter the real-world width and length of that rectangle in metres\n"
                "4. Click Calibrate — homography.npy saved, speed becomes accurate"
            )).pack(padx=18,pady=12,anchor="w")

        # Controls
        cr=ctk.CTkFrame(self,fg_color="transparent")
        cr.grid(row=2,column=0,padx=32,pady=(0,8),sticky="ew")
        ctk.CTkButton(cr,text="📂  Load Video Frame",width=170,height=38,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            command=self._load).pack(side="left",padx=(0,10))
        ctk.CTkLabel(cr,text="Real width (m):",font=("Segoe UI",12)).pack(side="left",padx=(10,4))
        self.w_var=tk.StringVar(value="3.5")
        ctk.CTkEntry(cr,textvariable=self.w_var,width=65).pack(side="left",padx=(0,10))
        ctk.CTkLabel(cr,text="Real length (m):",font=("Segoe UI",12)).pack(side="left",padx=(0,4))
        self.l_var=tk.StringVar(value="8.0")
        ctk.CTkEntry(cr,textvariable=self.l_var,width=65).pack(side="left",padx=(0,10))
        ctk.CTkButton(cr,text="✕  Clear points",width=120,height=38,
            fg_color="transparent",border_width=1,font=("Segoe UI",12),
            command=self._clear).pack(side="left",padx=(0,8))
        self.cal_btn=ctk.CTkButton(cr,text="📐  Calibrate",width=120,height=38,
            fg_color=ACC_TEAL,hover_color="#0d9488",
            font=("Segoe UI",13,"bold"),state="disabled",
            command=self._calibrate)
        self.cal_btn.pack(side="left",padx=(0,8))
        self.status_lbl=ctk.CTkLabel(cr,text="Load a video, then click 4 corners on the road.",
                                      font=("Segoe UI",11),text_color="#64748b")
        self.status_lbl.pack(side="left",padx=(6,0))

        # Canvas
        cf=ctk.CTkFrame(self,corner_radius=12,border_width=1)
        cf.grid(row=3,column=0,padx=32,pady=(0,24),sticky="nsew")
        self.canvas=tk.Canvas(cf,bg="#0a0d14",highlightthickness=0,cursor="crosshair")
        self.canvas.pack(fill="both",expand=True,padx=2,pady=2)
        self.canvas.bind("<Button-1>",self._click)

    def _load(self):
        from tkinter import filedialog
        p=filedialog.askopenfilename(initialdir="videos",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")])
        if not p: return
        cap=cv2.VideoCapture(p)
        # Get middle frame for best road visibility
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES,total//2)
        ret,frame=cap.read(); cap.release()
        if ret:
            self._frame=frame.copy(); self._pts=[]; self._redraw()
            self.status_lbl.configure(text="Click 4 corners of a known rectangle on the road.")

    def _click(self,e):
        if self._frame is None or len(self._pts)>=4: return
        # Convert canvas coords back to frame coords
        h,w=self._frame.shape[:2]
        cw=max(self.canvas.winfo_width(),640); ch=max(self.canvas.winfo_height(),360)
        sc=min(cw/w,ch/h,1.0); nw,nh=int(w*sc),int(h*sc)
        ox=(cw-nw)//2; oy=(ch-nh)//2
        fx=max(0,min(w-1,int((e.x-ox)/sc)))
        fy=max(0,min(h-1,int((e.y-oy)/sc)))
        self._pts.append((fx,fy))
        self._redraw()
        n=len(self._pts)
        if n<4:
            self.status_lbl.configure(text=f"Point {n}/4 set. Click next corner.")
        else:
            self.status_lbl.configure(text="4 points set! Enter dimensions and click Calibrate.")
            self.cal_btn.configure(state="normal")

    def _redraw(self):
        self.canvas.delete("all")
        if self._frame is None: return
        h,w=self._frame.shape[:2]
        cw=max(self.canvas.winfo_width(),640); ch=max(self.canvas.winfo_height(),360)
        sc=min(cw/w,ch/h,1.0); nw,nh=int(w*sc),int(h*sc)
        ox=(cw-nw)//2; oy=(ch-nh)//2
        self._ox=ox;self._oy=oy;self._sc=sc
        disp=cv2.resize(self._frame,(nw,nh))
        from PIL import Image,ImageTk
        rgb=cv2.cvtColor(disp,cv2.COLOR_BGR2RGB)
        self._photo=ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(ox,oy,anchor="nw",image=self._photo)
        cols=["#2dd4bf","#fbbf24","#f87171","#a78bfa"]
        labels=["TL","TR","BR","BL"]
        for i,(fx,fy) in enumerate(self._pts):
            px=ox+int(fx*sc); py=oy+int(fy*sc)
            c=cols[i]
            self.canvas.create_oval(px-8,py-8,px+8,py+8,fill=c,outline="white",width=2)
            self.canvas.create_text(px+12,py,text=labels[i],fill=c,
                                    font=("Segoe UI",11,"bold"))
        if len(self._pts)==4:
            pts=[(ox+int(fx*sc),oy+int(fy*sc)) for fx,fy in self._pts]
            flat=[c for xy in pts+[pts[0]] for c in xy]
            self.canvas.create_line(flat,fill="#2dd4bf",width=2,dash=(6,3))

    def _clear(self):
        self._pts=[]; self.cal_btn.configure(state="disabled")
        self.status_lbl.configure(text="Points cleared. Click 4 corners.")
        self._redraw()

    def _calibrate(self):
        if len(self._pts)!=4:
            self.status_lbl.configure(text="Need exactly 4 points!",text_color=ACC_RED)
            return
        try:
            W=float(self.w_var.get()); L=float(self.l_var.get())
        except ValueError:
            self.status_lbl.configure(text="Invalid dimensions.",text_color=ACC_RED)
            return
        # Image points
        img_pts=np.float32(self._pts)
        # World points: TL, TR, BR, BL of rectangle W×L metres
        wld_pts=np.float32([[0,0],[W,0],[W,L],[0,L]])
        H,mask=cv2.findHomography(img_pts,wld_pts,cv2.RANSAC,5.0)
        if H is None:
            self.status_lbl.configure(text="Calibration failed — try different points.",
                                       text_color=ACC_RED)
            return
        os.makedirs("data",exist_ok=True)
        np.save("data/homography.npy",H)
        inliers=int(mask.sum()) if mask is not None else 4
        self.status_lbl.configure(
            text=f"✓ Calibrated! ({inliers}/4 inliers) — homography.npy saved. "
                 f"Restart detection for accurate speed.",
            text_color=ACC_GREEN)
        self.cal_btn.configure(state="disabled")
        if self.status_bar:
            self.status_bar.set("Homography calibrated ✓ — speed will be accurate","idle")


# ================================================================
#  ANALYTICS DASHBOARD  — Advanced
# ================================================================
class DashboardPage(Page):
    def __init__(self,master):
        super().__init__(master)
        self.grid_rowconfigure(3,weight=1)
        self._ever_shown=False
        self.page_header("📊","Analytics Dashboard","KPI strip · 8 chart types · export")

        # ── KPI summary strip (from summary CSVs) ─────────────
        kpi_f=ctk.CTkFrame(self,fg_color="transparent")
        kpi_f.grid(row=1,column=0,padx=32,pady=(10,0),sticky="ew")
        kpi_f.grid_columnconfigure(list(range(7)),weight=1)
        self._kpi_cards={}
        kpi_items=[
            ("total",  "Total Vehicles", "🚗", ACC_BLUE),
            ("phf",    "PHF",            "📈", ACC_AMBER),
            ("v85",    "V85 km/h",       "🚀", ACC_GREEN),
            ("los",    "LOS",            "🏁", ACC_PURPLE),
            ("headway","Avg Headway",    "⏱", ACC_TEAL),
            ("satflow","Sat. Flow",      "🔄", "#fb923c"),
            ("safety", "Safety Events",  "⚠️", ACC_RED),
        ]
        for col,(key,lbl,icon,acc) in enumerate(kpi_items):
            f=ctk.CTkFrame(kpi_f,corner_radius=10,border_width=1)
            f.grid(row=0,column=col,padx=(0 if col==0 else 6,0),sticky="ew")
            ctk.CTkFrame(f,height=3,corner_radius=0,fg_color=acc).pack(fill="x")
            ctk.CTkLabel(f,text=f"{icon} {lbl}",font=("Segoe UI",8),
                         text_color="#64748b").pack(pady=(6,0))
            v=ctk.CTkLabel(f,text="—",font=("Segoe UI",14,"bold"),text_color=acc)
            v.pack(pady=(0,6))
            self._kpi_cards[key]=v

        # ── Filter bar ────────────────────────────────────────
        fbar=ctk.CTkFrame(self,corner_radius=12,border_width=1)
        fbar.grid(row=2,column=0,padx=32,pady=(10,0),sticky="ew")

        r1=ctk.CTkFrame(fbar,fg_color="transparent"); r1.pack(padx=16,pady=(10,4),fill="x")
        self.cvar=tk.StringVar(value="Daily")
        chart_types=["Daily","Hourly+Speed","Monthly","Types","Speed Dist","LOS Timeline","Direction","TMC Matrix","By Zone"]
        for lbl in chart_types:
            ctk.CTkRadioButton(r1,text=lbl,variable=self.cvar,value=lbl,
                               font=("Segoe UI",11),command=self._render_chart
            ).pack(side="left",padx=(0,12))

        r2=ctk.CTkFrame(fbar,fg_color="transparent"); r2.pack(padx=16,pady=(0,10),fill="x")
        ctk.CTkLabel(r2,text="From:",font=("Segoe UI",11)).pack(side="left",padx=(0,4))
        self.sv=tk.StringVar()
        ctk.CTkEntry(r2,textvariable=self.sv,placeholder_text="YYYY-MM-DD",width=110
                    ).pack(side="left",padx=(0,4))
        ctk.CTkButton(r2,text="📅",width=26,height=26,font=("Segoe UI",11),
            fg_color="transparent",border_width=1,
            command=lambda: self._pick_date(self.sv)).pack(side="left",padx=(0,10))
        ctk.CTkLabel(r2,text="To:",font=("Segoe UI",11)).pack(side="left",padx=(0,4))
        self.ev=tk.StringVar()
        ctk.CTkEntry(r2,textvariable=self.ev,placeholder_text="YYYY-MM-DD",width=110
                    ).pack(side="left",padx=(0,4))
        ctk.CTkButton(r2,text="📅",width=26,height=26,font=("Segoe UI",11),
            fg_color="transparent",border_width=1,
            command=lambda: self._pick_date(self.ev)).pack(side="left",padx=(0,12))
        ctk.CTkLabel(r2,text="Session:",font=("Segoe UI",11)).pack(side="left",padx=(0,4))
        self.sess_var=tk.StringVar(value="All")
        self.sess_cb=ctk.CTkComboBox(r2,variable=self.sess_var,values=["All"],
            width=155,command=lambda _: self._render_chart())
        self.sess_cb.pack(side="left",padx=(0,10))
        ctk.CTkButton(r2,text="✕ Clear",width=72,height=26,font=("Segoe UI",11),
            fg_color="transparent",border_width=1,
            command=self._clear_filters).pack(side="left",padx=(0,6))
        ctk.CTkButton(r2,text="↺ Refresh",width=80,height=26,font=("Segoe UI",11),
            command=self._render_chart).pack(side="left",padx=(0,6))
        ctk.CTkButton(r2,text="💾 PNG",width=72,height=26,font=("Segoe UI",11),
            fg_color=ACC_TEAL,hover_color="#0d9488",
            command=self._export_png).pack(side="left")

        # ── Chart canvas ──────────────────────────────────────
        cf=ctk.CTkFrame(self,corner_radius=14,border_width=1)
        cf.grid(row=3,column=0,padx=32,pady=(10,24),sticky="nsew")
        cf.grid_rowconfigure(1,weight=1); cf.grid_columnconfigure(0,weight=1)
        tb_f=ctk.CTkFrame(cf,corner_radius=0,height=32)
        tb_f.grid(row=0,column=0,sticky="ew",padx=2,pady=(2,0))
        tb_f.grid_propagate(False)

        dark = ctk.get_appearance_mode() == "Dark"
        bg = "#111827" if dark else "#ffffff"
        self.fig = Figure(facecolor=bg)
        self.fig.set_tight_layout(False)
        self.fig.subplots_adjust(left=0.08,right=0.97,top=0.88,bottom=0.18)
        self.ax=self.fig.add_subplot(111)
        self._style_ax()
        self.canvas=FigureCanvasTkAgg(self.fig,master=cf)
        self.canvas.get_tk_widget().grid(row=1,column=0,sticky="nsew",padx=4,pady=4)
        self.toolbar=NavigationToolbar2Tk(self.canvas,tb_f)
        self.toolbar.update()

    # ── KPI strip loader ──────────────────────────────────────
    def _load_kpis(self):
        """Load KPI strip metrics — filtered by current session selection."""
        try:
            sfiles=glob.glob(os.path.join("data","*_summary.csv"))
            if not sfiles: return
            sdf=pd.concat([pd.read_csv(f) for f in sfiles],ignore_index=True)
            if sdf.empty: return

            # Apply session filter — match same selection as chart area
            sv = self.sess_var.get()
            if sv and sv != "All" and "session" in sdf.columns:
                sdf = sdf[sdf["session"]==sv]
                if sdf.empty: return

            total = int(sdf["total_vehicles"].sum()) if "total_vehicles" in sdf else 0
            phf   = f"{sdf['phf'].mean():.2f}"            if "phf" in sdf else "—"
            v85   = f"{sdf['speed_85th_kmh'].mean():.0f}" if "speed_85th_kmh" in sdf else "—"
            hdwy  = f"{sdf['avg_headway_sec'].mean():.1f}s" if "avg_headway_sec" in sdf else "—"
            sat   = f"{int(sdf['saturation_flow_vph'].mean())}" if "saturation_flow_vph" in sdf else "—"
            sev   = f"{int(sdf['safety_events'].sum())}"   if "safety_events" in sdf else "—"
            # LOS from last session in filtered set
            los   = str(sdf["los_letter"].iloc[-1]) if "los_letter" in sdf else "—"
            los_colours={"A":ACC_GREEN,"B":ACC_GREEN,"C":ACC_TEAL,
                         "D":ACC_AMBER,"E":ACC_RED,"F":"#7f1d1d","—":"#64748b"}
            self._kpi_cards["total"].configure(text=str(total))
            self._kpi_cards["phf"].configure(text=phf)
            self._kpi_cards["v85"].configure(text=v85)
            self._kpi_cards["los"].configure(text=los,
                text_color=los_colours.get(los,"#64748b"))
            self._kpi_cards["headway"].configure(text=hdwy)
            self._kpi_cards["satflow"].configure(text=sat)
            self._kpi_cards["safety"].configure(text=sev)
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────
    def _pick_date(self,var):
        import datetime as dt
        top=tk.Toplevel(self); top.title("Pick date"); top.geometry("260x200")
        top.configure(bg="#1e2535"); top.resizable(False,False)
        today=dt.date.today()
        tk.Label(top,text="Enter date (YYYY-MM-DD):",bg="#1e2535",fg="#e8eaf0",
                 font=("Segoe UI",11)).pack(pady=(18,6))
        e=tk.Entry(top,font=("Segoe UI",13),width=16,justify="center")
        e.insert(0,str(today)); e.pack(pady=4)
        def quick(days): d=today-dt.timedelta(days=days); e.delete(0,"end"); e.insert(0,str(d))
        bf=tk.Frame(top,bg="#1e2535"); bf.pack(pady=6)
        for label,days in [("Today",0),("7d",7),("30d",30)]:
            tk.Button(bf,text=label,bg="#2d3748",fg="#94a3b8",relief="flat",
                      command=lambda d=days: quick(d)).pack(side="left",padx=4)
        def ok(): var.set(e.get().strip()); top.destroy(); self._render_chart()
        tk.Button(top,text="OK",bg="#3b82f6",fg="white",font=("Segoe UI",12),
                  relief="flat",padx=20,command=ok).pack(pady=8)

    def _clear_filters(self):
        self.sv.set(""); self.ev.set("")
        self.sess_var.set("All"); self._render_chart()

    def _update_sessions(self,df):
        if df is None or df.empty or "session" not in df.columns:
            self.sess_cb.configure(values=["All"]); return
        sessions=["All"]+sorted(df["session"].dropna().unique().tolist())
        self.sess_cb.configure(values=sessions)

    def _style_ax(self, ax=None):
        if ax is None: ax = self.ax
        dark = ctk.get_appearance_mode() == "Dark"
        bg   = "#111827" if dark else "#ffffff"   # slightly lighter than #0e1117
        grid = "#1f2d3d" if dark else "#e2e8f0"   # brighter grid lines
        fg   = "#f1f5f9" if dark else "#0f172a"   # brighter text
        ax.set_facecolor(bg)
        ax.tick_params(colors="#7a90b0", labelsize=9)
        ax.title.set_color(fg)
        ax.title.set_fontweight("bold")
        for sp in ax.spines.values():
            sp.set_color(grid); sp.set_linewidth(0.7)
        ax.grid(True, color=grid, linewidth=0.6, linestyle="--", alpha=0.7)

    def _df(self):
        files=glob.glob(os.path.join("data","log_*.csv"))
        if not files: return pd.DataFrame()
        dfs=[d for d in [pd.read_csv(f) for f in files] if not d.empty]
        if not dfs: return pd.DataFrame()
        df=pd.concat(dfs,ignore_index=True)
        if "timestamp" in df.columns:
            df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
            df["date"]=df["timestamp"].dt.date
            df["hour"]=df["timestamp"].dt.hour
            df["month"]=df["timestamp"].dt.to_period("M").astype(str)
        self._update_sessions(df)
        s=self.sv.get().strip(); e=self.ev.get().strip()
        if s and "date" in df.columns:
            try: df=df[df["date"]>=datetime.date.fromisoformat(s)]
            except: pass
        if e and "date" in df.columns:
            try: df=df[df["date"]<=datetime.date.fromisoformat(e)]
            except: pass
        sv=self.sess_var.get()
        if sv and sv!="All" and "session" in df.columns:
            df=df[df["session"]==sv]
        return df

    def _export_png(self):
        from tkinter import filedialog
        path=filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png")],initialfile="veloxis_chart.png")
        if not path: return
        self.fig.savefig(path,dpi=180,bbox_inches="tight")

    # ── Chart renderer ────────────────────────────────────────
    def _render_chart(self):
        df=self._df(); self._load_kpis()
        dark=ctk.get_appearance_mode()=="Dark"
        bg="#0e1117" if dark else "#ffffff"
        fg="#e8eaf0" if dark else "#0f172a"
        mut="#64748b"

        # Clear and reset
        self.fig.clear()
        self.fig.set_facecolor(bg)
        self.fig.set_tight_layout(False)

        chart=self.cvar.get()

        # ── helper: add value labels on bars ─────────────────
        def _bar_labels(ax,bars,color=mut,fmt="{:.0f}",rot=0):
            for b in bars:
                h=b.get_height()
                if h>0:
                    ax.text(b.get_x()+b.get_width()/2, h+max(h*0.01,0.3),
                            fmt.format(h), ha="center", va="bottom",
                            color=color, fontsize=8, rotation=rot)

        if df.empty:
            ax=self.fig.add_subplot(111)
            self._style_ax(ax)
            ax.text(0.5,0.5,"No data yet.\nRun a detection session first.",
                ha="center",va="center",color=mut,fontsize=14,transform=ax.transAxes)
            self.canvas.draw_idle(); return

        # ── DAILY ─────────────────────────────────────────────
        if chart=="Daily" and "date" in df.columns:
            ax=self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.08,right=0.97,top=0.88,bottom=0.22)
            self._style_ax(ax)
            c=df.groupby("date").size()
            x=range(len(c))
            bars = ax.bar(x, c.values, color=CHART_PALETTE[0], alpha=0.88, width=0.65,
                          edgecolor="#2563eb", linewidth=0.5)
            # 7-day rolling average line if enough data
            if len(c) >= 3:
                import numpy as np
                roll = pd.Series(c.values).rolling(min(3, len(c)), min_periods=1).mean()
                ax.plot(list(x), roll.values, "--",
                        color=CHART_PALETTE[2], linewidth=2.0, alpha=0.95, label="3-day avg")
                ax.legend(facecolor=bg, labelcolor=fg, fontsize=9, framealpha=0.5)
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(d) for d in c.index],rotation=35,ha="right",fontsize=8)
            ax.set_title("Daily Vehicle Count",color=fg,fontsize=13,pad=10)
            ax.set_ylabel("Vehicles",color=mut,fontsize=10)
            _bar_labels(ax,bars)

        # ── HOURLY + SPEED ────────────────────────────────────
        elif chart=="Hourly+Speed":
            ax=self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.09,right=0.92,top=0.88,bottom=0.18)
            self._style_ax(ax)
            if "hour" not in df.columns:
                ax.text(0.5,0.5,"No timestamp data.",ha="center",va="center",
                    color=mut,fontsize=13,transform=ax.transAxes)
            else:
                c=df.groupby("hour").size().reindex(range(24),fill_value=0)
                bars=ax.bar(c.index,c.values,color=ACC_BLUE,alpha=0.8,width=0.7,
                            edgecolor="#1d4ed8",linewidth=0.4,label="Volume")
                ax.set_ylabel("Vehicles / hour",color=ACC_BLUE,fontsize=10)
                ax.tick_params(axis='y',colors=ACC_BLUE)
                # Speed overlay on right axis
                if "speed_kmh" in df.columns:
                    spd=df[df["speed_kmh"].notna() & (df["speed_kmh"]>0)]
                    if not spd.empty:
                        ax2=ax.twinx()
                        avg_spd=spd.groupby("hour")["speed_kmh"].mean().reindex(range(24))
                        ax2.plot(avg_spd.index,avg_spd.values,"o-",
                                 color=ACC_AMBER,linewidth=2,markersize=4,
                                 alpha=0.9,label="Avg Speed")
                        ax2.set_ylabel("Speed (km/h)",color=ACC_AMBER,fontsize=10)
                        ax2.tick_params(axis='y',colors=ACC_AMBER,labelsize=9)
                        ax2.spines["right"].set_color(ACC_AMBER)
                        ax2.grid(False)
                        # Combined legend
                        h1,l1=ax.get_legend_handles_labels()
                        h2,l2=ax2.get_legend_handles_labels()
                        ax.legend(h1+h2,l1+l2,facecolor=bg,labelcolor=fg,
                                  fontsize=9,framealpha=0.5,loc="upper left")
                ax.set_xticks(range(0,24,2))
                ax.set_xticklabels([f"{h:02d}:00" for h in range(0,24,2)],
                                   rotation=30,ha="right",fontsize=8)
                ax.set_title("Hourly Volume + Average Speed",color=fg,fontsize=13,pad=10)

        # ── MONTHLY ───────────────────────────────────────────
        elif chart=="Monthly" and "month" in df.columns:
            ax=self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.09,right=0.97,top=0.88,bottom=0.18)
            self._style_ax(ax)
            c=df.groupby("month").size()
            bars = ax.bar(range(len(c)), c.values, color=CHART_PALETTE[1],
                          alpha=0.88, width=0.65, edgecolor="#b45309", linewidth=0.5)
            ax.set_xticks(range(len(c)))
            ax.set_xticklabels(c.index.astype(str),rotation=20,ha="right",fontsize=9)
            ax.set_title("Monthly Traffic Trend",color=fg,fontsize=13,pad=10)
            ax.set_ylabel("Vehicles",color=mut,fontsize=10)
            _bar_labels(ax,bars)

        # ── TYPES (donut + bar side-by-side) ──────────────────
        elif chart=="Types" and "vehicle_type" in df.columns:
            self.fig.subplots_adjust(left=0.04,right=0.98,top=0.88,bottom=0.06,wspace=0.3)
            ax1=self.fig.add_subplot(121)
            ax2=self.fig.add_subplot(122)
            self._style_ax(ax1); self._style_ax(ax2)
            c=df["vehicle_type"].value_counts()
            cols = CHART_PALETTE[:len(c)]
            # Donut
            wedges,_,auts=ax1.pie(c.values,labels=None,autopct="%1.0f%%",colors=cols,
                startangle=140,pctdistance=0.75,
                wedgeprops={"linewidth":2,"edgecolor":bg,"width":0.55})
            for at in auts: at.set_color("#0f172a"); at.set_fontsize(8)
            ax1.set_title("Type Distribution",color=fg,fontsize=12,pad=8)
            ax1.legend(c.index,loc="lower center",ncol=2,fontsize=8,
                       facecolor=bg,labelcolor=fg,framealpha=0.5,
                       bbox_to_anchor=(0.5,-0.08))
            # Horizontal bar
            y=range(len(c))
            ax2.barh(list(y),c.values,color=cols,alpha=0.85,edgecolor=bg,linewidth=0.3)
            ax2.set_yticks(list(y))
            ax2.set_yticklabels(c.index,fontsize=9)
            ax2.set_xlabel("Count",color=mut,fontsize=9)
            ax2.set_title("Count by Type",color=fg,fontsize=12,pad=8)
            for i,v in enumerate(c.values):
                ax2.text(v+max(c.values)*0.01,i,str(v),va="center",
                         color=mut,fontsize=8)
            ax2.grid(axis="x",color="#1f2937" if dark else "#e2e8f0",
                     linewidth=0.5,linestyle="--",alpha=0.6)
            ax2.set_facecolor(bg)
            ax2.tick_params(colors="#64748b",labelsize=9)
            ax2.title.set_color(fg)
            for sp in ax2.spines.values():
                sp.set_color("#1f2937" if dark else "#e2e8f0")

        # ── SPEED DISTRIBUTION ────────────────────────────────
        elif chart=="Speed Dist":
            ax=self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.09,right=0.97,top=0.88,bottom=0.16)
            self._style_ax(ax)
            if "speed_kmh" not in df.columns:
                ax.text(0.5,0.5,"No speed data.\nEnable PIXELS_PER_METER or homography.",
                    ha="center",va="center",color=mut,fontsize=13,transform=ax.transAxes)
            else:
                spd=df["speed_kmh"].dropna()
                spd=spd[(spd>0)&(spd<150)]
                if spd.empty:
                    ax.text(0.5,0.5,"No valid speed readings yet.",
                        ha="center",va="center",color=mut,fontsize=13,transform=ax.transAxes)
                else:
                    import numpy as np
                    n,bins,patches=ax.hist(spd,bins=30,color=ACC_BLUE,
                                           alpha=0.75,edgecolor=bg,linewidth=0.4)
                    # Colour bars above V85
                    v85p=float(spd.quantile(0.85))
                    for patch,left in zip(patches,bins[:-1]):
                        if left>=v85p: patch.set_facecolor(ACC_RED)
                    # V85 line
                    ax.axvline(v85p,color=ACC_RED,linewidth=2,linestyle="--",
                               label=f"V85 = {v85p:.0f} km/h")
                    # Mean line
                    mn=float(spd.mean())
                    ax.axvline(mn,color=ACC_AMBER,linewidth=1.5,linestyle=":",
                               label=f"Mean = {mn:.0f} km/h")
                    # Speed limit line if saved
                    try:
                        lim=int(load_prefs().get("speed_limit","0") or 0)
                        if lim>0:
                            ax.axvline(lim,color=ACC_GREEN,linewidth=1.5,linestyle="-.",
                                       label=f"Limit = {lim} km/h")
                    except: pass
                    ax.legend(facecolor=bg,labelcolor=fg,fontsize=10,framealpha=0.5)
                    ax.set_title("Speed Distribution",color=fg,fontsize=13,pad=10)
                    ax.set_xlabel("Speed (km/h)",color=mut,fontsize=10)
                    ax.set_ylabel("Frequency",color=mut,fontsize=10)
                    # Annotation
                    ax.text(0.98,0.95,f"n = {len(spd)} readings",
                        transform=ax.transAxes,ha="right",va="top",
                        color=mut,fontsize=9)

        # ── LOS TIMELINE ──────────────────────────────────────
        elif chart=="LOS Timeline":
            ax=self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.06,right=0.97,top=0.88,bottom=0.22)
            self._style_ax(ax)
            sfiles=glob.glob(os.path.join("data","*_summary.csv"))
            if not sfiles:
                ax.text(0.5,0.5,"No summary data yet.\nRun detection sessions first.",
                    ha="center",va="center",color=mut,fontsize=13,transform=ax.transAxes)
            else:
                try:
                    sdf=pd.concat([pd.read_csv(f) for f in sorted(sfiles)],ignore_index=True)
                    sdf=sdf.dropna(subset=["los_letter"]) if "los_letter" in sdf.columns else sdf
                    if sdf.empty or "los_letter" not in sdf.columns:
                        raise ValueError("no LOS data")
                    los_map={"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}
                    los_cols_m={"A":ACC_GREEN,"B":"#65a30d","C":ACC_TEAL,
                                "D":ACC_AMBER,"E":ACC_RED,"F":"#7f1d1d"}
                    sdf["los_num"]=sdf["los_letter"].map(los_map).fillna(0)
                    x=range(len(sdf))
                    # Step line
                    ax.step(list(x),sdf["los_num"].tolist(),where="mid",
                            color=ACC_BLUE,linewidth=2,alpha=0.8)
                    # Coloured scatter
                    for i,row in sdf.iterrows():
                        col=los_cols_m.get(str(row.get("los_letter","—")),mut)
                        ax.scatter(list(x)[list(sdf.index).index(i)],
                                   row["los_num"],color=col,s=80,zorder=5)
                    # PHF overlay on right axis
                    if "phf" in sdf.columns:
                        ax2=ax.twinx()
                        ax2.plot(list(x),sdf["phf"].tolist(),"o--",
                                 color=ACC_AMBER,linewidth=1.5,markersize=4,
                                 alpha=0.8,label="PHF")
                        ax2.set_ylabel("PHF",color=ACC_AMBER,fontsize=9)
                        ax2.tick_params(axis='y',colors=ACC_AMBER,labelsize=8)
                        ax2.set_ylim(0,1.1); ax2.grid(False)
                        ax2.axhline(0.85,color=ACC_AMBER,linewidth=0.8,
                                    linestyle=":",alpha=0.5)
                        ax2.text(len(sdf)-0.5,0.86,"ideal PHF",
                                 color=ACC_AMBER,fontsize=7,ha="right")
                    ax.set_yticks([1,2,3,4,5,6])
                    ax.set_yticklabels(["A","B","C","D","E","F"],
                                       fontsize=11,fontweight="bold")
                    ax.set_ylim(0.5,6.5); ax.invert_yaxis()
                    sessions=sdf.get("session",pd.Series(range(len(sdf)))).tolist()
                    short=[str(s)[:10]+"…" if len(str(s))>12 else str(s) for s in sessions]
                    ax.set_xticks(list(x))
                    ax.set_xticklabels(short,rotation=35,ha="right",fontsize=8)
                    ax.set_title("LOS + PHF Timeline — Session by Session",
                                 color=fg,fontsize=13,pad=10)
                    ax.set_ylabel("Level of Service",color=fg,fontsize=10)
                    # LOS band legend
                    for grade,num in los_map.items():
                        col=los_cols_m.get(grade,mut)
                        ax.axhspan(num-0.4,num+0.4,alpha=0.06,color=col)
                except Exception as ex:
                    ax.text(0.5,0.5,f"Could not load LOS data:\n{ex}",
                        ha="center",va="center",color=mut,fontsize=11,transform=ax.transAxes)

        # ── DIRECTION ─────────────────────────────────────────
        elif chart=="Direction":
            ax=self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.09,right=0.97,top=0.88,bottom=0.22)
            self._style_ax(ax)
            chart_bg=bg
            if "direction" not in df.columns or df.empty:
                ax.text(0.5,0.5,"No direction data yet.",
                    ha="center",va="center",color=mut,fontsize=13,transform=ax.transAxes)
            else:
                fwd=df[df["direction"].str.contains("FWD|Forward|→",na=False,regex=True)]
                bwd=df[df["direction"].str.contains("BWD|Backward|←",na=False,regex=True)]
                sess_val=self.sess_var.get()
                if sess_val!="All" and "session" in df.columns:
                    types=sorted(df["vehicle_type"].dropna().unique().tolist())
                    fwd_c=[len(fwd[fwd["vehicle_type"]==t]) for t in types]
                    bwd_c=[len(bwd[bwd["vehicle_type"]==t]) for t in types]
                else:
                    if "session" in df.columns:
                        types=sorted(df["session"].dropna().unique().tolist())
                        fwd_c=[len(fwd[fwd["session"]==t]) for t in types]
                        bwd_c=[len(bwd[bwd["session"]==t]) for t in types]
                    else:
                        types=["All"]
                        fwd_c=[len(fwd)]; bwd_c=[len(bwd)]
                x=list(range(len(types))); w2=0.36
                b1=ax.bar([i-w2/2 for i in x],fwd_c,w2,label="FWD",
                          color=ACC_BLUE,alpha=0.85,edgecolor="#1d4ed8",linewidth=0.4)
                b2=ax.bar([i+w2/2 for i in x],bwd_c,w2,label="BWD",
                          color=ACC_AMBER,alpha=0.85,edgecolor="#b45309",linewidth=0.4)
                short=[str(t)[:12]+"…" if len(str(t))>14 else str(t) for t in types]
                ax.set_xticks(x); ax.set_xticklabels(short,rotation=30,ha="right",fontsize=9)
                ax.set_title("Forward vs Backward",color=fg,fontsize=13,pad=10)
                ax.set_ylabel("Vehicles",color=mut,fontsize=10)
                ax.legend(facecolor=bg,labelcolor=fg,fontsize=10,framealpha=0.5)
                _bar_labels(ax,list(b1)+list(b2))

        # ── TMC MATRIX ────────────────────────────────────────
        elif chart=="TMC Matrix":
            ax=self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.18,right=0.97,top=0.88,bottom=0.22)
            self._style_ax(ax)
            # Load from tmc CSV files
            tmc_files=glob.glob(os.path.join("data","*_tmc.csv"))
            if not tmc_files:
                ax.text(0.5,0.5,
                    "No TMC data yet.\n\n"
                    "1. Draw approach zones in Lane Drawing\n"
                    "2. Enable Zones in Settings\n"
                    "3. Run detection",
                    ha="center",va="center",color=mut,fontsize=12,
                    transform=ax.transAxes,linespacing=2)
            else:
                try:
                    tdf=pd.read_csv(sorted(tmc_files)[-1],index_col=0)
                    tdf=tdf.drop(columns=["TOTAL"],errors="ignore")
                    if tdf.empty: raise ValueError("empty")
                    import numpy as np
                    data=tdf.values.astype(float)
                    im=ax.imshow(data,cmap="Blues",aspect="auto")
                    ax.set_xticks(range(len(tdf.columns)))
                    ax.set_yticks(range(len(tdf.index)))
                    ax.set_xticklabels(tdf.columns,rotation=30,ha="right",fontsize=9)
                    ax.set_yticklabels(tdf.index,fontsize=9)
                    ax.set_xlabel("Exit Zone →",color=mut,fontsize=10)
                    ax.set_ylabel("← Entry Zone",color=mut,fontsize=10)
                    ax.set_title("Turning Movement Count Matrix",color=fg,fontsize=13,pad=10)
                    # Annotate each cell
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            v=int(data[i,j])
                            txt_col="#0f172a" if v>data.max()*0.5 else fg
                            ax.text(j,i,str(v) if v>0 else "—",
                                    ha="center",va="center",
                                    color=txt_col,fontsize=10,fontweight="bold")
                    self.fig.colorbar(im,ax=ax,label="Vehicles",shrink=0.8)
                    # Total row
                    totals=[int(data[:,j].sum()) for j in range(data.shape[1])]
                    ax.set_title(
                        f"TMC Matrix  ·  Total: {int(data.sum())} movements  "
                        f"·  Peak approach: {tdf.index[data.sum(axis=1).argmax()]}",
                        color=fg,fontsize=12,pad=10)
                except Exception as ex:
                    ax.text(0.5,0.5,f"TMC data error:\n{ex}",
                        ha="center",va="center",color=mut,fontsize=11,
                        transform=ax.transAxes)

        # ── BY ZONE ───────────────────────────────────────────
        elif chart=="By Zone":
            ax=self.fig.add_subplot(111)
            self.fig.subplots_adjust(left=0.16,right=0.97,top=0.88,bottom=0.12)
            self._style_ax(ax)
            if "zone" not in df.columns:
                ax.text(0.5,0.5,"No zone data.\nDraw lanes in Lane Drawing page.",
                    ha="center",va="center",color=mut,fontsize=13,transform=ax.transAxes)
            else:
                c=df.groupby("zone").size().sort_values(ascending=True)
                c=c[c.index!="all"] if "all" in c.index else c
                if c.empty:
                    ax.text(0.5,0.5,"Only 'all' zone found.\nDraw named lanes first.",
                        ha="center",va="center",color=mut,fontsize=13,transform=ax.transAxes)
                else:
                    cols=LANE_COLS[:len(c)]
                    ax.barh(c.index,c.values,color=cols,alpha=0.85,
                            edgecolor=bg,linewidth=0.4)
                    for i,v in enumerate(c.values):
                        ax.text(v+max(c.values)*0.01,i,str(v),va="center",
                                color=mut,fontsize=9)
                    ax.set_title("Volume by Road / Zone",color=fg,fontsize=13,pad=10)
                    ax.set_xlabel("Vehicles",color=mut,fontsize=10)
        else:
            ax=self.fig.add_subplot(111)
            self._style_ax(ax)
            ax.text(0.5,0.5,"No data for this chart type.",
                ha="center",va="center",color=mut,fontsize=13,transform=ax.transAxes)

        self.canvas.draw_idle()

    def refresh(self):
        if not self._ever_shown:
            self._ever_shown=True
        self._render_chart()


# ================================================================
#  LANE PAGE
# ================================================================
class LanePage(Page):
    COLS=LANE_COLS
    def __init__(self,master):
        super().__init__(master); self.cap=None; self.lanes=[]; self.cur_pts=[]; self.fphoto=None
        self.grid_rowconfigure(3,weight=1)
        self.page_header("🗺","Lane Drawing","Click to define road zones — any shape, any angle")

        r1=ctk.CTkFrame(self,corner_radius=12,border_width=1)
        r1.grid(row=1,column=0,padx=32,pady=(0,8),sticky="ew")
        rr=ctk.CTkFrame(r1,fg_color="transparent"); rr.pack(padx=14,pady=12,fill="x")
        ctk.CTkButton(rr,text="📂  Load Video",width=140,height=36,
            fg_color="transparent",border_width=1,font=("Segoe UI",13),
            command=self._load).pack(side="left",padx=(0,14))
        ctk.CTkLabel(rr,text="Seek:",font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.svar=tk.IntVar(value=0)
        self.slider=ctk.CTkSlider(rr,from_=0,to=100,variable=self.svar,
            command=self._seek,width=280,state="disabled")
        self.slider.pack(side="left",padx=(0,10))
        self.flbl=ctk.CTkLabel(rr,text="Frame 0",font=("Segoe UI",11),width=75)
        self.flbl.pack(side="left")

        ctk.CTkFrame(self,corner_radius=10,border_width=1,height=30
                    ).grid(row=2,column=0,padx=32,pady=(0,6),sticky="ew")
        ctk.CTkLabel(self,text="  LEFT CLICK = add point   ·   RIGHT CLICK = remove last point   ·   Name it → Finish Lane",
            font=("Segoe UI",11)).grid(row=2,column=0,padx=48,pady=0)

        cvf=ctk.CTkFrame(self,corner_radius=14,border_width=1)
        cvf.grid(row=3,column=0,padx=32,pady=(0,8),sticky="nsew")
        self.canvas=tk.Canvas(cvf,bg="#0a0d14",highlightthickness=0,cursor="crosshair")
        self.canvas.pack(fill="both",expand=True,padx=2,pady=2)
        self.canvas.bind("<Button-1>",self._click); self.canvas.bind("<Button-3>",self._rclick)

        r2=ctk.CTkFrame(self,fg_color="transparent"); r2.grid(row=4,column=0,padx=32,pady=(0,24),sticky="ew")
        self.ne=ctk.CTkEntry(r2,placeholder_text="Lane name (e.g. North Road)",width=240)
        self.ne.pack(side="left",padx=(0,10))
        for txt,fn,kw in [
            ("✓  Finish Lane",self._finish,{"fg_color":ACC_BLUE,"hover_color":"#2563eb","font":("Segoe UI",13,"bold")}),
            ("↩  Undo",self._undo,{"fg_color":"transparent","border_width":1}),
            ("✕  Clear",self._clr,{"fg_color":"transparent","border_width":1}),
        ]:
            ctk.CTkButton(r2,text=txt,width=120,height=38,command=fn,**kw
                         ).pack(side="left",padx=(0,8))
        ctk.CTkButton(r2,text="💾  Save All Lanes",width=160,height=38,
            fg_color="#064e3b",hover_color="#047857",border_width=1,
            border_color="#047857",text_color=(ACC_GREEN,"#047857"),
            font=("Segoe UI",13,"bold"),command=self._save).pack(side="left")
        self.ll=ctk.CTkLabel(r2,text="0 lanes",font=("Segoe UI",12))
        self.ll.pack(side="left",padx=14)

    def _load(self):
        from tkinter import filedialog
        p=filedialog.askopenfilename(initialdir="videos",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")])
        if not p: return
        if self.cap: self.cap.release()   # release previous handle before opening new
        self.cap=cv2.VideoCapture(p)
        tf=max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))-1,1)
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
            self.canvas.create_text(sum(p[0] for p in pts)/len(pts)+1,sum(p[1] for p in pts)/len(pts)+1,
                text=lane["name"],fill="#000",font=("Segoe UI",11,"bold"))
            self.canvas.create_text(sum(p[0] for p in pts)/len(pts),sum(p[1] for p in pts)/len(pts),
                text=lane["name"],fill="white",font=("Segoe UI",11,"bold"))
        if self.cur_pts:
            col=self.COLS[len(self.lanes)%len(self.COLS)]
            flat=[c for xy in self.cur_pts for c in xy]
            for px,py in self.cur_pts: self.canvas.create_oval(px-6,py-6,px+6,py+6,fill=col,outline="white")
            if len(self.cur_pts)>1: self.canvas.create_line(flat,fill=col,width=2,dash=(6,3))
            if len(self.cur_pts)>=3:
                self.canvas.create_line(self.cur_pts[-1][0],self.cur_pts[-1][1],
                    self.cur_pts[0][0],self.cur_pts[0][1],fill=col,width=1,dash=(4,4))

        # Draw auto counting line preview (midpoint of all lanes)
        if self.lanes:
            all_ys=[]
            for lane in self.lanes:
                all_ys.extend([oy+fy*nh for fx,fy in lane["points"]])
            if all_ys:
                road_top=min(all_ys); road_bot=max(all_ys)
                mid_y = (road_top+road_bot)/2
                line_a = road_top+(road_bot-road_top)*0.33
                line_b = road_top+(road_bot-road_top)*0.67
                x0=ox; x1=ox+nw
                # Single line (teal)
                self.canvas.create_line(x0,mid_y,x1,mid_y,fill="#2dd4bf",width=2,dash=(8,4))
                self.canvas.create_text(x0+6,mid_y-10,text="Counting line",
                    fill="#2dd4bf",font=("Segoe UI",9),anchor="w")
                # Dual lines (amber)
                self.canvas.create_line(x0,line_a,x1,line_a,fill="#fbbf24",width=1,dash=(6,4))
                self.canvas.create_line(x0,line_b,x1,line_b,fill="#fbbf24",width=1,dash=(6,4))
                self.canvas.create_text(x0+6,line_a-10,text="Line A (Fwd)",
                    fill="#fbbf24",font=("Segoe UI",8),anchor="w")
                self.canvas.create_text(x0+6,line_b-10,text="Line B (Bwd)",
                    fill="#fbbf24",font=("Segoe UI",8),anchor="w")
    def _click(self,e): self.cur_pts.append((e.x,e.y)); self._redraw()
    def _rclick(self,e):
        if self.cur_pts: self.cur_pts.pop(); self._redraw()
    def _finish(self):
        if len(self.cur_pts)<3:
            from tkinter import messagebox; messagebox.showwarning("Need points","Place ≥ 3 points."); return
        name=self.ne.get().strip() or f"Lane {len(self.lanes)+1}"
        nw=self._nw or 640; nh=self._nh or 360; ox=self._ox or 0; oy=self._oy or 0
        frac=[((px-ox)/max(nw,1),(py-oy)/max(nh,1)) for px,py in self.cur_pts]
        self.lanes.append({"name":name,"points":frac}); self.cur_pts=[]; self.ne.delete(0,"end")
        self.ll.configure(text=f"{len(self.lanes)} lane(s)"); self._redraw()
    def _undo(self):
        if self.lanes: self.lanes.pop(); self.ll.configure(text=f"{len(self.lanes)} lane(s)"); self._redraw()
    def _clr(self): self.cur_pts=[]; self._redraw()
    def _save(self):
        if not self.lanes:
            from tkinter import messagebox
            messagebox.showwarning("No lanes","Draw at least one."); return
        os.makedirs("data", exist_ok=True)
        with open("data/lanes.json","w",encoding="utf-8") as f:
            json.dump({"lanes":self.lanes}, f, indent=2)
        try:
            try:
                with open("config.py",encoding="utf-8") as f: c=f.read()
            except UnicodeDecodeError:
                with open("config.py",encoding="latin-1") as f: c=f.read()
            # Build ZONES from lane polygons
            zl=["ZONES = {\n"]
            for lane in self.lanes:
                xs=[p[0] for p in lane["points"]]
                ys=[p[1] for p in lane["points"]]
                zl.append(f'    "{lane["name"]}": ({min(xs):.3f},{min(ys):.3f},{max(xs):.3f},{max(ys):.3f}),\n')
            zl.append("}\n")
            c=re.sub(r"ENABLE_ZONES\s*=\s*\w+","ENABLE_ZONES = True",c)
            c=re.sub(r"ZONES\s*=\s*\{[^}]*\}","".join(zl).rstrip(),c,flags=re.DOTALL)

            # Auto-set counting line to road mid-point based on drawn polygon
            # Uses the vertical midpoint of ALL drawn lanes combined
            all_ys = []
            for lane in self.lanes:
                all_ys.extend([p[1] for p in lane["points"]])
            if all_ys:
                road_top    = min(all_ys)
                road_bottom = max(all_ys)
                # Single line: midpoint of road
                mid_y = round((road_top + road_bottom) / 2, 3)
                # Dual lines: 33% and 67% within road height
                line_a = round(road_top + (road_bottom - road_top) * 0.33, 3)
                line_b = round(road_top + (road_bottom - road_top) * 0.67, 3)
                # Update config
                c=re.sub(r"COUNTING_LINE_POSITION\s*=\s*[\d.]+",
                          f"COUNTING_LINE_POSITION = {mid_y}", c)
                c=re.sub(r"LINE_POS_A\s*=\s*[\d.]+",
                          f"LINE_POS_A = {line_a}", c)
                c=re.sub(r"LINE_POS_B\s*=\s*[\d.]+",
                          f"LINE_POS_B = {line_b}", c)

            with open("config.py","w",encoding="utf-8") as f: f.write(c)
        except Exception as e:
            pass

        from tkinter import messagebox
        messagebox.showinfo(
            "Saved ✓",
            f"{len(self.lanes)} lane(s) saved.\n"
            "Counting line auto-set to road midpoint.\n"
            "config.py updated."
        )


# ================================================================
#  SETTINGS PAGE
# ================================================================
class SettingsPage(Page):
    def __init__(self,master):
        super().__init__(master); self.grid_rowconfigure(1,weight=1)
        self.page_header("⚙️","Settings","Detection parameters & profile")
        scroll=ctk.CTkScrollableFrame(self,corner_radius=0)
        scroll.grid(row=1,column=0,padx=32,pady=(0,24),sticky="nsew")
        scroll.grid_columnconfigure(0,weight=1)

        def sec(t):
            f=ctk.CTkFrame(scroll,corner_radius=14,border_width=1)
            f.pack(fill="x",pady=(0,14)); SLabel(f,t).pack(anchor="w",padx=18,pady=(14,10)); return f
        def row(p,label,w,hint=""):
            r=ctk.CTkFrame(p,fg_color="transparent"); r.pack(fill="x",padx=16,pady=(0,10))
            ctk.CTkLabel(r,text=label,font=("Segoe UI",13),width=270,anchor="w").pack(side="left")
            w.pack(side="left",padx=(0,8))
            if hint: ctk.CTkLabel(r,text=hint,font=("Segoe UI",11)).pack(side="left")

        # Profile
        s0=sec("Profile — Your Identity")
        p=load_prefs()
        self.name_e=ctk.CTkEntry(s0,width=200); self.name_e.insert(0,p.get("author_name","Nishan"))
        row(s0,"Your name",self.name_e,"Shown in sidebar, watermark & reports")
        self.inst_e=ctk.CTkEntry(s0,width=280); self.inst_e.insert(0,p.get("institution","SUST · CEE Dept."))
        row(s0,"Institution",self.inst_e)

        # Model
        s1=sec("Detection Model")
        self.model_cb=ctk.CTkComboBox(s1,width=320,
            values=["bd_vehicles_yolo11.pt  (custom BD · YOLOv11)",
                    "bd_vehicles_best.pt  (custom BD · YOLOv8)",
                    "yolo11n.pt  (fastest — no custom model)",
                    "yolo11s.pt  (balanced — no custom model)",
                    "yolov8n.pt  (YOLOv8 fastest)",
                    "yolov8s.pt  (YOLOv8 balanced)"])
        row(s1,"YOLO Model",self.model_cb,"bd_vehicles_yolo11.pt = your trained custom model")
        self.conf=ctk.CTkSlider(s1,from_=0.1,to=0.9,width=260)
        self.conf.set(0.40); row(s1,"Confidence (0.1–0.9)",self.conf,"Lower = detect more, Higher = fewer false positives")

        # Speed
        s2=sec("Speed Estimation")
        self.ppm=ctk.CTkEntry(s2,width=120); self.ppm.insert(0,"55")
        row(s2,"Pixels per metre",self.ppm,"Measure a known distance in your video frame")
        self.fps_e=ctk.CTkEntry(s2,width=120); self.fps_e.insert(0,"25")
        row(s2,"Video FPS",self.fps_e)

        # Display
        s3=sec("Display Options")
        self.sw_sp=ctk.CTkSwitch(s3,text="Show speed (km/h)",onvalue=True,offvalue=False); self.sw_sp.select(); row(s3,"",self.sw_sp)
        self.sw_id=ctk.CTkSwitch(s3,text="Show track IDs",onvalue=True,offvalue=False); self.sw_id.select(); row(s3,"",self.sw_id)
        self.sw_zo=ctk.CTkSwitch(s3,text="Enable lane/zone counting",onvalue=True,offvalue=False); row(s3,"",self.sw_zo)

        # Performance
        s_perf=sec("Performance Mode")
        # Detect GPU now and show status
        try:
            import torch
            _gpu_ok = torch.cuda.is_available()
            _gpu_name = torch.cuda.get_device_name(0) if _gpu_ok else "Not detected"
        except Exception:
            _gpu_ok = False
            _gpu_name = "PyTorch not loaded yet"
        gpu_status = f"GPU: {_gpu_name}" if _gpu_ok else "GPU: None — integrated graphics / CPU only"
        gpu_color  = ACC_GREEN if _gpu_ok else ACC_AMBER
        ctk.CTkLabel(s_perf,text=f"  Hardware detected:  {gpu_status}",
                     font=("Segoe UI",11,"bold"),text_color=gpu_color
                     ).pack(anchor="w",padx=18,pady=(0,8))
        self.sw_cpu=ctk.CTkSwitch(s_perf,
                                   text="CPU Performance Mode",
                                   onvalue=True,offvalue=False,
                                   button_color=ACC_GREEN,progress_color=ACC_GREEN)
        if not _gpu_ok:
            self.sw_cpu.select()   # auto-ON if no GPU
        row(s_perf,"",self.sw_cpu)
        ctk.CTkLabel(s_perf,
                     text="  ON  → resize 416px, frame_skip=2  — smooth on HP Envy / Intel iGPU\n"
                          "  OFF → resize 640px, frame_skip=1  — full accuracy, use only with dedicated GPU",
                     font=("Segoe UI",10),justify="left",text_color="#64748b"
                     ).pack(anchor="w",padx=18,pady=(0,12))
        self.sw_dual=ctk.CTkSwitch(s3,text="Dual line mode (bidirectional road)",
                                    onvalue=True,offvalue=False,
                                    button_color=ACC_AMBER,progress_color=ACC_AMBER)
        row(s3,"",self.sw_dual)
        ctk.CTkLabel(s3,text="  Line A (teal) = upper = forward vehicles  |  Line B (amber) = lower = backward vehicles",
                     font=("Segoe UI",10),justify="left").pack(anchor="w",padx=16,pady=(0,6))
        self.lpos=ctk.CTkSlider(s3,from_=0.1,to=0.9,width=260); self.lpos.set(0.55)
        row(s3,"Single line position (when dual OFF)",self.lpos)
        self.lpos_a=ctk.CTkSlider(s3,from_=0.1,to=0.6,width=260); self.lpos_a.set(0.38)
        row(s3,"Line A position (upper, when dual ON)",self.lpos_a)
        self.lpos_b=ctk.CTkSlider(s3,from_=0.4,to=0.9,width=260); self.lpos_b.set(0.70)
        row(s3,"Line B position (lower, when dual ON)",self.lpos_b)

        # Custom model info
        s4=sec("Custom AI Model — Train for Bangladeshi Roads")
        ctk.CTkLabel(s4,font=("Segoe UI",12),
            text="Train YOLOv8 specifically on rickshaw, CNG, motorcycle, car, bus, truck, bicycle.\n"
                 "This will make detection angle-independent and more accurate for BD roads.",
            justify="left",wraplength=600).pack(anchor="w",padx=16,pady=(0,8))
        ctk.CTkButton(s4,text="🚀  Open Training Guide",width=200,height=36,
            fg_color=ACC_TEAL,hover_color="#0d9488",
            font=("Segoe UI",12,"bold"),
            command=lambda: subprocess.Popen([sys.executable,"train_custom_model.py"])
        ).pack(anchor="w",padx=16,pady=(0,14))

        # ── Study Location ────────────────────────────────────
        s5=sec("📍  Study Location — Site Tagging")
        ctk.CTkLabel(s5,font=("Segoe UI",11),text_color="#64748b",
            text="Tag your data collection site. Coordinates + name saved to every session CSV.",
            justify="left").pack(anchor="w",padx=16,pady=(0,12))

        loc_r=ctk.CTkFrame(s5,fg_color="transparent"); loc_r.pack(fill="x",padx=16,pady=(0,4))
        loc_r.grid_columnconfigure(1,weight=1)

        ctk.CTkLabel(loc_r,text="Location name:",font=("Segoe UI",12),
                     width=140,anchor="w").grid(row=0,column=0,pady=4,sticky="w")
        self.loc_name=ctk.CTkEntry(loc_r,
            placeholder_text="e.g. Zindabazar Intersection, Sylhet",width=380)
        self.loc_name.grid(row=0,column=1,sticky="ew",padx=(0,10))

        ctk.CTkLabel(loc_r,text="Latitude:",font=("Segoe UI",12),
                     width=140,anchor="w").grid(row=1,column=0,pady=4,sticky="w")
        lat_f=ctk.CTkFrame(loc_r,fg_color="transparent"); lat_f.grid(row=1,column=1,sticky="ew")
        self.loc_lat=ctk.CTkEntry(lat_f,placeholder_text="24.8949",width=155)
        self.loc_lat.pack(side="left",padx=(0,8))
        ctk.CTkLabel(lat_f,text="Longitude:",font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.loc_lng=ctk.CTkEntry(lat_f,placeholder_text="91.8687",width=155)
        self.loc_lng.pack(side="left")

        btn_r=ctk.CTkFrame(s5,fg_color="transparent"); btn_r.pack(fill="x",padx=16,pady=(8,4))
        ctk.CTkButton(btn_r,text="🌐  Lookup Address",width=165,height=36,
            fg_color="transparent",border_width=1,font=("Segoe UI",12),
            command=self._geocode_address).pack(side="left",padx=(0,8))
        ctk.CTkButton(btn_r,text="📍  Use GPS",width=130,height=36,
            fg_color="transparent",border_width=1,font=("Segoe UI",12),
            command=self._get_gps).pack(side="left",padx=(0,8))
        ctk.CTkButton(btn_r,text="🗺  Preview Map",width=140,height=36,
            fg_color=ACC_BLUE,hover_color="#2563eb",font=("Segoe UI",12),
            command=self._preview_map).pack(side="left")

        self.loc_status=ctk.CTkLabel(s5,text="",font=("Segoe UI",11),
                                      text_color="#64748b")
        self.loc_status.pack(anchor="w",padx=16,pady=(4,0))

        # Site preview — live label that reflects current entry fields (not old saved prefs)
        self._loc_preview = ctk.CTkLabel(s5,
            text="",font=("Segoe UI",10),text_color=ACC_TEAL,justify="left")
        self._loc_preview.pack(anchor="w",padx=16,pady=(6,0))
        # Update preview whenever entries change
        def _update_loc_preview(*_):
            n = self.loc_name.get().strip()
            la = self.loc_lat.get().strip()
            ln = self.loc_lng.get().strip()
            if n or la:
                self._loc_preview.configure(
                    text=f"  📍  {n or 'unnamed'}   Lat {la or '—'}  ·  Lng {ln or '—'}")
            else:
                self._loc_preview.configure(text="")
        self.loc_name.bind("<KeyRelease>", _update_loc_preview)
        self.loc_lat.bind("<KeyRelease>",  _update_loc_preview)
        self.loc_lng.bind("<KeyRelease>",  _update_loc_preview)
        _update_loc_preview()   # populate immediately from pre-filled values

        ctk.CTkLabel(s5,text="  Tip: GPS also auto-fills the location name via reverse geocoding.",
                     font=("Segoe UI",10),text_color="#64748b").pack(anchor="w",padx=16,pady=(4,0))

        # Road type + speed limit
        rtype_row=ctk.CTkFrame(s5,fg_color="transparent")
        rtype_row.pack(fill="x",padx=16,pady=(8,0))
        ctk.CTkLabel(rtype_row,text="Road type:",font=("Segoe UI",12),
                     width=140,anchor="w").pack(side="left")
        self.road_type_cb=ctk.CTkComboBox(rtype_row,width=220,
            values=["Urban arterial","Urban collector","Rural highway",
                    "Residential","Signalized intersection","Unsignalized intersection",
                    "Roundabout","Mid-block section"])
        self.road_type_cb.set(p.get("road_type","Urban arterial"))
        self.road_type_cb.pack(side="left",padx=(0,16))
        ctk.CTkLabel(rtype_row,text="Speed limit (km/h):",
                     font=("Segoe UI",12)).pack(side="left",padx=(0,6))
        self.speed_limit_e=ctk.CTkEntry(rtype_row,width=70)
        self.speed_limit_e.insert(0,p.get("speed_limit","50"))
        self.speed_limit_e.pack(side="left")

        ctk.CTkLabel(s5,
            text="  Road type and speed limit are included in map reports and session summaries.",
            font=("Segoe UI",10),text_color="#64748b").pack(anchor="w",padx=16,pady=(4,14))

        # Load saved location into fields
        if p.get("loc_name"): self.loc_name.insert(0,p["loc_name"])
        if p.get("loc_lat"):  self.loc_lat.insert(0,str(p["loc_lat"]))
        if p.get("loc_lng"):  self.loc_lng.insert(0,str(p["loc_lng"]))

        sf=ctk.CTkFrame(scroll,fg_color="transparent")
        sf.pack(fill="x",pady=(8,0))
        ctk.CTkButton(sf,text="💾  Save Settings",width=220,height=46,
            font=("Segoe UI",14,"bold"),corner_radius=10,
            fg_color=ACC_BLUE,hover_color="#2563eb",
            command=self._save
        ).pack(side="left")
        ctk.CTkLabel(sf,text="Changes take effect on next detection session.",
                     font=("Segoe UI",11),text_color="#64748b").pack(side="left",padx=14)
        self._load()

    def _load(self):
        try:
            import config
            m={"bd_vehicles_yolo11.pt":0,"bd_vehicles_best.pt":1,
                "yolo11n.pt":2,"yolo11s.pt":3,"yolov8n.pt":4,"yolov8s.pt":5}
            self.model_cb.set(self.model_cb.cget("values")[m.get(config.YOLO_MODEL,0)])
            self.conf.set(config.CONFIDENCE)
            self.ppm.delete(0,"end"); self.ppm.insert(0,str(config.PIXELS_PER_METER))
            self.fps_e.delete(0,"end"); self.fps_e.insert(0,str(config.VIDEO_FPS))
            (self.sw_sp.select if config.SHOW_SPEED else self.sw_sp.deselect)()
            (self.sw_id.select if config.SHOW_IDS else self.sw_id.deselect)()
            (self.sw_zo.select if config.ENABLE_ZONES else self.sw_zo.deselect)()
            cpu = getattr(config,'CPU_PERFORMANCE_MODE',True)
            (self.sw_cpu.select if cpu else self.sw_cpu.deselect)()
            dual = getattr(config,'USE_DUAL_LINES',False)
            (self.sw_dual.select if dual else self.sw_dual.deselect)()
            self.lpos.set(config.COUNTING_LINE_POSITION)
            self.lpos_a.set(getattr(config,'LINE_POS_A',0.38))
            self.lpos_b.set(getattr(config,'LINE_POS_B',0.70))
        except: pass

    def _geocode_address(self):
        """Nominatim (OpenStreetMap) geocoding — no API key needed."""
        name = self.loc_name.get().strip()
        if not name:
            self.loc_status.configure(text="Enter a location name first.",
                                       text_color=ACC_AMBER); return
        self.loc_status.configure(text="Looking up coordinates…",
                                   text_color=ACC_AMBER); self.update()
        try:
            import urllib.request, json as _json, urllib.parse
            q   = urllib.parse.quote(name)
            url = f"https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1"
            req = urllib.request.Request(
                url, headers={"User-Agent":"VELOXIS/2.0 traffic-research"})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = _json.loads(r.read())
            if data:
                lat = float(data[0]["lat"])
                lng = float(data[0]["lon"])
                disp = data[0].get("display_name","")[:70]
                self.loc_lat.delete(0,"end"); self.loc_lat.insert(0,f"{lat:.6f}")
                self.loc_lng.delete(0,"end"); self.loc_lng.insert(0,f"{lng:.6f}")
                self.loc_status.configure(
                    text=f"✓ {disp}…", text_color=ACC_GREEN)
                if hasattr(self, '_loc_preview'): self._update_loc_preview_now()
            else:
                self.loc_status.configure(
                    text="✗ Not found. Try more specific name (e.g. Zindabazar, Sylhet).",
                    text_color=ACC_RED)
        except Exception as e:
            self.loc_status.configure(
                text=f"✗ Lookup failed (check internet): {e}",
                text_color=ACC_RED)

    def _get_gps(self):
        """Try Windows Location API via PowerShell, then reverse-geocode to fill name."""
        self.loc_status.configure(text="Requesting GPS…",
                                   text_color=ACC_AMBER); self.update()
        try:
            ps = ("Add-Type -AssemblyName System.Device;"
                  "$w=New-Object System.Device.Location.GeoCoordinateWatcher;"
                  "$w.Start();Start-Sleep 4;"
                  "$c=$w.Position.Location;"
                  "Write-Output \"$($c.Latitude),$($c.Longitude)\"")
            r = subprocess.run(["powershell","-Command",ps],
                               capture_output=True,text=True,timeout=12)
            out = r.stdout.strip()
            if "," in out:
                lat_s, lng_s = out.split(",")
                lat, lng = float(lat_s.strip()), float(lng_s.strip())
                # Guard against NaN / zero from devices without GPS
                import math as _math
                if not (_math.isnan(lat) or _math.isnan(lng) or lat==0 or lng==0):
                    self.loc_lat.delete(0,"end"); self.loc_lat.insert(0,f"{lat:.6f}")
                    self.loc_lng.delete(0,"end"); self.loc_lng.insert(0,f"{lng:.6f}")
                    self.loc_status.configure(
                        text=f"✓ GPS acquired: {lat:.5f}, {lng:.5f}  — looking up name…",
                        text_color=ACC_GREEN)
                    if hasattr(self, '_loc_preview'): self._update_loc_preview_now()
                    self.update()
                    # Auto reverse-geocode in background so UI stays responsive
                    threading.Thread(
                        target=self._reverse_geocode,
                        args=(lat, lng), daemon=True).start()
                    return
            self.loc_status.configure(
                text="GPS unavailable — use Lookup from Address instead.",
                text_color=ACC_AMBER)
        except Exception:
            self.loc_status.configure(
                text="GPS not available on this device — use Lookup.",
                text_color=ACC_AMBER)

    def _update_loc_preview_now(self):
        """Force-refresh the live location preview label from current entry values."""
        n  = self.loc_name.get().strip()
        la = self.loc_lat.get().strip()
        ln = self.loc_lng.get().strip()
        if n or la:
            self._loc_preview.configure(
                text=f"  📍  {n or 'unnamed'}   Lat {la or '—'}  ·  Lng {ln or '—'}")
        else:
            self._loc_preview.configure(text="")

    def _reverse_geocode(self, lat, lng):
        """Nominatim reverse geocode — fills location name from GPS coords in English."""
        try:
            import urllib.request, json as _json
            url = (f"https://nominatim.openstreetmap.org/reverse"
                   f"?lat={lat}&lon={lng}&format=json&zoom=17&addressdetails=1")
            req = urllib.request.Request(url, headers={
                "User-Agent": "VELOXIS/2.0 traffic-research",
                "Accept-Language": "en"   # force English response
            })
            with urllib.request.urlopen(req, timeout=8) as r:
                data = _json.loads(r.read())
            addr = data.get("address", {})
            # Build: "Road/Locality, City" — prefer specific then broad
            parts = []
            for key in ("road", "neighbourhood", "suburb", "city_district"):
                val = addr.get(key, "")
                if val and val not in parts:
                    parts.append(val)
                if len(parts) >= 1:
                    break
            # Add city
            city = (addr.get("city") or addr.get("town") or
                    addr.get("municipality") or addr.get("county") or "")
            if city and city not in parts:
                parts.append(city)
            name = ", ".join(parts) if parts else data.get("display_name","")[:60]
            if name:
                self.after(0, lambda n=name: [
                    self.loc_name.delete(0,"end"),
                    self.loc_name.insert(0, n),
                    self.loc_status.configure(
                        text=f"✓ GPS + location: {n}",
                        text_color=ACC_GREEN),
                    self._update_loc_preview_now()
                ])
        except Exception:
            self.after(0, lambda: self.loc_status.configure(
                text="✓ GPS acquired — name lookup failed (no internet?)",
                text_color=ACC_AMBER))

    def _preview_map(self):
        """Open location in browser — OpenStreetMap, no API key needed."""
        try:
            lat = float(self.loc_lat.get().strip() or "0")
            lng = float(self.loc_lng.get().strip() or "0")
        except ValueError:
            self.loc_status.configure(
                text="Invalid coordinates.", text_color=ACC_RED); return
        if lat == 0 and lng == 0:
            self.loc_status.configure(
                text="Enter coordinates first.", text_color=ACC_AMBER); return
        import webbrowser
        # zoom=17 shows intersection-level detail
        url = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lng}#map=17/{lat}/{lng}"
        webbrowser.open(url)
        self.loc_status.configure(
            text=f"Opened: {lat:.5f}, {lng:.5f}", text_color=ACC_GREEN)

    def _save(self):
        try:
            # Read with utf-8, fallback to latin-1 if file has legacy encoding
            try:
                with open("config.py", encoding="utf-8") as f: c=f.read()
            except UnicodeDecodeError:
                with open("config.py", encoding="latin-1") as f: c=f.read()
            mn=["bd_vehicles_yolo11.pt","bd_vehicles_best.pt",
                "yolo11n.pt","yolo11s.pt","yolov8n.pt","yolov8s.pt"]
            idx=next((i for i,v in enumerate(self.model_cb.cget("values")) if self.model_cb.get() in v),0)
            for pat,rep in [
                (r'YOLO_MODEL\s*=\s*"[^"]*"',       f'YOLO_MODEL = "{mn[idx]}"'),
                (r'CONFIDENCE\s*=\s*[\d.]+',         f'CONFIDENCE = {self.conf.get():.2f}'),
                (r'PIXELS_PER_METER\s*=\s*\d+',      f'PIXELS_PER_METER = {self.ppm.get()}'),
                (r'VIDEO_FPS\s*=\s*\d+',             f'VIDEO_FPS = {self.fps_e.get()}'),
                (r'SHOW_SPEED\s*=\s*\w+',            f'SHOW_SPEED = {bool(self.sw_sp.get())}'),
                (r'SHOW_IDS\s*=\s*\w+',              f'SHOW_IDS = {bool(self.sw_id.get())}'),
                (r'ENABLE_ZONES\s*=\s*\w+',          f'ENABLE_ZONES = {bool(self.sw_zo.get())}'),
                (r'CPU_PERFORMANCE_MODE\s*=\s*\w+',  f'CPU_PERFORMANCE_MODE = {bool(self.sw_cpu.get())}'),
                (r'USE_DUAL_LINES\s*=\s*\w+',        f'USE_DUAL_LINES = {bool(self.sw_dual.get())}'),
                (r'COUNTING_LINE_POSITION\s*=\s*[\d.]+',f'COUNTING_LINE_POSITION = {self.lpos.get():.2f}'),
                (r'LINE_POS_A\s*=\s*[\d.]+',         f'LINE_POS_A = {self.lpos_a.get():.2f}'),
                (r'LINE_POS_B\s*=\s*[\d.]+',         f'LINE_POS_B = {self.lpos_b.get():.2f}'),
            ]: c=re.sub(pat,rep,c)
            with open("config.py","w",encoding="utf-8") as f: f.write(c)
            save_prefs({
                "author_name": self.name_e.get().strip() or "Nishan",
                "institution": self.inst_e.get().strip() or "SUST · CEE Dept.",
                "loc_name":    self.loc_name.get().strip(),
                "loc_lat":     self.loc_lat.get().strip(),
                "loc_lng":     self.loc_lng.get().strip(),
                "road_type":   self.road_type_cb.get(),
                "speed_limit": self.speed_limit_e.get().strip() or "50",
            })
            from tkinter import messagebox; messagebox.showinfo("Saved ✓","Settings saved ✓")
        except Exception as e:
            from tkinter import messagebox; messagebox.showerror("Error",str(e))


# ================================================================
#  ABOUT PAGE
# ================================================================
class AboutPage(Page):
    def __init__(self, master):
        super().__init__(master)
        self.grid_rowconfigure(1, weight=1)
        self.page_header("ℹ️", "About", "VELOXIS · Product information · NextCity Tessera")

        scroll = ctk.CTkScrollableFrame(self, corner_radius=0)
        scroll.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        # ── Hero ──────────────────────────────────────────────
        hero = ctk.CTkFrame(scroll, corner_radius=16, border_width=1)
        hero.pack(fill="x", padx=32, pady=(20, 0))
        hero.grid_columnconfigure(1, weight=1)

        icon_f = ctk.CTkFrame(hero, width=88, height=88, corner_radius=22)
        icon_f.grid(row=0, column=0, rowspan=3, padx=(28, 20), pady=28)
        icon_f.grid_propagate(False)
        ctk.CTkLabel(icon_f, text="🚦", font=("Segoe UI", 40)
                     ).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(hero, text="VELOXIS",
                     font=("Segoe UI", 34, "bold"),
                     text_color=(ACC_BLUE,"#1d4ed8")).grid(row=0, column=1, sticky="w", pady=(26, 2))
        ctk.CTkLabel(hero, text="AI-Powered Traffic Analysis Platform  ·  Bangladesh Edition",
                     font=("Segoe UI", 13)).grid(row=1, column=1, sticky="w")
        ctk.CTkLabel(hero,
                     text="Version 2.0  ·  YOLOv11 + BoTSORT  ·  2026  ·  MIT License",
                     font=("Segoe UI", 11),
                     text_color="#64748b").grid(row=2, column=1, sticky="w", pady=(2, 26))

        # ── Capability cards ──────────────────────────────────
        cap_f = ctk.CTkFrame(scroll, fg_color="transparent")
        cap_f.pack(fill="x", padx=32, pady=(16, 0))
        cap_f.grid_columnconfigure((0,1,2,3), weight=1)

        caps = [
            ("🎯", "Detection",    "YOLOv11 + BoTSORT\nCustom BD model\n45,862 images",  ACC_BLUE),
            ("🛺", "BD Vehicles",  "Rickshaw · CNG · Car\nMoto · Bus · Truck\nBike · Easybike + 7 more",  ACC_AMBER),
            ("📊", "Analytics",    "PHF · Headway\nSaturation flow · V85\nCSV · Vissim export", ACC_TEAL),
            ("⚡", "Performance",  "CPU & GPU modes\nReal-time HUD\nHomography speed", ACC_GREEN),
        ]
        for i, (icon, title, body, color) in enumerate(caps):
            f = ctk.CTkFrame(cap_f, corner_radius=14, border_width=1)
            f.grid(row=0, column=i, padx=(0 if i==0 else 10, 0), sticky="nsew")
            ctk.CTkFrame(f, height=4, fg_color=color, corner_radius=0).pack(fill="x")
            ctk.CTkLabel(f, text=icon, font=("Segoe UI", 28)).pack(pady=(16,4))
            ctk.CTkLabel(f, text=title, font=("Segoe UI", 12, "bold")).pack()
            ctk.CTkLabel(f, text=body, font=("Segoe UI", 10),
                         text_color="#64748b", justify="center").pack(pady=(4,16))

        # ── Developer + Research ──────────────────────────────
        dev = ctk.CTkFrame(scroll, corner_radius=14, border_width=1)
        dev.pack(fill="x", padx=32, pady=(16, 0))
        dev.grid_columnconfigure((0,1), weight=1)

        # Left: developer
        dev_l = ctk.CTkFrame(dev, fg_color="transparent")
        dev_l.grid(row=0, column=0, padx=(24,12), pady=22, sticky="nsew")

        ctk.CTkLabel(dev_l, text="DEVELOPER",
                     font=("Segoe UI", 9, "bold"),
                     text_color="#64748b").pack(anchor="w", pady=(0,10))

        dev_row = ctk.CTkFrame(dev_l, fg_color="transparent")
        dev_row.pack(fill="x")
        dev_row.grid_columnconfigure(1, weight=1)

        av = ctk.CTkFrame(dev_row, width=52, height=52, corner_radius=26,
                           fg_color=ACC_BLUE)
        av.grid(row=0, column=0, rowspan=2, padx=(0,14))
        av.grid_propagate(False)
        ctk.CTkLabel(av, text="N", font=("Segoe UI", 22, "bold"),
                     text_color="white").place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(dev_row, text="Nishan",
                     font=("Segoe UI", 15, "bold")).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(dev_row,
                     text="B.Sc. Civil & Environmental Engineering\nSUST · 2026",
                     font=("Segoe UI", 11), text_color="#64748b",
                     justify="left").grid(row=1, column=1, sticky="w")

        # Divider
        ctk.CTkFrame(dev, width=1, fg_color="#1f2937"
                     ).grid(row=0, column=0, sticky="nse", padx=0, pady=16)

        # Right: research context
        dev_r = ctk.CTkFrame(dev, fg_color="transparent")
        dev_r.grid(row=0, column=1, padx=(12,24), pady=22, sticky="nsew")

        ctk.CTkLabel(dev_r, text="RESEARCH CONTEXT",
                     font=("Segoe UI", 9, "bold"),
                     text_color="#64748b").pack(anchor="w", pady=(0,10))
        ctk.CTkLabel(dev_r,
                     text="VELOXIS is a research instrument for transportation\n"
                          "engineering in Bangladesh, focusing on non-motorized\n"
                          "transport (NMT), rickshaw dominance, and intersection\n"
                          "capacity analysis on mixed-traffic urban roads.",
                     font=("Segoe UI", 11), text_color="#94a3b8",
                     justify="left").pack(anchor="w")

        # ── NextCity Tessera ──────────────────────────────────
        nct = ctk.CTkFrame(scroll, corner_radius=14, border_width=1)
        nct.pack(fill="x", padx=32, pady=(16, 0))

        nct_inner = ctk.CTkFrame(nct, fg_color="transparent")
        nct_inner.pack(fill="x", padx=24, pady=20)
        nct_inner.grid_columnconfigure(1, weight=1)

        nct_ic = ctk.CTkFrame(nct_inner, width=54, height=54,
                               corner_radius=14, fg_color=ACC_BLUE)
        nct_ic.grid(row=0, column=0, rowspan=2, padx=(0,18))
        nct_ic.grid_propagate(False)
        ctk.CTkLabel(nct_ic, text="🏙", font=("Segoe UI", 26)
                     ).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(nct_inner, text="NextCity Tessera",
                     font=("Segoe UI", 14, "bold"),
                     text_color=(ACC_BLUE,"#1d4ed8")).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(nct_inner,
                     text="VELOXIS is a product of NextCity Tessera — building\n"
                          "intelligent tools for urban mobility, traffic engineering,\n"
                          "and smart city research.",
                     font=("Segoe UI", 11), text_color="#64748b",
                     justify="left").grid(row=1, column=1, sticky="w")

        # ── Tech stack ────────────────────────────────────────
        tech = ctk.CTkFrame(scroll, corner_radius=14, border_width=1)
        tech.pack(fill="x", padx=32, pady=(16, 0))
        ctk.CTkLabel(tech, text="TECHNOLOGY STACK",
                     font=("Segoe UI", 9, "bold"),
                     text_color="#64748b").pack(anchor="w", padx=24, pady=(16,10))

        tech_grid = ctk.CTkFrame(tech, fg_color="transparent")
        tech_grid.pack(fill="x", padx=24, pady=(0,16))
        tech_grid.grid_columnconfigure((0,1,2,3), weight=1)

        tech_items = [
            ("YOLOv11",       "Object Detection",      ACC_BLUE),
            ("BoTSORT",       "Multi-Object Tracking", ACC_TEAL),
            ("OpenCV",        "Computer Vision",       ACC_GREEN),
            ("CustomTkinter", "Desktop UI",            ACC_PURPLE),
            ("Matplotlib",    "Analytics Charts",      ACC_AMBER),
            ("Pandas",        "Data Processing",       "#fb923c"),
            ("Homography",    "Speed Calibration",     ACC_RED),
            ("Flask",         "Web Dashboard",         "#60a5fa"),
        ]
        for i, (name_t, desc_t, col_t) in enumerate(tech_items):
            f = ctk.CTkFrame(tech_grid, corner_radius=10, border_width=1)
            f.grid(row=i//4, column=i%4,
                   padx=(0 if i%4==0 else 8, 0), pady=(0,8), sticky="ew")
            ctk.CTkFrame(f, height=3, fg_color=col_t, corner_radius=0).pack(fill="x")
            ctk.CTkLabel(f, text=name_t,
                         font=("Segoe UI", 11, "bold")).pack(pady=(10,2))
            ctk.CTkLabel(f, text=desc_t, font=("Segoe UI", 9),
                         text_color="#64748b").pack(pady=(0,10))

        # ── License ───────────────────────────────────────────
        lic = ctk.CTkFrame(scroll, corner_radius=14, border_width=1)
        lic.pack(fill="x", padx=32, pady=(16, 32))

        lic_row = ctk.CTkFrame(lic, fg_color="transparent")
        lic_row.pack(fill="x", padx=24, pady=18)
        lic_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(lic_row, text="⚖️", font=("Segoe UI", 28)
                     ).grid(row=0, column=0, rowspan=2, padx=(0,16), sticky="n")
        ctk.CTkLabel(lic_row, text="MIT License",
                     font=("Segoe UI", 13, "bold")).grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(lic_row,
                     text="Copyright © 2026 Nishan, NextCity Tessera. "
                          "Free to use, modify, and distribute under the standard MIT terms. "
                          "This software is provided as-is, without warranty of any kind.",
                     font=("Segoe UI", 11), text_color="#64748b",
                     justify="left", wraplength=680
                     ).grid(row=1, column=1, sticky="w", pady=(4,0))


# ================================================================
#  MAIN WINDOW
# ================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        p=load_prefs()
        name=p.get("author_name","Nishan")
        inst=p.get("institution","SUST")
        self.title(f"VELOXIS  ·  {name}, {inst}  ·  NextCity Tessera")
        self.geometry("1340x840"); self.minsize(1100,680)
        self.grid_columnconfigure(1,weight=1); self.grid_rowconfigure(0,weight=1)

        # ── Sidebar ───────────────────────────────────────────
        sb = ctk.CTkFrame(self, width=248, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew"); sb.grid_propagate(False)
        sb.grid_rowconfigure(12, weight=1); sb.grid_columnconfigure(0, weight=1)

        # Logo block — thick accent bar + icon
        lf = ctk.CTkFrame(sb, corner_radius=0, fg_color="transparent")
        lf.grid(row=0, column=0, sticky="ew")
        ctk.CTkFrame(lf, height=4, corner_radius=0,
                     fg_color=ACC_BLUE).pack(fill="x")
        li = ctk.CTkFrame(lf, fg_color="transparent")
        li.pack(padx=16, pady=(16, 16), fill="x")
        ic = ctk.CTkFrame(li, width=48, height=48, corner_radius=14,
                          fg_color=(ACC_BLUE, "#1a3a6e"))
        ic.pack(side="left", padx=(0, 14)); ic.pack_propagate(False)
        ctk.CTkLabel(ic, text="🚦", font=("Segoe UI", 22)
                    ).place(relx=0.5, rely=0.5, anchor="center")
        tx = ctk.CTkFrame(li, fg_color="transparent"); tx.pack(side="left", fill="y")
        ctk.CTkLabel(tx, text="VELOXIS",
                     font=("Segoe UI", 17, "bold"),
                     text_color=(ACC_BLUE, "#6ba4ff")).pack(anchor="w")
        ctk.CTkLabel(tx, text="NextCity Tessera  ·  v2.0",
                     font=("Segoe UI", 9), text_color="#4f5e78").pack(anchor="w")
        # Separator
        ctk.CTkFrame(sb, height=1, corner_radius=0
                    ).grid(row=1, column=0, sticky="ew")

        SLabel(sb, "Navigation").grid(row=2, column=0, padx=18, pady=(12, 4), sticky="w")

        nav = [("🏠", "Home"), ("📹", "Live Detection"), ("🎬", "File Detection"),
               ("🗺", "Lane Drawing"), ("📐", "Calibrate Speed"), ("📊", "Analytics"),
               ("⚙️", "Settings"), ("ℹ️", "About")]
        self.nav_btns = []
        for i, (icon, lbl) in enumerate(nav):
            btn = NavBtn(sb, icon, lbl, lambda idx=i: self._switch(idx))
            btn.grid(row=3 + i, column=0, padx=8, pady=1, sticky="ew")
            self.nav_btns.append(btn)

        ctk.CTkFrame(sb, height=1, corner_radius=0
                    ).grid(row=12, column=0, sticky="ew", padx=0, pady=(4, 0))

        # Refresh button
        ctk.CTkButton(sb, text="🔄  Refresh App", height=36,
            fg_color="transparent", border_width=1,
            border_color=("#c7d6f0", "#2a3a55"),
            font=("Segoe UI", 11), corner_radius=10,
            command=self._refresh
        ).grid(row=13,column=0,padx=12,pady=(8,4),sticky="ew")

        # Theme toggle
        tr=ctk.CTkFrame(sb,fg_color="transparent")
        tr.grid(row=14,column=0,padx=14,pady=(0,6),sticky="ew")
        ctk.CTkLabel(tr,text="☀️",font=("Segoe UI",13)).pack(side="left",padx=(4,6))
        self.theme_sw=ctk.CTkSwitch(tr,text="Light mode",
            button_color=ACC_AMBER,progress_color=ACC_AMBER,
            width=44,height=22,command=self._toggle_theme,
            font=("Segoe UI",11))
        self.theme_sw.pack(side="left")
        if _THEME=="light": self.theme_sw.select()

        # Author block — avatar + info
        af = ctk.CTkFrame(sb, fg_color="transparent", corner_radius=10)
        af.grid(row=15, column=0, padx=12, pady=(0, 18), sticky="ew")
        ctk.CTkFrame(af, height=1, corner_radius=0
                    ).pack(fill="x", pady=(0, 10))
        av_row = ctk.CTkFrame(af, fg_color="transparent")
        av_row.pack(fill="x", padx=6)
        # Avatar circle
        av = ctk.CTkFrame(av_row, width=34, height=34, corner_radius=17,
                          fg_color=(ACC_BLUE, "#1a3a6e"))
        av.pack(side="left", padx=(0, 10)); av.pack_propagate(False)
        ctk.CTkLabel(av, text=name[0].upper() if name else "N",
                     font=("Segoe UI", 14, "bold"),
                     text_color="white").place(relx=0.5, rely=0.5, anchor="center")
        txt = ctk.CTkFrame(av_row, fg_color="transparent")
        txt.pack(side="left", fill="y")
        ctk.CTkLabel(txt, text=name, font=("Segoe UI", 11, "bold"),
                     text_color=(ACC_BLUE, "#6ba4ff")).pack(anchor="w")
        ctk.CTkLabel(txt, text=p.get("institution", "SUST · CEE Dept."),
                     font=("Segoe UI", 9), text_color="#64748b").pack(anchor="w")
        ctk.CTkLabel(af, text="© 2026 NextCity Tessera",
                     font=("Segoe UI", 9),
                     text_color=(ACC_TEAL, "#2fe6d4")).pack(anchor="w", padx=6, pady=(6, 4))

        # Status bar
        self.sb2=StatusBar(self)
        self.grid_rowconfigure(1,weight=0)
        self.sb2.grid(row=1,column=0,columnspan=2,sticky="ew")

        # Content
        content=ctk.CTkFrame(self,corner_radius=0)
        content.grid(row=0,column=1,sticky="nsew")
        content.grid_columnconfigure(0,weight=1); content.grid_rowconfigure(0,weight=1)

        self.hp=HomePage(content)
        self.lp=LivePage(content,status_bar=self.sb2,home_page=self.hp)
        self.fp=FilePage(content,status_bar=self.sb2,home_page=self.hp)
        self.la=LanePage(content)
        self.cal=CalibratePage(content,status_bar=self.sb2)
        self.dp=DashboardPage(content)
        self.sp=SettingsPage(content)
        self.ab=AboutPage(content)
        self._pages=[self.hp,self.lp,self.fp,self.la,self.cal,self.dp,self.sp,self.ab]
        for pg in self._pages: pg.grid(row=0,column=0,sticky="nsew")
        self._switch(0)

    def _refresh(self):
        """Reload home stats, dashboard and settings from disk."""
        try: self.hp._load_stats()
        except: pass
        try: self.dp._ever_shown=False; self.dp.refresh()
        except: pass
        try: self.sp._load()
        except: pass
        self.sb2.set("App refreshed ✓","idle")

    def _switch(self,idx):
        self._pages[idx].tkraise()
        for i,b in enumerate(self.nav_btns): b.set_active(i==idx)
        self.sb2.set(["Home","Live Detection","File Detection",
                      "Lane Drawing","Calibrate Speed","Analytics","Settings","About"][idx],"idle")
        if idx==5 and not self.dp._ever_shown: self.dp.refresh()

    def _toggle_theme(self):
        global _THEME
        _THEME="light" if _THEME=="dark" else "dark"
        ctk.set_appearance_mode(_THEME)
        save_prefs({"theme":_THEME})


# ── Entry point ────────────────────────────────────────────────
if __name__=="__main__":
    for d in ["videos","data","data/snapshots"]: os.makedirs(d,exist_ok=True)
    App().mainloop()
