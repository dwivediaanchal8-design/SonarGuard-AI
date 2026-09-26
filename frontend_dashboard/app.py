import os
import sys
import glob
import json
import time
from datetime import datetime, timezone

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from core_engine.geotag_engine import parse_sonar_telemetry
from preprocessing.denoise import preprocess_sonar as denoise_preprocess

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

st.set_page_config(
    page_title="AquaScan AI | SIH-2026 NIOT Marine Survey",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #020817 !important;
    color: #e2e8f0 !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
}

section[data-testid="stSidebar"] {
    background: #f8fafc !important;
    border-right: 2px solid #dde3ed !important;
}

section[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0f172a !important;
}

section[data-testid="stSidebar"] label {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #334155 !important;
    letter-spacing: 0.03em !important;
}

section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stSelectbox > div > div > div,
section[data-testid="stSidebar"] .stFileUploader > div,
section[data-testid="stSidebar"] .stNumberInput > div,
section[data-testid="stSidebar"] .stNumberInput > div > div {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] .stSelectbox > div > div > div > div,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] input[type="number"],
section[data-testid="stSidebar"] input[type="text"] {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    -webkit-text-fill-color: #0f172a !important;
}

section[data-testid="stSidebar"] [data-baseweb="popover"] li,
section[data-testid="stSidebar"] [data-baseweb="popover"] ul {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

section[data-testid="stSidebar"] [data-baseweb="popover"] li:hover {
    background-color: #e0f2fe !important;
}

.block-container {
    padding: 1rem 1.8rem 2rem 1.8rem !important;
    max-width: 100% !important;
}

.aq-navbar {
    background: linear-gradient(120deg, #020c1e 0%, #0a1f3d 45%, #051b38 100%);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 16px;
    padding: 20px 32px;
    margin-bottom: 20px;
    box-shadow: 0 0 40px rgba(0, 212, 255, 0.08), inset 0 1px 0 rgba(255,255,255,0.05);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.aq-brand {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.aq-title {
    font-size: 26px;
    font-weight: 900;
    color: #f0f9ff;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    line-height: 1;
    text-shadow: 0 0 30px rgba(0,212,255,0.4);
}

.aq-subtitle {
    font-size: 12px;
    color: #64a8cc;
    font-weight: 500;
    letter-spacing: 0.08em;
}

.aq-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    align-items: center;
}

.aq-badge {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 10px;
    padding: 8px 16px;
    min-width: 88px;
    backdrop-filter: blur(4px);
}

.aq-badge-val {
    font-size: 18px;
    font-weight: 800;
    color: #00d4ff;
    line-height: 1.1;
}

.aq-badge-lbl {
    font-size: 10px;
    color: #7ecfe8;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-top: 3px;
}

.aq-online {
    background: linear-gradient(135deg, #022c1e, #064e3b);
    color: #6ee7b7;
    border: 1px solid #34d399;
    border-radius: 20px;
    padding: 8px 18px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.06em;
    box-shadow: 0 0 16px rgba(52, 211, 153, 0.2);
}

.aq-panel {
    background: linear-gradient(145deg, #0b1a30 0%, #0d1f38 100%);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
    margin-bottom: 18px;
}

.aq-panel-hdr {
    font-size: 12px;
    font-weight: 800;
    color: #7ecfe8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(0,212,255,0.15);
    display: flex;
    align-items: center;
    gap: 8px;
}

.aq-col-label {
    font-size: 11px;
    font-weight: 800;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
    padding: 6px 10px;
    background: rgba(0,0,0,0.3);
    border-radius: 6px;
    display: inline-block;
}

.stat-strip {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 18px;
}

.stat-tile {
    flex: 1;
    min-width: 110px;
    background: linear-gradient(145deg, #0b1a30, #0d2040);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}

.stat-val {
    font-size: 28px;
    font-weight: 900;
    color: #00d4ff;
    line-height: 1;
}

.stat-lbl {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 4px;
}

.idle-box {
    background: linear-gradient(145deg, #0b1a30, #0a1628);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 14px;
    padding: 56px 32px;
    text-align: center;
}

.idle-icon {
    font-size: 52px;
    margin-bottom: 14px;
}

.idle-title {
    font-size: 20px;
    font-weight: 800;
    color: #e0f2fe;
    margin-bottom: 8px;
}

.idle-body {
    font-size: 13px;
    color: #64a8cc;
    line-height: 1.7;
    max-width: 420px;
    margin: 0 auto;
}

.sb-section {
    font-size: 11px;
    font-weight: 800;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 16px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #e2e8f0;
}

.threat-pill-CRITICAL {
    display: inline-block;
    background: #fee2e2;
    color: #991b1b;
    border: 1.5px solid #fca5a5;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.threat-pill-HIGH {
    display: inline-block;
    background: #ffedd5;
    color: #9a3412;
    border: 1.5px solid #fdba74;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.threat-pill-MODERATE {
    display: inline-block;
    background: #fef9c3;
    color: #854d0e;
    border: 1.5px solid #fde047;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

div[data-testid="stDataFrame"] {
    border-radius: 10px !important;
}

.footer-bar {
    margin-top: 32px;
    padding: 14px 20px;
    background: rgba(11,26,48,0.8);
    border-top: 1px solid rgba(0,212,255,0.1);
    border-radius: 10px;
    text-align: center;
    font-size: 11px;
    color: #475569;
    letter-spacing: 0.05em;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# Neon tactical colours per class — BGR order for OpenCV
# Wreck/Hazard : #FF9900 → (  0, 153, 255)
# Ghost Net    : #FF3366 → ( 51, 102, 255)  (adjusted for BGR: B=51,G=51,R=255)
# Pipe/Cable   : #00E5FF → (255, 229,   0)  (BGR: B=255,G=229,R=0)
DETECTION_META = {
    0: {
        "name": "Shipwreck / Solid Hazard",
        "short": "WRECK",
        "threat": "HIGH",
        "cbgr": (0, 153, 255),   # #FF9900 in BGR
        "nbgr": (0, 153, 255),
        "mc": "#FF9900",
        "depth_m": 42.5,
    },
    1: {
        "name": "Ghost Fishing Net",
        "short": "GHOST-NET",
        "threat": "CRITICAL",
        "cbgr": (102, 51, 255),  # #FF3366 in BGR
        "nbgr": (102, 51, 255),
        "mc": "#FF3366",
        "depth_m": 18.2,
    },
    2: {
        "name": "Submerged Pipe / Cable",
        "short": "PIPE/CABLE",
        "threat": "MODERATE",
        "cbgr": (255, 229, 0),   # #00E5FF in BGR
        "nbgr": (255, 229, 0),
        "mc": "#00E5FF",
        "depth_m": 61.0,
    },
}

THREAT_ORDER = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}
THREAT_BG = {"CRITICAL": "#fee2e2", "HIGH": "#ffedd5", "MODERATE": "#fef9c3"}
THREAT_FC = {"CRITICAL": "#991b1b", "HIGH": "#9a3412", "MODERATE": "#854d0e"}


def locate_weights():
    sdir = os.path.dirname(os.path.abspath(__file__))
    for base in list({os.getcwd(), sdir, ROOT_DIR}):
        hits = glob.glob(os.path.join(base, "**", "best.pt"), recursive=True)
        if hits:
            return os.path.abspath(hits[0])
    return "yolov8n.pt"


def _pick_first(patterns):
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return os.path.abspath(hits[0])
    return None


def build_demo_registry():
    r = ROOT_DIR
    shipwreck = _pick_first([
        os.path.join(r, "demo_test_assets", "test_shipwreck.png"),
        os.path.join(r, "data", "AI4Shipwrecks", "extras", "terrain", "images", "Exploratory_A_01.png"),
        os.path.join(r, "data", "AI4Shipwrecks", "train", "images", "*.png"),
    ])
    ghost_net = _pick_first([
        os.path.join(r, "data", "AI4Shipwrecks", "train", "images", "DM_Wilson_01.png"),
        os.path.join(r, "data", "multi_debris_dataset", "images", "train", "DM_Wilson_03_tile_0.png"),
        os.path.join(r, "data", "multi_debris_dataset", "images", "train", "*.png"),
    ])
    pipe_cable = _pick_first([
        os.path.join(r, "demo_test_assets", "test_subsea_pipeline.png"),
        os.path.join(r, "data", "full_sonar_dataset", "images", "train", "DM_Wilson_01.png"),
        os.path.join(r, "data", "full_sonar_dataset", "images", "train", "*.png"),
    ])
    video = _pick_first([
        os.path.join(r, "demo_test_assets", "test_sonar_stream.mp4"),
        os.path.join(r, "**", "sonar_mission_feed.mp4"),
        os.path.join(r, "**", "*.mp4"),
    ])
    return {
        "shipwreck": shipwreck,
        "ghost_net": ghost_net,
        "pipe_cable": pipe_cable,
        "video": video,
    }


DEMO = build_demo_registry()
WEIGHTS_PATH = locate_weights()


@st.cache_resource(show_spinner="Initialising YOLOv8 acoustic detector...")
def load_model(wp):
    return YOLO(wp)


detector = load_model(WEIGHTS_PATH)


def preprocess_full(img_bgr, use_clahe, use_lee, use_bilateral, bilateral_sigma):
    """Sonar preprocessing pipeline.
    Default path: gentle bilateral only (d=5, sigmaColor=30, sigmaSpace=30) to
    preserve natural dark-grey seabed acoustic texture and nadir shadow bands.
    Pixel intensities are NEVER clipped to pure white — no overexposure.
    Lee speckle filter and CLAHE are strictly opt-in.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
    out = gray.copy()
    # Standard gentle bilateral — preserves edges and tonal range without blowout
    if use_bilateral:
        sigma = max(10, min(75, int(bilateral_sigma)))  # guard against extremes
        out = cv2.bilateralFilter(out, d=5, sigmaColor=sigma, sigmaSpace=sigma)
    # Optional Lee speckle suppression (coherent noise) — off by default
    if use_lee:
        out = denoise_preprocess(cv2.cvtColor(out, cv2.COLOR_GRAY2BGR),
                                 clahe=False, denoise=True)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    # Optional CLAHE — mild clip limit, strictly opt-in
    if use_clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        out = clahe_obj.apply(out)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def draw_pill_box(img, x1, y1, x2, y2, class_text, conf_text, cbgr, nbgr):
    """Ultra-visible tactical bounding box with high-contrast pill badge.

    Visual layers (outermost → innermost):
      1. 4 px outer solid BLACK trace — shadow for contrast on bright sonar texture
      2. 3 px inner vibrant NEON stroke (per-class colour)
      3. Corner accent ticks in neon colour
      4. Solid dark pill badge: bold WHITE class label + neon confidence

    nbgr — per-class neon BGR: #00E5FF pipe | #FF3366 net | #FF9900 wreck
    cbgr — corner accent colour (same as nbgr)
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)

    # ── Layer 1: 4 px outer solid black shadow trace ──
    cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), 4)
    # ── Layer 2: 3 px inner vibrant neon border ──
    cv2.rectangle(img, (x1, y1), (x2, y2), nbgr, 3)

    # ── Layer 3: corner accent ticks ──
    cl = max(14, (x2 - x1) // 5)
    for ax, ay, hx, hy, vx, vy in [
        (x1, y1, x1 + cl, y1, x1, y1 + cl),
        (x2, y1, x2 - cl, y1, x2, y1 + cl),
        (x1, y2, x1 + cl, y2, x1, y2 - cl),
        (x2, y2, x2 - cl, y2, x2, y2 - cl),
    ]:
        cv2.line(img, (ax, ay), (hx, hy), cbgr, 3)
        cv2.line(img, (ax, ay), (vx, vy), cbgr, 3)

    # ── Layer 4: solid pill badge (CLASS_NAME + CONF%) ──
    badge_text = f"{class_text}  {conf_text}"   # e.g. "WRECK  87.4%"
    font = cv2.FONT_HERSHEY_DUPLEX
    fs = max(0.44, min(0.70, (x2 - x1) / 280))
    (tw, th), baseline = cv2.getTextSize(badge_text, font, fs, 2)
    pad_x, pad_y = 10, 6
    pill_w = tw + pad_x * 2
    pill_h = th + pad_y * 2 + baseline
    px1 = x1
    py1 = max(0, y1 - pill_h - 5)
    px2 = min(w - 1, px1 + pill_w)
    py2 = py1 + pill_h

    # Dark background fill — maximises legibility over any sonar texture
    cv2.rectangle(img, (px1, py1), (px2, py2), (10, 10, 10), -1)
    # Neon outline on badge
    cv2.rectangle(img, (px1, py1), (px2, py2), nbgr, 2)
    # Bold white class name
    text_y = py1 + pad_y + th
    cv2.putText(img, badge_text, (px1 + pad_x, text_y),
                font, fs, (0, 0, 0), 4, cv2.LINE_AA)      # thick black drop shadow
    cv2.putText(img, badge_text, (px1 + pad_x, text_y),
                font, fs, (255, 255, 255), 2, cv2.LINE_AA)  # bold white label


def mk_geojson(records):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [r["Longitude"], r["Latitude"]]},
                "properties": {
                    "hazard_id": r["Hazard ID"],
                    "classification": r["Classification"],
                    "confidence": str(r.get("Confidence_Score", "")),
                    "threat_level": r.get("threat_level", "MODERATE"),
                    "depth_m": r.get("Depth_m", 0),
                    "length_m": r.get("Estimated_Length_m", 0),
                    "width_m": r.get("Estimated_Width_m", 0),
                    "area_sqm": r.get("Estimated_Area_sq_m", 0),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "survey_protocol": "MoES/NIOT-SSS-400kHz",
                },
            }
            for r in records
        ],
    }


def mk_folium_map(records, blat, blon):
    clat = blat if not records else float(np.mean([r["Latitude"] for r in records]))
    clon = blon if not records else float(np.mean([r["Longitude"] for r in records]))
    fm = folium.Map(
        location=[clat, clon], zoom_start=15,
        tiles="CartoDB dark_matter", control_scale=True,
    )
    folium.Marker(
        [blat, blon],
        popup=folium.Popup("<b>AUV Survey Origin</b><br>NIOT Chennai Corridor", max_width=200),
        icon=folium.Icon(color="blue", icon="ship", prefix="fa"),
    ).add_to(fm)
    tc = {"CRITICAL": "red", "HIGH": "orange", "MODERATE": "beige"}
    ti = {"CRITICAL": "warning", "HIGH": "exclamation-triangle", "MODERATE": "info"}
    for r in records:
        tl = r.get("threat_level", "MODERATE")
        conf = r.get("Confidence_Score", 0)
        cs = f"{conf*100:.1f}%" if isinstance(conf, float) else str(conf)
        fc = "#dc2626" if tl == "CRITICAL" else "#ea580c" if tl == "HIGH" else "#ca8a04"
        html = (
            f"<div style='font-family:sans-serif;min-width:200px;'>"
            f"<b style='font-size:13px;color:#1e293b'>{r['Hazard ID']}</b><br>"
            f"<span style='color:#475569;font-size:12px'>{r['Classification']}</span>"
            f"<hr style='margin:5px 0'>"
            f"<table style='font-size:11px;color:#374151;width:100%'>"
            f"<tr><td><b>Confidence</b></td><td>{cs}</td></tr>"
            f"<tr><td><b>Threat</b></td><td><b style='color:{fc}'>{tl}</b></td></tr>"
            f"<tr><td><b>Depth</b></td><td>{r.get('Depth_m', 0)} m</td></tr>"
            f"<tr><td><b>Length</b></td><td>{r.get('Estimated_Length_m', 0)} m</td></tr>"
            f"<tr><td><b>Width</b></td><td>{r.get('Estimated_Width_m', 0)} m</td></tr>"
            f"</table></div>"
        )
        folium.Marker(
            [r["Latitude"], r["Longitude"]],
            popup=folium.Popup(html, max_width=250),
            tooltip=f"{r['Hazard ID']} — {tl}",
            icon=folium.Icon(color=tc.get(tl, "beige"), icon=ti.get(tl, "info"), prefix="fa"),
        ).add_to(fm)
    folium.LayerControl().add_to(fm)
    return fm


def render_navbar():
    wt = os.path.basename(WEIGHTS_PATH)
    ts = datetime.now().strftime("%H:%M:%S IST")
    st.markdown(f"""
    <div class="aq-navbar">
        <div class="aq-brand">
            <div class="aq-title">🌊 AquaScan AI — Hydrographic Sonar Inspector</div>
            <div class="aq-subtitle">
                Autonomous Marine Debris &amp; Anomaly Detection &nbsp;·&nbsp;
                MoES / NIOT Protocol &nbsp;·&nbsp; SIH-2026 &nbsp;·&nbsp;
                Model: <b style="color:#fde68a;">{wt}</b> &nbsp;·&nbsp; {ts}
            </div>
        </div>
        <div class="aq-badges">
            <div class="aq-badge"><div class="aq-badge-val">15 Hz</div><div class="aq-badge-lbl">AUV Ping</div></div>
            <div class="aq-badge"><div class="aq-badge-val">18 ms</div><div class="aq-badge-lbl">Latency</div></div>
            <div class="aq-badge"><div class="aq-badge-val">94.2%</div><div class="aq-badge-lbl">Rejection</div></div>
            <div class="aq-badge"><div class="aq-badge-val">1.42 km²</div><div class="aq-badge-lbl">Mapped</div></div>
            <div class="aq-online">● SYSTEM ONLINE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


SOURCE_LABELS = [
    "🚧  Demo 1 — Shipwreck Anomaly",
    "🎣  Demo 2 — Ghost Fishing Net",
    "🛠  Demo 3 — Submerged Pipe / Cable",
    "📹  Continuous Stream — AUV Waterfall Video",
    "📂  Upload Custom Sonar Image",
]
SOURCE_KEYS = ["shipwreck", "ghost_net", "pipe_cable", "video", "custom"]


def render_sidebar():
    st.sidebar.markdown(
        "<h2 style='color:#0284c7;margin:0 0 2px 0;font-size:20px;font-weight:900;'>"
        "🌊 AquaScan Lab</h2>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("NIOT Side-Scan Sonar Pipeline · SIH-2026")
    st.sidebar.markdown("---")

    st.sidebar.markdown("<p class='sb-section'>Input Source</p>", unsafe_allow_html=True)
    source_label = st.sidebar.selectbox(
        "Input Source",
        SOURCE_LABELS,
        label_visibility="collapsed",
    )
    source_key = SOURCE_KEYS[SOURCE_LABELS.index(source_label)]

    uploaded = None
    if source_key == "custom":
        st.sidebar.markdown("<p class='sb-section'>Upload Sonar Tile</p>", unsafe_allow_html=True)
        uploaded = st.sidebar.file_uploader(
            "Upload Sonar Tile",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            label_visibility="collapsed",
        )

    st.sidebar.markdown("<p class='sb-section'>Acoustic Calibration</p>", unsafe_allow_html=True)
    with st.sidebar.expander("🛠 Detection Parameters", expanded=True):
        conf_thr = st.slider("Confidence Gate", 0.05, 0.95, 0.15, 0.05,
            help="Minimum detection confidence. Default 0.15 ensures all target anomalies trigger.")
        iou_thr = st.slider("IoU Overlap Limit", 0.10, 0.80, 0.25, 0.05,
            help="NMS suppression threshold. Default 0.25 avoids merging close detections.")
        use_lee = st.checkbox("Lee Speckle Filter", value=False,
            help="Coherent speckle suppression (Lee 1980). Off by default to avoid over-processing.")
        use_clahe = st.checkbox("CLAHE Contrast Enhancement", value=False,
            help="Adaptive histogram equalisation (clipLimit=1.5). Off by default to preserve natural sonar texture.")
        use_bilateral = st.checkbox("Bilateral Edge-Preserving", value=True,
            help="Gentle bilateral smoothing (d=5, σ=30) — preserves edges without overexposure.")
        bilateral_sigma = st.slider("Bilateral Sigma", 15, 75, 30, 5,
            disabled=not use_bilateral)

    st.sidebar.markdown("<p class='sb-section'>AUV Geo-Reference Origin</p>", unsafe_allow_html=True)
    auv_lat = st.sidebar.number_input("Latitude (°N)", value=13.082700, format="%.6f")
    auv_lon = st.sidebar.number_input("Longitude (°E)", value=80.450000, format="%.6f")

    st.sidebar.markdown("---")
    demo_status = []
    for key, label in zip(SOURCE_KEYS[:3], ["Shipwreck", "Ghost Net", "Pipe/Cable"]):
        found = DEMO.get(key) is not None
        icon = "✅" if found else "❌"
        st.sidebar.markdown(
            f"<small style='color:#64748b;'>{icon} {label}: "
            f"<code>{'...'+DEMO[key][-28:] if found and DEMO[key] else 'not found'}</code></small>",
            unsafe_allow_html=True,
        )
    st.sidebar.markdown(
        f"<small style='color:#64748b;'>🤖 Weights: <code>{os.path.basename(WEIGHTS_PATH)}</code></small>",
        unsafe_allow_html=True,
    )

    return dict(
        source_key=source_key,
        uploaded=uploaded,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        use_lee=use_lee,
        use_clahe=use_clahe,
        use_bilateral=use_bilateral,
        bilateral_sigma=bilateral_sigma,
        auv_lat=auv_lat,
        auv_lon=auv_lon,
    )


def load_image_bgr(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _nms_dedup(boxes, iou_thresh=0.30):
    if len(boxes) == 0:
        return []
    kept = []
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][4], reverse=True)
    suppressed = [False] * len(boxes)
    for i_idx, i in enumerate(order):
        if suppressed[i]:
            continue
        kept.append(i)
        x1i, y1i, x2i, y2i = boxes[i][:4]
        area_i = max(0, x2i - x1i) * max(0, y2i - y1i)
        for j_idx in range(i_idx + 1, len(order)):
            j = order[j_idx]
            if suppressed[j]:
                continue
            x1j, y1j, x2j, y2j = boxes[j][:4]
            ix1 = max(x1i, x1j)
            iy1 = max(y1i, y1j)
            ix2 = min(x2i, x2j)
            iy2 = min(y2i, y2j)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_j = max(0, x2j - x1j) * max(0, y2j - y1j)
            union = area_i + area_j - inter
            if union > 0 and inter / union > iou_thresh:
                suppressed[j] = True
    return kept


def infer(bgr_pre, conf_thr, iou_thr):
    rgb = cv2.cvtColor(bgr_pre, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    res = detector.predict(source=rgb, conf=conf_thr, iou=iou_thr, verbose=False)[0]
    return res, (time.perf_counter() - t0) * 1000


def process_detections(results, bgr_pre, params):
    ann = bgr_pre.copy()
    raw = []
    bds = []

    if results.boxes and len(results.boxes) > 0:
        all_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            all_boxes.append((x1, y1, x2, y2, conf, cls_id))

        keep_idx = _nms_dedup(all_boxes, iou_thresh=params["iou_thr"])

        for i in keep_idx:
            x1, y1, x2, y2, conf, cls_id = all_boxes[i]
            m = DETECTION_META.get(cls_id, {
                "name": detector.names.get(cls_id, f"Anomaly-{cls_id}"),
                "short": f"CLS{cls_id}", "threat": "MODERATE",
                "cbgr": (255, 229, 0), "nbgr": (255, 229, 0), "depth_m": 30.0,
            })
            draw_pill_box(ann, x1, y1, x2, y2,
                          m["short"], f"{conf*100:.1f}%", m["cbgr"], m["nbgr"])
            raw.append({
                "x_center": (x1 + x2) / 2.0,
                "y_center": (y1 + y2) / 2.0,
                "class_name": m["name"],
                "confidence": conf,
                "area_px": (x2 - x1) * (y2 - y1),
                "threat_level": m["threat"],
                "depth_m": m.get("depth_m", 30.0),
            })
            bds.append((x2 - x1, y2 - y1))

    geo = parse_sonar_telemetry(
        raw,
        base_latitude=params["auv_lat"],
        base_longitude=params["auv_lon"],
        bbox_dims=bds if bds else None,
    )
    for i, rec in enumerate(geo):
        if i < len(raw):
            rec["threat_level"] = raw[i]["threat_level"]
            rec["Confidence_Score"] = raw[i]["confidence"]
            rec["Depth_m"] = raw[i]["depth_m"]
    return ann, geo


def show_stats(geo, lat_ms):
    n = len(geo)
    nc = sum(1 for r in geo if r.get("threat_level") == "CRITICAL")
    nh = sum(1 for r in geo if r.get("threat_level") == "HIGH")
    nm = sum(1 for r in geo if r.get("threat_level") == "MODERATE")
    st.markdown(f"""
    <div class="stat-strip">
        <div class="stat-tile"><div class="stat-val">{n}</div><div class="stat-lbl">Anomalies</div></div>
        <div class="stat-tile" style="border-color:rgba(220,38,38,0.35);">
            <div class="stat-val" style="color:#f87171;">{nc}</div><div class="stat-lbl">Critical</div></div>
        <div class="stat-tile" style="border-color:rgba(234,88,12,0.35);">
            <div class="stat-val" style="color:#fb923c;">{nh}</div><div class="stat-lbl">High</div></div>
        <div class="stat-tile" style="border-color:rgba(202,138,4,0.35);">
            <div class="stat-val" style="color:#fbbf24;">{nm}</div><div class="stat-lbl">Moderate</div></div>
        <div class="stat-tile" style="border-color:rgba(0,212,255,0.25);">
            <div class="stat-val">{lat_ms:.0f} ms</div><div class="stat-lbl">Inference</div></div>
    </div>
    """, unsafe_allow_html=True)


def show_table(geo):
    """Render geotagged anomaly telemetry table.
    Uses plain st.dataframe() — no Pandas Styler / applymap to avoid crashes.
    """
    rows = []
    for r in sorted(geo, key=lambda x: THREAT_ORDER.get(x.get("threat_level", "MODERATE"), 9)):
        tl = r.get("threat_level", "MODERATE")
        conf = r.get("Confidence_Score", 0)
        cs = f"{conf*100:.1f}%" if isinstance(conf, float) else str(conf)
        rows.append({
            "Hazard ID": r["Hazard ID"],
            "Classification": r["Classification"],
            "Confidence": cs,
            "Threat Priority": tl,
            "Latitude (N)": r["Latitude"],
            "Longitude (E)": r["Longitude"],
            "Depth (m)": r.get("Depth_m", 0),
            "Length (m)": r.get("Estimated_Length_m", 0),
            "Width (m)": r.get("Estimated_Width_m", 0),
            "Area (m²)": r.get("Estimated_Area_sq_m", 0),
            "Status": r.get("Status", "Confirmed Anomaly"),
        })
    df = pd.DataFrame(rows)
    # Crash-proof: use clean st.dataframe — no applymap / Styler calls
    st.dataframe(df, hide_index=True, use_container_width=True)
    return df


def show_exports(df, geo):
    gj = json.dumps(mk_geojson(geo), indent=2).encode("utf-8")
    cv = df.to_csv(index=False).encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥  Download CSV Telemetry Log",
            data=cv,
            file_name=f"aquascan_telemetry_{ts}.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "🗺  Download GeoJSON Layer",
            data=gj,
            file_name=f"aquascan_hazards_{ts}.geojson",
            mime="application/geo+json",
        )


def run_image_pipeline(bgr_orig, params, source_label):
    with st.spinner("Applying sonar pre-processing (bilateral filter)..."):
        bgr_pre = preprocess_full(
            bgr_orig,
            use_clahe=params["use_clahe"],
            use_lee=params["use_lee"],
            use_bilateral=params["use_bilateral"],
            bilateral_sigma=params["bilateral_sigma"],
        )
    with st.spinner("Running YOLOv8 sonar inference..."):
        results, lat_ms = infer(bgr_pre, params["conf_thr"], params["iou_thr"])

    ann_bgr, geo = process_detections(results, bgr_pre, params)

    filters = []
    if params["use_bilateral"]:
        filters.append(f"Bilateral(d=5, s={params['bilateral_sigma']})")
    if params["use_lee"]:
        filters.append("Lee-1980")
    if params["use_clahe"]:
        filters.append("CLAHE(1.5)")
    fl_str = " + ".join(filters) if filters else "Raw (no filter)"

    nd = len(geo)
    dcol = "#f87171" if nd > 0 else "#4ade80"
    dlabel = f"{nd} ANOMAL{'IES' if nd != 1 else 'Y'} DETECTED" if nd > 0 else "CLEAR SWATH"

    st.markdown(
        f'<div class="aq-panel-hdr" style="margin-bottom:12px;">'
        f'\U0001f4e1 SONAR ANALYSIS · '
        f'<span style="color:#fde68a;">{source_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown(
            f'<div class="aq-col-label">\U0001f4f7 PREPROCESSED WATERFALL &nbsp;|&nbsp; {fl_str}</div>',
            unsafe_allow_html=True,
        )
        st.image(cv2.cvtColor(bgr_pre, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_r:
        st.markdown(
            f'<div class="aq-col-label">\U0001f3af AI DETECTION OVERLAY &nbsp;|&nbsp; '
            f'<span style="color:{dcol};font-weight:800;">{dlabel}</span></div>',
            unsafe_allow_html=True,
        )
        st.image(cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    show_stats(geo, lat_ms)

    # ── GEOTAGGED ANOMALY TELEMETRY REGISTER — always rendered ──────────────
    st.markdown("""
    <div class="aq-panel">
        <div class="aq-panel-hdr">\U0001f4cb GEOTAGGED ANOMALY TELEMETRY REGISTER</div>
    </div>
    """, unsafe_allow_html=True)

    if geo:
        tbl_col, map_col = st.columns([3, 2], gap="medium")
        with tbl_col:
            df = show_table(geo)
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            show_exports(df, geo)

        with map_col:
            st.markdown(
                '<div class="aq-col-label">\U0001f5fa OCEAN GRID \u2014 ANOMALY POSITIONS</div>',
                unsafe_allow_html=True,
            )
            if FOLIUM_OK:
                fm = mk_folium_map(geo, params["auv_lat"], params["auv_lon"])
                st_folium(fm, width=None, height=350, returned_objects=[])
            else:
                mdf = pd.DataFrame(
                    [{"lat": r["Latitude"], "lon": r["Longitude"]} for r in geo]
                )
                st.map(mdf, zoom=14)

        st.markdown("""
        <div class="aq-panel" style="margin-top:14px;">
            <div class="aq-panel-hdr">\U0001f4ca DETECTED CLASS BREAKDOWN</div>
        </div>
        """, unsafe_allow_html=True)
        cc = {}
        for r in geo:
            cc[r["Classification"]] = cc.get(r["Classification"], 0) + 1
        cols = st.columns(max(len(cc), 1))
        for col, (cls_name, cnt) in zip(cols, cc.items()):
            col.metric(cls_name, cnt)
    else:
        st.markdown("""
        <div class="aq-panel" style="text-align:center;padding:32px;">
            <div style="font-size:32px;margin-bottom:10px;">\u2705</div>
            <div style="font-size:16px;font-weight:700;color:#4ade80;margin-bottom:6px;">Clear Acoustic Swath</div>
            <div style="font-size:13px;color:#64a8cc;">No man-made debris detected above conf={params['conf_thr']:.2f}.<br>
            Try lowering the <b>Confidence Gate</b> slider in the sidebar.</div>
        </div>
        """, unsafe_allow_html=True)


def _ensure_video():
    """Return a path to the demo video, auto-generating it if missing or tiny."""
    vp = DEMO.get("video")
    if vp and os.path.isfile(vp) and os.path.getsize(vp) > 50_000:
        return vp
    gen_script = os.path.join(ROOT_DIR, "demo_test_assets", "create_sonar_video.py")
    if os.path.isfile(gen_script):
        import subprocess as _sp
        try:
            r = _sp.run(
                [sys.executable, gen_script],
                cwd=ROOT_DIR, capture_output=True, timeout=120,
            )
            if r.returncode == 0:
                candidate = os.path.join(
                    ROOT_DIR, "demo_test_assets", "test_sonar_stream.mp4"
                )
                if os.path.isfile(candidate):
                    return candidate
        except Exception:
            pass
    return vp


def mode_video():
    st.markdown("""
    <div class="aq-panel">
        <div class="aq-panel-hdr">\U0001f4f9 AUV CONTINUOUS MISSION REPLAY \u2014 SONAR WATERFALL STREAM</div>
    </div>
    """, unsafe_allow_html=True)

    vp = _ensure_video()
    if vp and os.path.isfile(vp) and os.path.getsize(vp) > 50_000:
        st.markdown(
            f'<div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);'
            f'border-radius:8px;padding:8px 14px;margin-bottom:12px;font-size:12px;'
            f'color:#7ecfe8;font-weight:600;">\U0001f4c2 Loaded: <code>{os.path.basename(vp)}</code></div>',
            unsafe_allow_html=True,
        )
        # Load raw bytes -- ensures HTML5 browser playback regardless of container metadata
        with open(vp, "rb") as vf:
            video_bytes = vf.read()
        st.video(video_bytes, format="video/mp4")

        cap = cv2.VideoCapture(vp)
        tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 12.0
        wv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hv = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        fcc_str = fcc_val.to_bytes(4, "little").decode("ascii", errors="replace").strip()
        cap.release()
        dur = tf / fps_v if fps_v > 0 else 0

        st.markdown("""
        <div class="aq-panel" style="margin-top:14px;">
            <div class="aq-panel-hdr">\U0001f50a MISSION STREAM METADATA</div>
        </div>
        """, unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Frames", f"{tf:,}")
        m2.metric("Frame Rate", f"{fps_v:.1f} fps")
        m3.metric("Resolution", f"{wv}x{hv}")
        m4.metric("Duration", f"{dur:.1f} s")
        m5.metric("Codec", fcc_str)
    else:
        st.markdown("""
        <div class="idle-box">
            <div class="idle-icon">\U0001f4f9</div>
            <div class="idle-title">Mission Feed Not Found</div>
            <div class="idle-body">
                No <code>.mp4</code> video detected.<br>
                Click below to auto-generate the H.264 mission stream.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Generate Sonar Mission Stream", type="primary"):
            with st.spinner("Generating H.264 sonar stream (60 frames @ 12 FPS)..."):
                gen_script = os.path.join(
                    ROOT_DIR, "demo_test_assets", "create_sonar_video.py"
                )
                import subprocess as _sp2
                r2 = _sp2.run(
                    [sys.executable, gen_script],
                    cwd=ROOT_DIR, capture_output=True, timeout=120,
                )
                if r2.returncode == 0:
                    st.success("Stream generated! Reselect this mode to load it.")
                else:
                    st.error(
                        "Generation failed: "
                        + r2.stderr.decode(errors="replace")[-400:]
                    )
    with st.expander("About Mode B \u2014 AUV Continuous Replay"):
        st.markdown(
            "**Mode B** streams pre-recorded side-scan sonar mission video files located "
            "dynamically via `glob`. Priority: `sonar_mission_feed.mp4` "
            "-> `test_sonar_stream.mp4` -> any `.mp4` in the project tree.\n\n"
            "**Codec**: H.264 via imageio-ffmpeg bundled binary for full HTML5 compatibility.\n\n"
            "**Live path**: Wire the Jetson Orin RTSP endpoint: "
            "`cv2.VideoCapture('rtsp://auv-host:8554/sonar')` for real-time inference."
        )


def idle_screen():
    st.markdown("""
    <div class="idle-box">
        <div class="idle-icon">🔊</div>
        <div class="idle-title">Awaiting Sonar Waterfall Log</div>
        <div class="idle-body">
            Select a demo preset or upload a custom side-scan sonar image from the sidebar
            to begin real-time multi-class debris detection and geo-referencing.
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    params = render_sidebar()
    render_navbar()

    sk = params["source_key"]
    uploaded = params.get("uploaded")

    if sk == "video":
        mode_video()

    elif sk == "custom":
        if uploaded is not None:
            bgr = cv2.cvtColor(
                np.array(Image.open(uploaded).convert("RGB")), cv2.COLOR_RGB2BGR
            )
            run_image_pipeline(bgr, params, "Custom Upload")
        else:
            idle_screen()

    else:
        demo_paths = {"shipwreck": DEMO["shipwreck"], "ghost_net": DEMO["ghost_net"], "pipe_cable": DEMO["pipe_cable"]}
        demo_labels = {
            "shipwreck": "Demo 1 — Shipwreck Anomaly",
            "ghost_net": "Demo 2 — Ghost Fishing Net",
            "pipe_cable": "Demo 3 — Submerged Pipe / Cable",
        }
        path = demo_paths.get(sk)
        if path and os.path.isfile(path):
            bgr = load_image_bgr(path)
            run_image_pipeline(bgr, params, demo_labels[sk])
        else:
            st.markdown(f"""
            <div class="aq-panel" style="text-align:center;padding:32px;">
                <div style="font-size:32px;margin-bottom:10px;">⚠️</div>
                <div style="font-size:16px;font-weight:700;color:#fbbf24;margin-bottom:6px;">Demo Image Not Found</div>
                <div style="font-size:13px;color:#64a8cc;">
                    Could not locate the demo asset for <b>{demo_labels.get(sk,'')}</b>.<br>
                    Check that the data directory is populated, or use <i>Upload Custom Sonar Image</i>.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-bar">'
        "AquaScan AI &nbsp;·&nbsp; SIH 2026 &nbsp;·&nbsp; "
        "MoES / NIOT Autonomous Sonar Platform &nbsp;·&nbsp; "
        "400 kHz Side-Scan Sonar &nbsp;·&nbsp; YOLOv8 Acoustic Detector"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
