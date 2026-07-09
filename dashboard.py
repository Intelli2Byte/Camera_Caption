#!/usr/bin/env python3
"""
Camera Caption — Public Production Dashboard
Full visibility redesign with proper color contrast
"""

import streamlit as st
import json
import os
import base64
import tempfile
import shutil
import re
from typing import List, Tuple

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Camera Caption — AI Video Captioning",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/riyaaisky/camera_caption",
        "Report a bug": "https://github.com/riyaaisky/camera_caption/issues",
        "About": (
            "## 📷 Camera Caption\n"
            "Multi-Style AI Video Captioning Agent\n\n"
            "Powered by Fireworks AI · Qwen3.7 Plus + GLM-5.2\n\n"
            "AMD Hackathon 2025"
        ),
    },
)

# ─────────────────────────────────────────────
# SAFE IMPORTS
# ─────────────────────────────────────────────
try:
    import requests
except ImportError:
    st.error("❌ 'requests' library not found.")
    st.stop()

try:
    import cv2
    import numpy as np
    # Quick sanity check
    _test = cv2.VideoCapture
    OPENCV_OK = True
except Exception as e:
    OPENCV_OK = False
    st.error(
        f"❌ OpenCV failed to load: `{e}`\n\n"
        "**Fix:** Make sure `packages.txt` contains `libgl1-mesa-glx` "
        "and redeploy the app."
    )
    st.stop()
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────
def get_api_key() -> str:
    try:
        key = st.secrets.get("FIREWORKS_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("FIREWORKS_API_KEY", "")

API_KEY = get_api_key()

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FIREWORKS_URL        = "https://api.fireworks.ai/inference/v1/chat/completions"
VISION_MODEL         = "accounts/fireworks/models/qwen3p7-plus"
TEXT_MODEL           = "accounts/fireworks/models/glm-5p2"
MAX_VIDEO_MB         = 100
MAX_REQUESTS_SESSION = 15

STYLE_ICONS = {
    "formal":            "📋",
    "sarcastic":         "😏",
    "humorous_tech":     "💻",
    "humorous_non_tech": "😂",
}
STYLE_LABELS = {
    "formal":            "Formal",
    "sarcastic":         "Sarcastic",
    "humorous_tech":     "Humorous Tech",
    "humorous_non_tech": "Humorous Non-Tech",
}
STYLE_INSTRUCTIONS = {
    "formal": (
        "Write a formal 2-3 sentence description. "
        "Use professional, academic language.", 0.6
    ),
    "sarcastic": (
        "Write a sarcastic 2-3 sentence commentary. "
        "Use dry wit and subtle irony.", 0.8
    ),
    "humorous_tech": (
        "Write a funny 2-3 sentence tech caption. "
        "Use programming terms: API, debug, git, deploy, code.", 0.9
    ),
    "humorous_non_tech": (
        "Write a funny 2-3 sentence everyday caption. "
        "Relatable humor. Absolutely NO tech terms.", 0.9
    ),
}
SAMPLE_VIDEOS = {
    "🌆 Urban Boulevard": (
        "https://storage.googleapis.com/amd-hackathon-clips/"
        "1860079-uhd_2560_1440_25fps.mp4"
    ),
    "🐱 Orange Kitten": (
        "https://storage.googleapis.com/amd-hackathon-clips/"
        "13825391-uhd_3840_2160_30fps.mp4"
    ),
    "💼 Office Worker": (
        "https://storage.googleapis.com/amd-hackathon-clips/"
        "3044693-uhd_3840_2160_24fps.mp4"
    ),
}
SYSTEM_GUIDELINE = (
    "You are Camera Caption, an automated API that produces raw caption strings. "
    "Output ONLY the caption text. No preambles, no meta-commentary, "
    "no thinking process. Start immediately with the caption itself."
)

# ─────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────��──────
if "request_count" not in st.session_state:
    st.session_state["request_count"] = 0
if "selected_url" not in st.session_state:
    st.session_state["selected_url"] = ""
if "batch_tasks" not in st.session_state:
    st.session_state["batch_tasks"] = []

# ─────────────────────────────────────────────
# CSS — HIGH CONTRAST VISIBLE DESIGN
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════
   GLOBAL BASE
══════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #0d1117 !important;
    color: #f0f6fc !important;
}

/* All default Streamlit text → white */
p, span, div, label, li, td, th {
    color: #f0f6fc !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ══════════════════════════════════════
   SIDEBAR
══════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
}

[data-testid="stSidebar"] * {
    color: #f0f6fc !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #c9d1d9 !important;
}

/* ══════════════════════════════════════
   INPUTS
══════════════════════════════════════ */
.stTextInput > div > div > input {
    background: #21262d !important;
    color: #f0f6fc !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}

.stTextInput > div > div > input::placeholder {
    color: #8b949e !important;
}

.stTextInput > div > div > input:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.15) !important;
}

.stTextInput label {
    color: #c9d1d9 !important;
    font-weight: 600 !important;
}

/* ══════════════════════════════════════
   MULTISELECT
══════════════════════════════════════ */
.stMultiSelect > div > div {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #f0f6fc !important;
}

.stMultiSelect label {
    color: #c9d1d9 !important;
    font-weight: 600 !important;
}

[data-baseweb="tag"] {
    background: rgba(245,158,11,0.2) !important;
    border: 1px solid #f59e0b !important;
    color: #fbbf24 !important;
    border-radius: 6px !important;
}

[data-baseweb="tag"] span {
    color: #fbbf24 !important;
}

/* ══════════════════════════════════════
   SLIDER
══════════════════════════════════════ */
.stSlider label {
    color: #c9d1d9 !important;
    font-weight: 600 !important;
}

.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: #8b949e !important;
}

/* ══════════════════════════════════════
   RADIO
══════════════════════════════════════ */
.stRadio label {
    color: #c9d1d9 !important;
    font-weight: 500 !important;
}

.stRadio > div {
    gap: 1rem !important;
}

/* ══════════════════════���═══════════════
   BUTTONS
══════════════════════════════════════ */
.stButton > button {
    background: #21262d !important;
    color: #f0f6fc !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    padding: 0.5rem 1rem !important;
}

.stButton > button:hover {
    background: #30363d !important;
    border-color: #f59e0b !important;
    color: #fbbf24 !important;
    transform: translateY(-1px) !important;
}

/* Primary button */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    color: #0d1117 !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.7rem 1.5rem !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(245,158,11,0.3) !important;
}

.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #fbbf24, #f59e0b) !important;
    box-shadow: 0 6px 20px rgba(245,158,11,0.4) !important;
    transform: translateY(-2px) !important;
    color: #0d1117 !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #34d399, #10b981) !important;
    transform: translateY(-1px) !important;
    color: #ffffff !important;
}

/* ══════════════════════════════════════
   FILE UPLOADER
══════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 2px dashed #30363d !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

[data-testid="stFileUploader"] label {
    color: #c9d1d9 !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] p {
    color: #8b949e !important;
}

/* ══════════════════════════════════════
   TABS
══════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #30363d !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f0f6fc !important;
    background: #21262d !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
}

/* ══════════════════════════════════════
   EXPANDER
══════════════════════════════════════ */
.streamlit-expanderHeader {
    background: #161b22 !important;
    color: #f0f6fc !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.streamlit-expanderContent {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ══════════════════════════════════════
   ALERTS (st.info, st.success, etc.)
══════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

[data-testid="stAlert"] p {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

/* st.info */
.stAlert[data-baseweb="notification"][kind="info"] {
    background: rgba(59,130,246,0.12) !important;
    border: 1px solid rgba(59,130,246,0.4) !important;
    color: #93c5fd !important;
}

/* st.success */
.stAlert[data-baseweb="notification"][kind="positive"] {
    background: rgba(16,185,129,0.12) !important;
    border: 1px solid rgba(16,185,129,0.4) !important;
    color: #6ee7b7 !important;
}

/* st.warning */
.stAlert[data-baseweb="notification"][kind="warning"] {
    background: rgba(245,158,11,0.12) !important;
    border: 1px solid rgba(245,158,11,0.4) !important;
    color: #fcd34d !important;
}

/* st.error */
.stAlert[data-baseweb="notification"][kind="negative"] {
    background: rgba(239,68,68,0.12) !important;
    border: 1px solid rgba(239,68,68,0.4) !important;
    color: #fca5a5 !important;
}

/* ══════════════════════════════════════
   PROGRESS BAR
══════════════════════════════════════ */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #f59e0b, #d97706) !important;
    border-radius: 10px !important;
}

.stProgress > div > div {
    background: #21262d !important;
    border-radius: 10px !important;
}

/* ══════════════════════════════════════
   METRICS
══════════════════════════════════════ */
[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f59e0b !important;
    font-size: 1.8rem !important;
    font-weight: 900 !important;
}

/* ══════════════════════════════════════
   CODE BLOCKS
══════════════════════════════════════ */
.stCodeBlock {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

.stCodeBlock code {
    color: #e6edf3 !important;
}

/* ══════════════════════════════════════
   DIVIDER
══════════════════════════════════════ */
hr {
    border-color: #30363d !important;
    margin: 1.5rem 0 !important;
}

/* ══════════════════════════════════════
   CAPTION (st.caption)
══════════════════════════════════════ */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #8b949e !important;
    font-size: 0.82rem !important;
}

/* ══════════════════════════════════════
   IMAGES
══════════════════════════════════════ */
[data-testid="stImage"] {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #30363d !important;
}

/* ══════════════════════════════════════
   HIDE STREAMLIT BRANDING
══════════════════════════════════════ */
#MainMenu  { visibility: hidden !important; }
footer     { visibility: hidden !important; }
header     { visibility: hidden !important; }

/* ══════════════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════════════ */

/* Main title */
.cc-title {
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(90deg, #f59e0b 0%, #ef4444 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 1.5rem 0 0.3rem 0;
    letter-spacing: -2px;
    line-height: 1.1;
}

.cc-subtitle {
    text-align: center;
    color: #8b949e !important;
    font-size: 1.05rem;
    margin-bottom: 0.3rem;
    font-weight: 400;
}

.cc-tagline {
    text-align: center;
    color: #f59e0b !important;
    font-size: 1rem;
    font-style: italic;
    margin-bottom: 2rem;
    font-weight: 500;
}

/* Stat boxes */
.cc-stat {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: border-color 0.2s ease;
}

.cc-stat:hover {
    border-color: #f59e0b;
}

.cc-stat-num {
    font-size: 2.2rem;
    font-weight: 900;
    color: #f59e0b !important;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.cc-stat-label {
    font-size: 0.75rem;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

/* Caption cards */
.cc-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin: 0.6rem 0;
    transition: all 0.25s ease;
}

.cc-card:hover {
    background: #1c2128;
    border-color: #f59e0b;
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(245,158,11,0.1);
}

/* Style badges */
.cc-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.cc-badge-formal {
    background: rgba(59,130,246,0.15);
    color: #93c5fd !important;
    border: 1px solid rgba(59,130,246,0.5);
}

.cc-badge-sarcastic {
    background: rgba(239,68,68,0.15);
    color: #fca5a5 !important;
    border: 1px solid rgba(239,68,68,0.5);
}

.cc-badge-humorous_tech {
    background: rgba(16,185,129,0.15);
    color: #6ee7b7 !important;
    border: 1px solid rgba(16,185,129,0.5);
}

.cc-badge-humorous_non_tech {
    background: rgba(245,158,11,0.15);
    color: #fcd34d !important;
    border: 1px solid rgba(245,158,11,0.5);
}

.cc-caption-text {
    color: #e6edf3 !important;
    font-size: 0.97rem;
    line-height: 1.65;
    margin: 0;
    font-weight: 400;
}

/* Section headers */
.cc-section {
    color: #f0f6fc !important;
    font-size: 1.2rem;
    font-weight: 700;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #30363d;
}

/* Banners */
.cc-success {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.5);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #6ee7b7 !important;
    font-weight: 600;
    text-align: center;
    margin: 1rem 0;
}

.cc-error {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.5);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #fca5a5 !important;
    margin: 0.5rem 0;
}

.cc-warning {
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.5);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #fcd34d !important;
    margin: 0.5rem 0;
}

.cc-info {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.4);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #93c5fd !important;
    font-size: 0.88rem;
    margin: 0.5rem 0;
}

/* Pipeline steps */
.cc-pipeline {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.55rem 0.9rem;
    font-size: 0.88rem;
    color: #c9d1d9 !important;
    margin-bottom: 6px;
    font-weight: 500;
}

/* How-to steps */
.cc-step {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    margin-bottom: 0.9rem;
    padding: 0.7rem;
    background: #161b22;
    border-radius: 8px;
    border: 1px solid #30363d;
}

.cc-step-num {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: #0d1117 !important;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 0.8rem;
    flex-shrink: 0;
    margin-top: 1px;
}

.cc-step-text {
    color: #c9d1d9 !important;
    font-size: 0.92rem;
    line-height: 1.5;
    padding-top: 2px;
}

/* Table */
table {
    background: #161b22 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    width: 100% !important;
}

th {
    background: #21262d !important;
    color: #f0f6fc !important;
    font-weight: 700 !important;
    padding: 0.7rem 1rem !important;
    border-bottom: 1px solid #30363d !important;
}

td {
    color: #c9d1d9 !important;
    padding: 0.6rem 1rem !important;
    border-bottom: 1px solid #21262d !important;
}

tr:last-child td {
    border-bottom: none !important;
}

/* Footer */
.cc-footer {
    text-align: center;
    color: #484f58 !important;
    font-size: 0.85rem;
    padding: 1.5rem 1rem;
    border-top: 1px solid #21262d;
    margin-top: 1rem;
}

.cc-footer a {
    color: #f59e0b !important;
    text-decoration: none;
    font-weight: 600;
}

.cc-footer a:hover {
    color: #fbbf24 !important;
    text-decoration: underline;
}

/* Sidebar brand */
.cc-sidebar-brand {
    background: linear-gradient(135deg, rgba(245,158,11,0.1), rgba(239,68,68,0.1));
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    text-align: center;
}

.cc-sidebar-title {
    font-size: 1.3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #f59e0b, #ef4444);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.cc-sidebar-sub {
    font-size: 0.78rem;
    color: #8b949e !important;
    margin-top: 2px;
}

/* Usage bar label */
.cc-usage-label {
    font-size: 0.82rem;
    color: #8b949e !important;
    margin-top: 4px;
}

/* Section divider with label */
.cc-divider {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 1.5rem 0;
}

.cc-divider-line {
    flex: 1;
    height: 1px;
    background: #30363d;
}

.cc-divider-text {
    color: #8b949e !important;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def check_rate_limit() -> bool:
    return st.session_state["request_count"] < MAX_REQUESTS_SESSION

def increment_count():
    st.session_state["request_count"] += 1

def clean_text(text: str, max_sentences: int = 3) -> str:
    patterns = [
        r'The user.*?[\n.]', r'Analysis:.*?[\n.]',
        r'\*\*.*?\*\*', r'Frame \d+:.*?[\n.]',
        r'Let me.*?[\n.]', r'I can see.*?[\n.]',
        r'Based on.*?[\n.]', r"Here\'s.*?[\n.]", r'Sure.*?[\n.]',
    ]
    for p in patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*',     r'\1', text)
    text = re.sub(
        r'^(CAPTION:|Caption:|OUTPUT:|Here is|Here\'s|Based on|Sure|Okay)[:.\s]*',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().strip('"\'')
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    if sentences:
        text = '. '.join(sentences[:max_sentences])
        if not text.endswith(('.', '!', '?')):
            text += '.'
    return text.strip()

def render_caption_card(style: str, caption: str):
    icon  = STYLE_ICONS.get(style, "📝")
    label = STYLE_LABELS.get(style, style)
    is_error = "[Error" in str(caption)
    if is_error:
        st.markdown(f"""
        <div class="cc-error">
            {icon} <strong>{label}:</strong> {caption}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="cc-card">
            <div class="cc-badge cc-badge-{style}">{icon} {label}</div>
            <p class="cc-caption-text">{caption}</p>
        </div>""", unsafe_allow_html=True)

def section_header(icon: str, title: str):
    st.markdown(
        f'<div class="cc-section">{icon} {title}</div>',
        unsafe_allow_html=True
    )

def divider_label(text: str):
    st.markdown(f"""
    <div class="cc-divider">
        <div class="cc-divider-line"></div>
        <div class="cc-divider-text">{text}</div>
        <div class="cc-divider-line"></div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CORE AGENT FUNCTIONS
# ─────────────────────────────────────────────
def download_video(video_url: str, temp_dir: str) -> str:
    try:
        response = requests.get(video_url, stream=True, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Could not download video: {e}")

    max_bytes = MAX_VIDEO_MB * 1024 * 1024
    content_length = int(response.headers.get("content-length", 0))
    if content_length > max_bytes:
        raise ValueError(
            f"Video is {content_length//(1024*1024)}MB — max allowed is {MAX_VIDEO_MB}MB."
        )

    video_path = os.path.join(temp_dir, "video.mp4")
    downloaded = 0
    with open(video_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded > max_bytes:
                raise ValueError(f"Video exceeded {MAX_VIDEO_MB}MB limit.")
    return video_path

def extract_keyframes(video_path: str, max_frames: int = 4) -> Tuple[List[str], List, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration     = total_frames / fps if fps > 0 else 0
    if total_frames == 0:
        raise ValueError("Video has no frames.")
    step = max(1, total_frames // max_frames)
    frame_indices = [i * step for i in range(min(max_frames, total_frames))]
    base64_frames, display_frames = [], []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        display_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        base64_frames.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    if not base64_frames:
        raise ValueError("No frames could be extracted.")
    return base64_frames, display_frames, duration

def call_api(payload: dict) -> str:
    if not API_KEY:
        raise ValueError("API key not configured.")
    response = requests.post(
        FIREWORKS_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        json=payload, timeout=90,
    )
    if response.status_code == 401:
        raise ValueError("Invalid API key.")
    if response.status_code == 429:
        raise ValueError("API rate limit reached. Please try again later.")
    if response.status_code != 200:
        raise ValueError(f"API error {response.status_code}: {response.text[:200]}")
    return response.json()["choices"][0]["message"]["content"].strip()

def analyze_video(base64_frames: List[str]) -> str:
    content = [{"type": "text", "text": (
        "Describe this video in 3 clear sentences. "
        "Start with 'The video shows'. "
        "Cover: subjects, actions, setting, mood."
    )}]
    for frame in base64_frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
    raw = call_api({
        "model": VISION_MODEL, "max_tokens": 150, "temperature": 0.3,
        "messages": [{"role": "user", "content": content}],
    })
    return clean_text(raw, 3)

def generate_caption(description: str, style: str) -> str:
    instruction, temperature = STYLE_INSTRUCTIONS.get(style, ("Write a 2-3 sentence caption.", 0.7))
    raw = call_api({
        "model": TEXT_MODEL, "max_tokens": 80, "temperature": temperature,
        "presence_penalty": 0.3, "frequency_penalty": 0.3,
        "messages": [
            {"role": "system", "content": SYSTEM_GUIDELINE},
            {"role": "user", "content": f"VIDEO: {description}\n\n{instruction}\n\nCAPTION:"},
        ],
    })
    return clean_text(raw, 3)

def run_pipeline(video_url: str, styles: List[str], max_frames: int, pb=None, st_txt=None):
    def upd(pct, msg):
        if pb:  pb.progress(pct, text=msg)
        if st_txt: st_txt.text(msg)
    temp_dir = tempfile.mkdtemp()
    try:
        upd(10, "📥 Downloading video...")
        video_path = download_video(video_url, temp_dir)
        upd(30, "🖼️ Extracting keyframes...")
        b64_frames, disp_frames, duration = extract_keyframes(video_path, max_frames)
        upd(55, "👁️ Qwen3.7 analyzing scene...")
        description = analyze_video(b64_frames)
        captions = {}
        n = len(styles)
        for i, style in enumerate(styles):
            upd(60 + int((i / n) * 35), f"✍️ Generating {STYLE_LABELS[style]}...")
            captions[style] = generate_caption(description, style)
            increment_count()
        upd(100, "✅ Done!")
        return description, captions, disp_frames
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="cc-sidebar-brand">
        <div class="cc-sidebar-title">📷 Camera Caption</div>
        <div class="cc-sidebar-sub">Multi-Style AI Video Captioning</div>
    </div>
    """, unsafe_allow_html=True)

    # API status
    if API_KEY:
        st.success("✅ API Connected & Ready")
    else:
        st.error("❌ API key missing")

    st.divider()

    st.markdown("### 🎨 Caption Styles")
    selected_styles = st.multiselect(
        "Select styles to generate",
        options=list(STYLE_ICONS.keys()),
        default=list(STYLE_ICONS.keys()),
        format_func=lambda x: f"{STYLE_ICONS[x]} {STYLE_LABELS[x]}",
    )

    st.divider()

    st.markdown("### ⚙️ Settings")
    max_frames = st.slider(
        "Keyframes per video", min_value=2, max_value=6, value=4,
        help="More frames = better accuracy but slower",
    )

    st.divider()

    st.markdown("### 🔄 Pipeline")
    for step in [
        "📥 Download video",
        "🖼️ Extract keyframes",
        "👁️ Qwen3.7+ analyzes",
        "✍️ GLM-5.2 captions",
        "📤 Export JSON",
    ]:
        st.markdown(f'<div class="cc-pipeline">{step}</div>', unsafe_allow_html=True)

    st.divider()

    used      = st.session_state["request_count"]
    remaining = MAX_REQUESTS_SESSION - used
    st.markdown("### 📊 Session Usage")
    st.progress(min(used / MAX_REQUESTS_SESSION, 1.0))
    st.markdown(
        f'<div class="cc-usage-label">🔢 {remaining} generation(s) remaining</div>',
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown(
        "<div style='font-size:0.78rem; color:#484f58; line-height:1.6'>"
        "🏆 AMD Hackathon 2025<br>"
        "🐳 riyaaisky/video-caption-agent<br>"
        "⚡ Fireworks AI · Qwen3.7 + GLM-5.2"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="cc-title">📷 Camera Caption</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cc-subtitle">Multi-Style AI Video Captioning · Qwen3.7 Plus Vision + GLM-5.2 Text</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="cc-tagline">"One video. Four personalities. Zero effort."</div>',
    unsafe_allow_html=True,
)

# Stats
for col, num, label in zip(
    st.columns(4),
    ["2", "4", "<10m", "100MB"],
    ["AI Models", "Caption Styles", "Runtime", "Max Video Size"],
):
    with col:
        st.markdown(
            f'<div class="cc-stat">'
            f'<div class="cc-stat-num">{num}</div>'
            f'<div class="cc-stat-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

if not API_KEY:
    st.markdown("""
    <div class="cc-error">
        ❌ <strong>Service Unavailable</strong> —
        API key not configured. Please contact the administrator.
    </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Try It Now",
    "📁 Batch Process",
    "📊 Results Viewer",
    "❓ How To Use",
])


# ══════════════════════════════════════════════
# TAB 1 — TRY IT NOW
# ══════════════════════════════════════════════
with tab1:
    section_header("🎥", "Caption Any Video Instantly")

    if not check_rate_limit():
        st.markdown("""
        <div class="cc-warning">
            ⚠️ <strong>Session limit reached.</strong>
            Refresh the page to start a new session.
        </div>""", unsafe_allow_html=True)
    else:
        input_method = st.radio(
            "Choose input method",
            ["⚡ Quick Test (Sample Videos)", "🔗 Paste Your Own URL"],
            horizontal=True,
        )

        video_url = ""

        if input_method == "⚡ Quick Test (Sample Videos)":
            divider_label("pick a sample")
            cols = st.columns(len(SAMPLE_VIDEOS))
            for col, (label, url) in zip(cols, SAMPLE_VIDEOS.items()):
                with col:
                    if st.button(label, use_container_width=True):
                        st.session_state["selected_url"] = url
                        st.rerun()
            if st.session_state["selected_url"]:
                video_url = st.session_state["selected_url"]
                st.markdown(
                    f'<div class="cc-info">📎 Selected: <code>{video_url[:65]}...</code></div>',
                    unsafe_allow_html=True
                )
        else:
            video_url = st.text_input(
                "Paste a direct MP4 video URL",
                placeholder="https://example.com/your-video.mp4",
                help="Must be a direct link to an MP4 file (max 100MB)",
            )
            if video_url:
                st.caption("⚠️ Only direct MP4 links supported. Max 100MB.")

        if not selected_styles:
            st.warning("⚠️ Select at least one caption style in the sidebar.")

        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button(
            "📷 Generate Captions",
            type="primary",
            use_container_width=True,
            disabled=(not video_url or not selected_styles),
        )

        if generate_btn and video_url and selected_styles:
            pb  = st.progress(0, text="Starting...")
            stx = st.empty()
            try:
                description, captions, display_frames = run_pipeline(
                    video_url, selected_styles, max_frames, pb, stx
                )
                stx.empty()

                divider_label("extracted keyframes")
                fcols = st.columns(len(display_frames))
                for i, (c, f) in enumerate(zip(fcols, display_frames)):
                    with c:
                        st.image(f, caption=f"Frame {i+1}", use_container_width=True)

                divider_label("scene analysis · qwen3.7 plus")
                st.info(f"🔍 **{description}**")

                divider_label("camera caption results · glm-5.2")
                for style, caption in captions.items():
                    render_caption_card(style, caption)

                st.markdown("""
                <div class="cc-success">
                    🎉 All captions generated successfully by Camera Caption!
                </div>""", unsafe_allow_html=True)

                result = {
                    "video_url": video_url,
                    "scene_description": description,
                    "captions": captions,
                }
                st.download_button(
                    label="📥 Download Results as JSON",
                    data=json.dumps([result], indent=2),
                    file_name="camera_caption_results.json",
                    mime="application/json",
                    use_container_width=True,
                )

            except Exception as e:
                pb.empty(); stx.empty()
                st.markdown(
                    f'<div class="cc-error">❌ <strong>Error:</strong> {e}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            f'<div class="cc-info" style="margin-top:1rem">'
            f'ℹ️ Free public demo · '
            f'{MAX_REQUESTS_SESSION} generations per session · '
            f'Each video uses {len(selected_styles)} generation(s)'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# TAB 2 — BATCH PROCESS
# ══════════════════════════════════════════════
with tab2:
    section_header("📁", "Batch Process Multiple Videos")
    st.markdown(
        '<p style="color:#8b949e; margin-bottom:1.5rem">'
        'Upload a <code>tasks.json</code> file to process multiple videos at once. '
        'Perfect for the hackathon submission format.'
        '</p>',
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📤 Upload tasks.json")
        uploaded_file = st.file_uploader(
            "Upload your tasks.json",
            type=["json"],
            help="JSON array with task_id, video_url, and styles",
        )
        tasks_from_upload = []
        if uploaded_file:
            try:
                tasks_from_upload = json.load(uploaded_file)
                st.success(f"✅ Loaded {len(tasks_from_upload)} task(s)")
                with st.expander("👁️ Preview tasks.json"):
                    st.json(tasks_from_upload)
            except Exception as e:
                st.error(f"❌ Invalid JSON: {e}")

    with col_right:
        st.markdown("#### ⚡ Or Use Hackathon Test Clips")
        if st.button("📋 Load Hackathon Tasks", use_container_width=True):
            st.session_state["batch_tasks"] = [
                {"task_id": "v1", "video_url": SAMPLE_VIDEOS["🌆 Urban Boulevard"],   "styles": list(STYLE_ICONS.keys())},
                {"task_id": "v2", "video_url": SAMPLE_VIDEOS["🐱 Orange Kitten"],     "styles": list(STYLE_ICONS.keys())},
                {"task_id": "v3", "video_url": SAMPLE_VIDEOS["💼 Office Worker"],     "styles": list(STYLE_ICONS.keys())},
            ]
            st.success("✅ Hackathon tasks loaded!")

        st.markdown("#### 📋 Expected Format")
        st.code(
            '[\n  {\n    "task_id": "v1",\n'
            '    "video_url": "https://...",\n'
            '    "styles": [\n      "formal",\n      "sarcastic",\n'
            '      "humorous_tech",\n      "humorous_non_tech"\n    ]\n  }\n]',
            language="json",
        )

    tasks_to_run = tasks_from_upload or st.session_state.get("batch_tasks", [])

    if tasks_to_run:
        total_gens = sum(len(t.get("styles", selected_styles)) for t in tasks_to_run)
        st.info(f"📊 {len(tasks_to_run)} video(s) · {total_gens} total caption generation(s)")

        if not check_rate_limit():
            st.markdown('<div class="cc-warning">⚠️ Session limit reached. Refresh to continue.</div>', unsafe_allow_html=True)
        else:
            if st.button(f"📷 Run Camera Caption on {len(tasks_to_run)} Video(s)", type="primary", use_container_width=True):
                all_results = []
                overall_pb  = st.progress(0, text="Starting batch...")

                for idx, task in enumerate(tasks_to_run):
                    task_id = task["task_id"]
                    styles  = task.get("styles", selected_styles)
                    overall_pb.progress(
                        int(idx / len(tasks_to_run) * 100),
                        text=f"Processing {task_id} ({idx+1}/{len(tasks_to_run)})..."
                    )
                    result = {"task_id": task_id, "captions": {}}

                    with st.expander(f"📹 Task: {task_id}", expanded=True):
                        try:
                            p = st.progress(0)
                            s = st.empty()
                            description, captions, disp_frames = run_pipeline(
                                task["video_url"], styles, max_frames, p, s
                            )
                            s.empty()
                            fcols = st.columns(len(disp_frames))
                            for i, (c, f) in enumerate(zip(fcols, disp_frames)):
                                with c:
                                    st.image(f, caption=f"Frame {i+1}", use_container_width=True)
                            st.info(f"🔍 **Scene:** {description}")
                            for style, caption in captions.items():
                                result["captions"][style] = caption
                                render_caption_card(style, caption)
                            st.success(f"✅ Task {task_id} complete!")
                        except Exception as e:
                            st.error(f"❌ Task {task_id} failed: {e}")
                            for style in styles:
                                if style not in result["captions"]:
                                    result["captions"][style] = f"[Error: {str(e)[:80]}]"
                    all_results.append(result)

                overall_pb.progress(100, text="✅ All tasks complete!")
                divider_label("final output")
                results_json = json.dumps(all_results, indent=2)
                st.code(results_json, language="json")
                st.download_button(
                    label="📥 Download results.json",
                    data=results_json,
                    file_name="results.json",
                    mime="application/json",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════
# TAB 3 — RESULTS VIEWER
# ══════════════════════════════════════════════
with tab3:
    section_header("📊", "View & Inspect Results")

    results_file = st.file_uploader(
        "Upload results.json to inspect",
        type=["json"],
        key="results_uploader",
    )

    if results_file:
        try:
            results    = json.load(results_file)
            total_caps = sum(len(r.get("captions", {})) for r in results)
            err_count  = sum(1 for r in results for c in r.get("captions", {}).values() if "[Error" in str(c))

            st.success(f"✅ Loaded {len(results)} result(s)")

            for col, val, label in zip(
                st.columns(4),
                [len(results), total_caps, total_caps - err_count, err_count],
                ["Videos", "Total Captions", "✅ Successful", "❌ Errors"],
            ):
                col.metric(label, val)

            st.divider()

            for result in results:
                task_id  = result.get("task_id", "Unknown")
                captions = result.get("captions", {})
                with st.expander(f"📷 Task: {task_id}", expanded=True):
                    for style, caption in captions.items():
                        render_caption_card(style, caption)

            st.download_button(
                label="📥 Re-download results.json",
                data=json.dumps(results, indent=2),
                file_name="results.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"❌ Could not parse file: {e}")
    else:
        st.info("👆 Upload a results.json file to inspect captions")
        st.code(
            '[\n  {\n    "task_id": "v1",\n    "captions": {\n'
            '      "formal": "The video depicts...",\n'
            '      "sarcastic": "Oh wow, another...",\n'
            '      "humorous_tech": "This is running on...",\n'
            '      "humorous_non_tech": "This has the energy of..."\n'
            '    }\n  }\n]',
            language="json",
        )


# ══════════════════════════════════════════════
# TAB 4 — HOW TO USE
# ══════════════════════════════════════════════
with tab4:
    section_header("❓", "How to Use Camera Caption")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🚀 Quick Start")
        for num, text in enumerate([
            "Go to the <strong>🚀 Try It Now</strong> tab",
            "Click a <strong>sample video</strong> or paste your own MP4 URL",
            "Select your <strong>caption styles</strong> in the sidebar",
            "Click <strong>📷 Generate Captions</strong>",
            "Download your results as <strong>JSON</strong>",
        ], 1):
            st.markdown(f"""
            <div class="cc-step">
                <div class="cc-step-num">{num}</div>
                <div class="cc-step-text">{text}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🎨 Caption Styles")
        descriptions = {
            "formal":            "Professional, objective, academic tone",
            "sarcastic":         "Dry wit, ironic, lightly mocking",
            "humorous_tech":     "Programming jokes, coding references",
            "humorous_non_tech": "Universal everyday humor, no tech terms",
        }
        for style, icon in STYLE_ICONS.items():
            st.markdown(f"""
            <div class="cc-card" style="margin:4px 0; padding:0.8rem 1rem">
                <div class="cc-badge cc-badge-{style}">{icon} {STYLE_LABELS[style]}</div>
                <p class="cc-caption-text" style="font-size:0.88rem; margin-top:4px">
                    {descriptions[style]}
                </p>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📁 Batch Processing")
        for num, text in enumerate([
            "Go to the <strong>📁 Batch Process</strong> tab",
            "Upload a <strong>tasks.json</strong> or load sample tasks",
            "Click <strong>Run Camera Caption</strong>",
            "Download the final <strong>results.json</strong>",
        ], 1):
            st.markdown(f"""
            <div class="cc-step">
                <div class="cc-step-num">{num}</div>
                <div class="cc-step-text">{text}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ⚙️ Technical Details")
        st.markdown("""
| Component | Detail |
|-----------|--------|
| Vision Model | Qwen3.7 Plus |
| Text Model | GLM-5.2 |
| API Provider | Fireworks AI |
| Max Video Size | 100 MB |
| Keyframes | 2–6 per video |
| Output Format | JSON |
""")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🐳 Docker Deployment")
        st.code("docker pull riyaaisky/video-caption-agent:latest", language="bash")
        st.code(
            "docker run --rm \\\n"
            "  -v /path/to/input:/input:ro \\\n"
            "  -v /path/to/output:/output \\\n"
            "  -e FIREWORKS_API_KEY=$KEY \\\n"
            "  riyaaisky/video-caption-agent:latest",
            language="bash",
        )


# ─────────────────────────────────────────────
# FOOTER
# ────────────────────────��────────────────────
st.markdown("""
<div class="cc-footer">
    📷 <strong>Camera Caption</strong> &nbsp;·&nbsp;
    Powered by <a href="https://fireworks.ai">Fireworks AI</a> &nbsp;·&nbsp;
    Qwen3.7 Plus + GLM-5.2 &nbsp;·&nbsp;
    AMD Hackathon 2025 &nbsp;·&nbsp;
    <a href="https://github.com/riyaaisky/camera_caption">GitHub</a> &nbsp;·&nbsp;
    🐳 riyaaisky/video-caption-agent:latest
</div>
""", unsafe_allow_html=True)
