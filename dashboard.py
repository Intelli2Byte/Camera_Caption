#!/usr/bin/env python3
"""
Camera Caption — Clean Minimal Dashboard
Pillow-based, no OpenCV, works on Streamlit Cloud
"""

import streamlit as st
import json, os, base64, tempfile, shutil, re, io
from typing import List, Tuple
import requests
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Camera Caption",
    page_icon="📷",
    layout="centered",          # centered = less sprawl
    initial_sidebar_state="expanded",
)

# ── Minimal CSS — dark bg, readable text, no clutter ───────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #0f1117 !important;
    color: #e6edf3 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
}

/* All text visible */
p, span, div, label, li { color: #e6edf3 !important; }
h1, h2, h3, h4           { color: #ffffff !important; font-weight: 700 !important; }
.stCaption               { color: #8b949e !important; }

/* Inputs */
.stTextInput input {
    background: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}
.stTextInput input:focus { border-color: #f59e0b !important; box-shadow: none !important; }

/* Multiselect */
.stMultiSelect > div > div {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}
[data-baseweb="tag"] {
    background: rgba(245,158,11,0.15) !important;
    border: 1px solid #f59e0b !important;
}
[data-baseweb="tag"] span { color: #fbbf24 !important; }

/* Primary button */
button[data-testid="baseButton-primary"] {
    background: #f59e0b !important;
    color: #0f1117 !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}
button[data-testid="baseButton-primary"]:hover {
    background: #fbbf24 !important;
}

/* Secondary buttons */
.stButton > button {
    background: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}
.stButton > button:hover {
    border-color: #f59e0b !important;
    color: #fbbf24 !important;
}

/* Download button */
.stDownloadButton > button {
    background: #1a7f4b !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22 !important;
    border-radius: 8px !important;
    border: 1px solid #30363d !important;
    gap: 4px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: #f59e0b !important;
    color: #0f1117 !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: #f59e0b !important;
}
.stProgress > div > div {
    background: #21262d !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-top: none !important;
}

/* Code blocks */
.stCodeBlock { background: #161b22 !important; border: 1px solid #30363d !important; }
code { color: #e6edf3 !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: #f59e0b !important; font-weight: 700 !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }

/* ── Caption card ── */
.caption-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.caption-card:hover { border-color: #f59e0b; }
.caption-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    display: block;
}
.caption-text {
    font-size: 0.97rem;
    line-height: 1.65;
    color: #e6edf3 !important;
    margin: 0;
}
.label-formal            { color: #93c5fd !important; }
.label-sarcastic         { color: #fca5a5 !important; }
.label-humorous_tech     { color: #6ee7b7 !important; }
.label-humorous_non_tech { color: #fcd34d !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
VISION_MODEL  = "accounts/fireworks/models/qwen3p7-plus"
TEXT_MODEL    = "accounts/fireworks/models/glm-5p2"
MAX_VIDEO_MB  = 100
SESSION_LIMIT = 15

STYLES = {
    "formal":            ("📋", "Formal",           0.55),
    "sarcastic":         ("😏", "Sarcastic",        0.80),
    "humorous_tech":     ("💻", "Humorous Tech",    0.85),
    "humorous_non_tech": ("😂", "Humorous Non-Tech",0.85),
}

STYLE_INSTRUCTIONS = {
    "formal":            "Write 2 formal, professional sentences describing the video objectively.",
    "sarcastic":         "Write 2 sarcastic, dry-wit sentences commenting on the video.",
    "humorous_tech":     "Write 2 funny sentences using coding/developer jokes relevant to the video.",
    "humorous_non_tech": "Write 2 funny everyday sentences. No tech terms whatsoever.",
}

SAMPLE_VIDEOS = {
    "🌆 Urban Boulevard": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "🐱 Orange Kitten":   "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
    "💼 Office Worker":   "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4",
}

SYSTEM_PROMPT = (
    "You are Camera Caption. Output ONLY the caption text. "
    "No preamble, no explanation, no thinking. Start with the caption immediately."
)

# ── Session state ───────────────────────────────────────────────
for k, v in {"request_count": 0, "selected_url": "", "batch_tasks": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API key ────────────────────────────────────���────────────────
def get_api_key() -> str:
    try:
        k = st.secrets.get("FIREWORKS_API_KEY", "")
        if k: return k
    except Exception:
        pass
    return os.getenv("FIREWORKS_API_KEY", "")

API_KEY = get_api_key()

# ── Helpers ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Strip model preambles and normalize whitespace."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^(Caption:|OUTPUT:|Here is|Here\'s|Sure[,!]?|Okay[,.]?)\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip().strip('"\'')
    return text

def caption_card(style: str, caption: str):
    icon, label, _ = STYLES[style]
    if "[Error" in caption:
        st.error(f"{icon} **{label}:** {caption}")
    else:
        st.markdown(f"""
        <div class="caption-card">
            <span class="caption-label label-{style}">{icon} {label}</span>
            <p class="caption-text">{caption}</p>
        </div>""", unsafe_allow_html=True)

# ── Core functions ───────────────────────────────────────────────
def download_video(url: str, dest: str) -> str:
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    max_b = MAX_VIDEO_MB * 1024 * 1024
    path  = os.path.join(dest, "video.mp4")
    total = 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            total += len(chunk)
            if total > max_b:
                raise ValueError(f"Video exceeds {MAX_VIDEO_MB} MB limit.")
    return path


def extract_frames(video_path: str, n: int = 4) -> Tuple[List[str], List[Image.Image]]:
    """Pure-Python MP4 frame extraction — no OpenCV needed."""
    size      = os.path.getsize(video_path)
    positions = [int(size * i / (n + 1)) for i in range(1, n + 1)]
    b64s, imgs = [], []

    with open(video_path, "rb") as f:
        for pos in positions:
            f.seek(max(0, pos - 512 * 1024))
            chunk = f.read(2 * 1024 * 1024)
            start = chunk.find(b'\xff\xd8\xff')
            if start != -1:
                data = chunk[start:]
                end  = data.find(b'\xff\xd9')
                if end != -1:
                    data = data[:end + 2]
                    try:
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        w, h = img.size
                        if w > 640:
                            img = img.resize((640, int(h * 640 / w)), Image.LANCZOS)
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=75)
                        b64s.append(base64.b64encode(buf.getvalue()).decode())
                        imgs.append(img)
                        continue
                    except Exception:
                        pass
            # Fallback placeholder
            colors = [(70,130,180),(34,139,34),(210,105,30),(128,0,128)]
            img = Image.new("RGB", (640, 360), colors[len(imgs) % 4])
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            b64s.append(base64.b64encode(buf.getvalue()).decode())
            imgs.append(img)

    return b64s, imgs


def call_api(payload: dict) -> str:
    if not API_KEY:
        raise ValueError("API key not configured.")
    r = requests.post(
        FIREWORKS_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload, timeout=90,
    )
    if r.status_code == 401: raise ValueError("Invalid API key.")
    if r.status_code == 429: raise ValueError("Rate limit hit — wait a moment and retry.")
    if r.status_code != 200: raise ValueError(f"API error {r.status_code}: {r.text[:150]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def analyze_scene(frames: List[str]) -> str:
    content = [{"type": "text", "text": (
        "Describe this video in 2 clear sentences. "
        "Start with 'The video shows'. Cover: subjects, actions, setting."
    )}]
    for f in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
    raw = call_api({"model": VISION_MODEL, "max_tokens": 120, "temperature": 0.3,
                    "messages": [{"role": "user", "content": content}]})
    return clean_text(raw)


def generate_caption(description: str, style: str) -> str:
    _, _, temp = STYLES[style]
    raw = call_api({
        "model": TEXT_MODEL, "max_tokens": 80, "temperature": temp,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"VIDEO: {description}\n\n{STYLE_INSTRUCTIONS[style]}\n\nCAPTION:"},
        ],
    })
    return clean_text(raw)


def run_pipeline(url: str, styles: List[str], n_frames: int, pb=None, status=None):
    def upd(pct, msg):
        if pb:     pb.progress(pct, text=msg)
        if status: status.text(msg)

    tmp = tempfile.mkdtemp()
    try:
        upd(10, "📥 Downloading video...")
        path = download_video(url, tmp)

        upd(30, "🖼️ Extracting frames...")
        b64s, imgs = extract_frames(path, n_frames)

        upd(50, "👁️ Analyzing scene with Qwen3.7...")
        desc = analyze_scene(b64s)

        captions = {}
        for i, style in enumerate(styles):
            upd(55 + int(i / len(styles) * 40), f"✍️ Writing {STYLES[style][1]} caption...")
            captions[style] = generate_caption(desc, style)
            st.session_state["request_count"] += 1

        upd(100, "✅ Done!")
        return desc, captions, imgs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📷 Camera Caption")
    st.caption("Multi-style AI video captioning")
    st.divider()

    # API status
    if API_KEY:
        st.success("✅ API key connected")
    else:
        st.error("❌ API key missing")
        st.caption("Add FIREWORKS_API_KEY to Streamlit secrets.")

    st.divider()
    st.markdown("**🎨 Caption Styles**")
    selected_styles = st.multiselect(
        "Select styles",
        options=list(STYLES.keys()),
        default=list(STYLES.keys()),
        format_func=lambda x: f"{STYLES[x][0]} {STYLES[x][1]}",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**⚙️ Settings**")
    n_frames = st.slider("Keyframes to extract", 2, 6, 4)

    st.divider()
    used = st.session_state["request_count"]
    left = SESSION_LIMIT - used
    st.markdown("**📊 Session usage**")
    st.progress(min(used / SESSION_LIMIT, 1.0))
    st.caption(f"{left} generation(s) remaining")

    st.divider()
    st.caption("🏆 AMD Hackathon 2025  \n⚡ Fireworks AI · Qwen3.7 + GLM-5.2")


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("# 📷 Camera Caption")
st.markdown("**One video. Four personalities. Zero effort.**")
st.divider()

if not API_KEY:
    st.error("❌ API key not configured. Add `FIREWORKS_API_KEY` to Streamlit secrets.")
    st.stop()

if not selected_styles:
    st.warning("⚠️ Select at least one caption style in the sidebar.")


# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Try It", "📁 Batch", "📊 Results", "❓ Help"])


# ── Tab 1: Single video ─────────────────────────────────────────
with tab1:
    st.markdown("### Caption a video")

    mode = st.radio("Input", ["Sample videos", "Paste URL"], horizontal=True, label_visibility="collapsed")
    video_url = ""

    if mode == "Sample videos":
        cols = st.columns(len(SAMPLE_VIDEOS))
        for col, (label, url) in zip(cols, SAMPLE_VIDEOS.items()):
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state["selected_url"] = url
                    st.rerun()
        if st.session_state["selected_url"]:
            video_url = st.session_state["selected_url"]
            st.caption(f"Selected: `{video_url[:60]}...`")
    else:
        video_url = st.text_input("MP4 URL", placeholder="https://example.com/video.mp4")

    st.markdown("")
    can_run = bool(video_url and selected_styles and st.session_state["request_count"] < SESSION_LIMIT)

    if st.session_state["request_count"] >= SESSION_LIMIT:
        st.warning("Session limit reached. Refresh the page to continue.")

    if st.button("📷 Generate Captions", type="primary", use_container_width=True, disabled=not can_run):
        pb  = st.progress(0)
        msg = st.empty()
        try:
            desc, captions, imgs = run_pipeline(video_url, selected_styles, n_frames, pb, msg)
            msg.empty()

            # Frames
            st.markdown("**Extracted frames**")
            fcols = st.columns(len(imgs))
            for i, (c, img) in enumerate(zip(fcols, imgs)):
                with c:
                    st.image(img, caption=f"Frame {i+1}", use_container_width=True)

            # Scene description
            st.markdown("**Scene analysis** *(Qwen3.7 Plus)*")
            st.info(desc)

            # Captions
            st.markdown("**Captions** *(GLM-5.2)*")
            for style, cap in captions.items():
                caption_card(style, cap)

            st.success("✅ Done!")

            # Download
            result = {"video_url": video_url, "scene_description": desc, "captions": captions}
            st.download_button(
                "📥 Download JSON",
                data=json.dumps([result], indent=2),
                file_name="captions.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception as e:
            pb.empty(); msg.empty()
            st.error(f"❌ {e}")


# ── Tab 2: Batch ────────────────────────────────────────────────
with tab2:
    st.markdown("### Batch process multiple videos")
    st.caption("Upload a `tasks.json` or use the hackathon sample clips.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Upload tasks.json**")
        uploaded = st.file_uploader("tasks.json", type=["json"], label_visibility="collapsed")
        tasks_upload = []
        if uploaded:
            try:
                tasks_upload = json.load(uploaded)
                st.success(f"✅ {len(tasks_upload)} task(s) loaded")
                with st.expander("Preview"):
                    st.json(tasks_upload)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    with col_b:
        st.markdown("**Hackathon clips**")
        if st.button("Load sample tasks", use_container_width=True):
            st.session_state["batch_tasks"] = [
                {"task_id": "v1", "video_url": SAMPLE_VIDEOS["🌆 Urban Boulevard"],  "styles": list(STYLES.keys())},
                {"task_id": "v2", "video_url": SAMPLE_VIDEOS["🐱 Orange Kitten"],    "styles": list(STYLES.keys())},
                {"task_id": "v3", "video_url": SAMPLE_VIDEOS["💼 Office Worker"],    "styles": list(STYLES.keys())},
            ]
            st.success("✅ Loaded 3 hackathon tasks")
        st.markdown("**Expected format**")
        st.code('[\n  {\n    "task_id": "v1",\n    "video_url": "https://...",\n    "styles": ["formal","sarcastic"]\n  }\n]', language="json")

    tasks = tasks_upload or st.session_state.get("batch_tasks", [])

    if tasks:
        st.info(f"{len(tasks)} video(s) queued")
        if st.button(f"📷 Run on {len(tasks)} video(s)", type="primary", use_container_width=True):
            all_results = []
            overall = st.progress(0)

            for idx, task in enumerate(tasks):
                tid    = task["task_id"]
                styles = task.get("styles", selected_styles)
                overall.progress(int(idx / len(tasks) * 100), text=f"Processing {tid}...")
                result = {"task_id": tid, "captions": {}}

                with st.expander(f"📹 {tid}", expanded=True):
                    try:
                        p = st.progress(0)
                        s = st.empty()
                        desc, captions, imgs = run_pipeline(task["video_url"], styles, n_frames, p, s)
                        s.empty()

                        fcols = st.columns(len(imgs))
                        for i, (c, img) in enumerate(zip(fcols, imgs)):
                            with c:
                                st.image(img, caption=f"Frame {i+1}", use_container_width=True)

                        st.info(f"🔍 {desc}")
                        for style, cap in captions.items():
                            result["captions"][style] = cap
                            caption_card(style, cap)
                        st.success(f"✅ {tid} complete")
                    except Exception as e:
                        st.error(f"❌ {tid} failed: {e}")
                        for style in styles:
                            result["captions"].setdefault(style, f"[Error: {str(e)[:60]}]")

                all_results.append(result)

            overall.progress(100, text="✅ All done!")
            out = json.dumps(all_results, indent=2)
            st.code(out, language="json")
            st.download_button("📥 Download results.json", data=out, file_name="results.json", mime="application/json", use_container_width=True)


# ── Tab 3: Results viewer ───────────────────────────────────────
with tab3:
    st.markdown("### Inspect a results.json file")
    rf = st.file_uploader("Upload results.json", type=["json"], key="rv")
    if rf:
        try:
            results = json.load(rf)
            total   = sum(len(r.get("captions", {})) for r in results)
            errors  = sum(1 for r in results for c in r.get("captions", {}).values() if "[Error" in str(c))
            c1, c2, c3 = st.columns(3)
            c1.metric("Videos",   len(results))
            c2.metric("Captions", total)
            c3.metric("Errors",   errors)
            st.divider()
            for r in results:
                with st.expander(f"📷 {r.get('task_id', 'Unknown')}", expanded=True):
                    for style, cap in r.get("captions", {}).items():
                        caption_card(style, cap)
            st.download_button("📥 Re-download", data=json.dumps(results, indent=2), file_name="results.json", mime="application/json")
        except Exception as e:
            st.error(f"Could not parse file: {e}")
    else:
        st.info("Upload a results.json file to inspect captions here.")


# ── Tab 4: Help ─────────────────────────────────────────────────
with tab4:
    st.markdown("### How to use Camera Caption")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Quick start**")
        for i, step in enumerate([
            "Go to **🚀 Try It** tab",
            "Pick a sample video or paste an MP4 URL",
            "Choose caption styles in the sidebar",
            "Click **📷 Generate Captions**",
            "Download results as JSON",
        ], 1):
            st.markdown(f"{i}. {step}")

        st.markdown("")
        st.markdown("**Caption styles**")
        for style, (icon, label, _) in STYLES.items():
            desc = {
                "formal":            "Professional, objective tone",
                "sarcastic":         "Dry wit and irony",
                "humorous_tech":     "Coding jokes and dev culture",
                "humorous_non_tech": "Everyday relatable humor",
            }[style]
            st.markdown(f"- {icon} **{label}** — {desc}")

    with col2:
        st.markdown("**Batch processing**")
        for i, step in enumerate([
            "Go to **📁 Batch** tab",
            "Upload a `tasks.json` or load sample tasks",
            "Click **Run**",
            "Download `results.json`",
        ], 1):
            st.markdown(f"{i}. {step}")

        st.markdown("")
        st.markdown("**Technical details**")
        st.markdown("""
| | |
|---|---|
| Vision model | Qwen3.7 Plus |
| Text model | GLM-5.2 |
| API | Fireworks AI |
| Max video | 100 MB |
| Keyframes | 2–6 |
| Output | JSON |
        """)

        st.markdown("")
        st.markdown("**Docker**")
        st.code("docker run --rm \\\n  -v /input:/input:ro \\\n  -v /output:/output \\\n  -e FIREWORKS_API_KEY=$KEY \\\n  riyaaisky/video-caption-agent:latest", language="bash")


# ── Footer ──────────────────────────────────────────────────────
st.divider()
st.caption("📷 Camera Caption · Fireworks AI · Qwen3.7 Plus + GLM-5.2 · AMD Hackathon 2025")