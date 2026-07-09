#!/usr/bin/env python3
"""
Camera Caption — Public Production Dashboard
Deployed on Streamlit Cloud
API key handled via st.secrets (hidden from users)
"""

import streamlit as st
import json
import os
import base64
import tempfile
import shutil
import re
from typing import List
import requests
import cv2

# ─────────────────────────────────────────────
# PAGE CONFIG — Must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Camera Caption — AI Video Captioning",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/riyaaisky/video-caption-agent',
        'Report a bug': 'https://github.com/riyaaisky/video-caption-agent/issues',
        'About': (
            "## 📷 Camera Caption\n"
            "Multi-Style AI Video Captioning Agent\n\n"
            "Built with Fireworks AI · Qwen3.7 Plus + GLM-5.2\n\n"
            "AMD Hackathon 2025"
        )
    }
)

# ─────────────────────────────────────────────
# API KEY — From Streamlit secrets (hidden)
# ─────────────────────────────────────────────
def get_api_key() -> str:
    """
    Get API key from Streamlit secrets (production)
    Falls back to environment variable (local dev)
    """
    try:
        # Production: key stored in Streamlit Cloud secrets
        return st.secrets["FIREWORKS_API_KEY"]
    except Exception:
        # Local dev: key from .env file
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("FIREWORKS_API_KEY", "")

API_KEY = get_api_key()

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ── */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }

    /* ── Header ── */
    .main-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #f59e0b, #ef4444, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0 0.2rem 0;
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 0.4rem;
    }
    .tagline {
        text-align: center;
        color: #f59e0b;
        font-size: 0.95rem;
        font-style: italic;
        margin-bottom: 1.5rem;
    }

    /* ── Stat boxes ── */
    .stat-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 900;
        color: #f59e0b;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Caption cards ── */
    .caption-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .caption-card:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(245,158,11,0.5);
        transform: translateY(-2px);
    }
    .caption-text {
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Style badges ── */
    .style-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-formal {
        background: rgba(59,130,246,0.2);
        color: #60a5fa;
        border: 1px solid #3b82f6;
    }
    .badge-sarcastic {
        background: rgba(239,68,68,0.2);
        color: #f87171;
        border: 1px solid #ef4444;
    }
    .badge-humorous_tech {
        background: rgba(16,185,129,0.2);
        color: #34d399;
        border: 1px solid #10b981;
    }
    .badge-humorous_non_tech {
        background: rgba(245,158,11,0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
    }

    /* ── Banners ── */
    .success-banner {
        background: rgba(16,185,129,0.15);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 1rem;
        color: #34d399;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    .error-banner {
        background: rgba(239,68,68,0.15);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 1rem;
        color: #f87171;
        margin: 0.5rem 0;
    }
    .warning-banner {
        background: rgba(245,158,11,0.15);
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem;
        color: #fbbf24;
        margin: 0.5rem 0;
    }

    /* ── Pipeline steps ── */
    .pipeline-box {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        font-size: 0.85rem;
        color: #fbbf24;
        margin-bottom: 6px;
    }

    /* ── How-to card ── */
    .howto-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .howto-step {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .step-num {
        background: #f59e0b;
        color: #0f0f1a;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 0.8rem;
        flex-shrink: 0;
    }

    /* ── Rate limit notice ── */
    .rate-notice {
        background: rgba(168,85,247,0.1);
        border: 1px solid rgba(168,85,247,0.3);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #c4b5fd;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }

    /* ── Footer ── */
    .footer-text {
        text-align: center;
        color: #475569;
        font-size: 0.85rem;
        padding: 1rem;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
STYLE_ICONS = {
    "formal":            "📋",
    "sarcastic":         "😏",
    "humorous_tech":     "💻",
    "humorous_non_tech": "😂"
}

STYLE_LABELS = {
    "formal":            "Formal",
    "sarcastic":         "Sarcastic",
    "humorous_tech":     "Humorous Tech",
    "humorous_non_tech": "Humorous Non-Tech"
}

SAMPLE_VIDEOS = {
    "🌆 Urban Autumn Boulevard": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "🐱 Orange Kitten in Garden": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
    "💼 Office Worker at Desk":   "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4",
}

# ─────────────────────────────────────────────
# RATE LIMITING (per session)
# ─────────────────────────────────────────────
if "request_count" not in st.session_state:
    st.session_state["request_count"] = 0

MAX_REQUESTS_PER_SESSION = 10  # Adjust as needed


def check_rate_limit() -> bool:
    """Returns True if user is within rate limit"""
    return st.session_state["request_count"] < MAX_REQUESTS_PER_SESSION


def increment_request_count():
    st.session_state["request_count"] += 1


# ─────────────────────────────────────────────
# CORE AGENT FUNCTIONS
# ─────────────────────────────────────────────

def download_video(video_url: str, temp_dir: str) -> str:
    """Download video with size limit (100MB max for public use)"""
    response = requests.get(video_url, stream=True, timeout=120)
    response.raise_for_status()

    # Check file size before downloading (100MB limit)
    content_length = int(response.headers.get('content-length', 0))
    if content_length > 100 * 1024 * 1024:
        raise ValueError(
            f"Video too large ({content_length // (1024*1024)}MB). "
            "Maximum allowed size is 100MB for public use."
        )

    video_path = os.path.join(temp_dir, "video.mp4")
    downloaded = 0

    with open(video_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            # Safety: stop if exceeds 100MB
            if downloaded > 100 * 1024 * 1024:
                raise ValueError("Video exceeded 100MB limit during download.")

    return video_path


def extract_keyframes(video_path: str, max_frames: int = 4):
    """Extract keyframes — returns base64 list, display frames, duration"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    step = max(1, total_frames // max_frames)
    frame_indices = [i * step for i in range(min(max_frames, total_frames))]

    base64_frames = []
    display_frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Display version (RGB)
            display_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # API version (resized + compressed)
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)))
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            base64_frames.append(base64.b64encode(buf).decode('utf-8'))

    cap.release()
    return base64_frames, display_frames, duration


def analyze_video_with_qwen(frames: List[str]) -> str:
    """Stage 1: Qwen3.7 Plus analyzes video frames"""
    content = [{
        "type": "text",
        "text": (
            "Describe this video in 3 clear sentences. "
            "Start with 'The video shows'. "
            "Cover: subjects, actions, setting, mood."
        )
    }]
    for frame in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
        })

    response = requests.post(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        json={
            "model": "accounts/fireworks/models/qwen3p7-plus",
            "max_tokens": 120,
            "temperature": 0.3,
            "messages": [{"role": "user", "content": content}]
        },
        timeout=60
    )
    response.raise_for_status()
    desc = response.json()['choices'][0]['message']['content'].strip()

    # Clean artifacts
    desc = re.sub(r'The user.*?[\n.]', '', desc, flags=re.I)
    desc = re.sub(r'\*\*.*?\*\*', '', desc)
    desc = re.sub(r'\n+', ' ', desc)
    sentences = [s.strip() for s in re.split(r'[.!?]', desc) if len(s) > 15]
    return '. '.join(sentences[:3]) + '.'


def generate_caption_with_glm(description: str, style: str) -> str:
    """Stage 2: GLM-5.2 generates styled caption"""
    style_map = {
        "formal": (
            "Write a formal 2-3 sentence description. "
            "Use professional, academic language.",
            0.6
        ),
        "sarcastic": (
            "Write a sarcastic 2-3 sentence commentary. "
            "Use dry wit and subtle irony.",
            0.8
        ),
        "humorous_tech": (
            "Write a funny 2-3 sentence tech caption. "
            "Use programming terms: API, debug, git, deploy, code.",
            0.9
        ),
        "humorous_non_tech": (
            "Write a funny 2-3 sentence everyday caption. "
            "Relatable humor. NO tech terms whatsoever.",
            0.9
        ),
    }

    instruction, temp = style_map.get(style, ("Write a 2-3 sentence caption.", 0.7))

    response = requests.post(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        json={
            "model": "accounts/fireworks/models/glm-5p2",
            "max_tokens": 80,
            "temperature": temp,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Camera Caption, an automated API that produces "
                        "raw caption strings. Output ONLY the caption text. "
                        "No preambles, no meta-commentary, no thinking. "
                        "Start immediately with the caption."
                    )
                },
                {
                    "role": "user",
                    "content": f"VIDEO: {description}\n\n{instruction}\n\nCAPTION:"
                }
            ]
        },
        timeout=45
    )
    response.raise_for_status()
    caption = response.json()['choices'][0]['message']['content'].strip()

    # Clean artifacts
    caption = re.sub(
        r'^(CAPTION:|Caption:|Here\'s|Based on|Sure|Okay)[:.\s]*',
        '', caption, flags=re.I
    )
    caption = caption.strip('"\'')
    caption = re.sub(r'\*\*([^*]+)\*\*', r'\1', caption)
    caption = re.sub(r'\n+', ' ', caption)
    sentences = [s.strip() for s in re.split(r'[.!?]', caption) if len(s) > 10]
    if sentences:
        caption = '. '.join(sentences[:3])
        if not caption.endswith(('.', '!', '?')):
            caption += '.'
    return caption


def render_caption_card(style: str, caption: str):
    """Render a styled caption card"""
    icon = STYLE_ICONS.get(style, "📝")
    label = STYLE_LABELS.get(style, style)
    is_error = '[Error' in str(caption)

    if is_error:
        st.markdown(f"""
        <div class="error-banner">
            {icon} <strong>{label}</strong>: {caption}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="caption-card">
            <span class="style-badge badge-{style}">{icon} {label}</span>
            <p class="caption-text">{caption}</p>
        </div>
        """, unsafe_allow_html=True)


def process_single_video(video_url: str, styles: List[str], max_frames: int):
    """Full pipeline for one video — returns description + captions dict"""
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()

        # Download
        video_path = download_video(video_url, temp_dir)

        # Extract frames
        base64_frames, display_frames, duration = extract_keyframes(video_path, max_frames)

        if not base64_frames:
            raise ValueError("No keyframes could be extracted from this video.")

        # Analyze
        description = analyze_video_with_qwen(base64_frames)

        # Generate captions
        captions = {}
        for style in styles:
            captions[style] = generate_caption_with_glm(description, style)
            increment_request_count()

        return description, captions, display_frames, duration

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📷 Camera Caption")
    st.markdown("*Multi-Style AI Video Captioning*")
    st.divider()

    # API status
    st.markdown("### 🔑 API Status")
    if API_KEY:
        st.success("✅ API Connected")
    else:
        st.error("❌ API not configured")

    st.divider()

    # Style selector
    st.markdown("### 🎨 Caption Styles")
    selected_styles = st.multiselect(
        "Select styles to generate",
        options=list(STYLE_ICONS.keys()),
        default=list(STYLE_ICONS.keys()),
        format_func=lambda x: f"{STYLE_ICONS[x]} {STYLE_LABELS[x]}"
    )

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    max_frames = st.slider(
        "Keyframes per video",
        min_value=2,
        max_value=6,
        value=4,
        help="More frames = better accuracy but slower"
    )

    st.divider()

    # Pipeline
    st.markdown("### 🔄 How It Works")
    for step in [
        "📥 Download video",
        "🖼️ Extract keyframes",
        "👁️ Qwen3.7+ analyzes",
        "✍️ GLM-5.2 captions",
        "📤 Export JSON"
    ]:
        st.markdown(f'<div class="pipeline-box">{step}</div>', unsafe_allow_html=True)

    st.divider()

    # Usage counter
    remaining = MAX_REQUESTS_PER_SESSION - st.session_state["request_count"]
    st.markdown("### 📊 Session Usage")
    st.progress(
        st.session_state["request_count"] / MAX_REQUESTS_PER_SESSION,
    )
    st.caption(f"{remaining} caption generations remaining this session")

    st.divider()
    st.markdown(
        "<div style='font-size:0.8rem;color:#475569'>"
        "Built for AMD Hackathon 2025<br>"
        "🐳 riyaaisky/video-caption-agent"
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────

st.markdown('<div class="main-title">📷 Camera Caption</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Multi-Style AI Video Captioning · '
    'Qwen3.7 Plus Vision + GLM-5.2 Text</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="tagline">"One video. Four personalities. Zero effort."</div>',
    unsafe_allow_html=True
)

# Stats row
for col, num, label in zip(
    st.columns(4),
    ["2", "4", "<10m", "100MB"],
    ["AI Models", "Caption Styles", "Runtime", "Max Video Size"]
):
    with col:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{num}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# API key guard
if not API_KEY:
    st.markdown("""
    <div class="error-banner">
        ❌ <strong>Service Unavailable</strong> — API key not configured.
        Please contact the administrator.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Try It Now",
    "📁 Batch Process",
    "📊 Results Viewer",
    "❓ How To Use"
])


# ─────────────────────────────────────────────
# TAB 1 — TRY IT NOW
# ─────────────────────────────────────────────

with tab1:
    st.markdown("### 🎥 Caption Any Video Instantly")

    # Rate limit check
    if not check_rate_limit():
        st.markdown("""
        <div class="warning-banner">
            ⚠️ <strong>Session limit reached.</strong>
            You've used all 10 free generations this session.
            Please refresh the page to start a new session.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Input method
        input_method = st.radio(
            "Choose input method",
            ["⚡ Quick Test (Sample Videos)", "🔗 Paste Your Own URL"],
            horizontal=True
        )

        video_url = ""

        if input_method == "⚡ Quick Test (Sample Videos)":
            st.markdown("**Pick a sample video:**")
            cols = st.columns(len(SAMPLE_VIDEOS))
            for col, (label, url) in zip(cols, SAMPLE_VIDEOS.items()):
                with col:
                    if st.button(label, use_container_width=True):
                        st.session_state['selected_url'] = url

            if 'selected_url' in st.session_state:
                video_url = st.session_state['selected_url']
                st.info(f"📎 Selected: {video_url[:60]}...")

        else:
            video_url = st.text_input(
                "Paste a direct MP4 video URL",
                placeholder="https://example.com/video.mp4",
                help="Must be a direct link to an MP4 file (max 100MB)"
            )
            if video_url:
                st.caption("⚠️ Only direct MP4 links are supported. Max file size: 100MB.")

        # Style check
        if not selected_styles:
            st.warning("⚠️ Please select at least one caption style in the sidebar.")

        # Generate button
        generate_btn = st.button(
            "📷 Generate Captions",
            type="primary",
            use_container_width=True,
            disabled=(not video_url or not selected_styles)
        )

        if generate_btn and video_url and selected_styles:
            with st.spinner("🎬 Camera Caption is working..."):
                try:
                    progress = st.progress(0, text="📥 Downloading video...")
                    progress.progress(10, text="📥 Downloading video...")

                    temp_dir = tempfile.mkdtemp()
                    video_path = download_video(video_url, temp_dir)
                    progress.progress(30, text="🖼️ Extracting keyframes...")

                    base64_frames, display_frames, duration = extract_keyframes(
                        video_path, max_frames
                    )
                    progress.progress(50, text="👁️ Qwen3.7 analyzing scene...")

                    # Show keyframes
                    st.markdown("#### 🖼️ Extracted Keyframes")
                    frame_cols = st.columns(len(display_frames))
                    for i, (col, frame) in enumerate(zip(frame_cols, display_frames)):
                        with col:
                            st.image(frame, caption=f"Frame {i+1}", use_container_width=True)

                    # Analyze
                    description = analyze_video_with_qwen(base64_frames)
                    progress.progress(65, text="✍️ Generating styled captions...")

                    st.markdown("#### 👁️ Scene Analysis (Qwen3.7 Plus)")
                    st.info(f"**{description}**")

                    # Generate captions
                    st.markdown("#### ✍️ Camera Caption Results")
                    captions = {}
                    for i, style in enumerate(selected_styles):
                        caption = generate_caption_with_glm(description, style)
                        captions[style] = caption
                        increment_request_count()
                        render_caption_card(style, caption)
                        progress.progress(
                            65 + int((i + 1) / len(selected_styles) * 30),
                            text=f"✍️ Generated {style}..."
                        )

                    progress.progress(100, text="✅ Done!")

                    st.markdown("""
                    <div class="success-banner">
                        📷 Camera Caption — All captions generated successfully!
                    </div>
                    """, unsafe_allow_html=True)

                    # Download
                    result = {
                        "video_url": video_url,
                        "scene_description": description,
                        "captions": captions
                    }
                    st.download_button(
                        label="📥 Download as JSON",
                        data=json.dumps([result], indent=2),
                        file_name="camera_caption_results.json",
                        mime="application/json",
                        use_container_width=True
                    )

                    # Cleanup
                    shutil.rmtree(temp_dir, ignore_errors=True)

                except Exception as e:
                    st.markdown(f"""
                    <div class="error-banner">
                        ❌ <strong>Error:</strong> {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
                    shutil.rmtree(temp_dir, ignore_errors=True)

        # Rate limit notice
        st.markdown(f"""
        <div class="rate-notice">
            ℹ️ Free public demo · {MAX_REQUESTS_PER_SESSION} caption generations per session ·
            Each video uses {len(selected_styles)} generation(s)
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 2 — BATCH PROCESS
# ─────────────────────────────────────────────

with tab2:
    st.markdown("### 📁 Batch Process Multiple Videos")
    st.markdown(
        "Upload a `tasks.json` file to process multiple videos at once. "
        "Perfect for the hackathon submission format."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 📤 Upload tasks.json")
        uploaded_file = st.file_uploader(
            "Upload your tasks.json",
            type=['json'],
            help="JSON array of tasks with task_id, video_url, and styles"
        )
        if uploaded_file:
            try:
                tasks = json.load(uploaded_file)
                st.success(f"✅ Loaded {len(tasks)} tasks")
                with st.expander("Preview tasks.json"):
                    st.json(tasks)
            except Exception as e:
                st.error(f"❌ Invalid JSON: {e}")
                tasks = []

    with col_right:
        st.markdown("#### ⚡ Or Use Hackathon Test Clips")
        if st.button("📋 Load Hackathon Tasks", use_container_width=True):
            st.session_state['batch_tasks'] = [
                {
                    "task_id": "v1",
                    "video_url": SAMPLE_VIDEOS["🌆 Urban Autumn Boulevard"],
                    "styles": list(STYLE_ICONS.keys())
                },
                {
                    "task_id": "v2",
                    "video_url": SAMPLE_VIDEOS["🐱 Orange Kitten in Garden"],
                    "styles": list(STYLE_ICONS.keys())
                },
                {
                    "task_id": "v3",
                    "video_url": SAMPLE_VIDEOS["💼 Office Worker at Desk"],
                    "styles": list(STYLE_ICONS.keys())
                }
            ]
            st.success("✅ Hackathon tasks loaded!")

        st.markdown("#### 📋 Expected Format")
        st.code("""[
  {
    "task_id": "v1",
    "video_url": "https://...",
    "styles": [
      "formal",
      "sarcastic",
      "humorous_tech",
      "humorous_non_tech"
    ]
  }
]""", language="json")

    tasks_to_run = (
        tasks if (uploaded_file and tasks)
        else st.session_state.get('batch_tasks', [])
    )

    if tasks_to_run:
        total_gens = sum(len(t.get('styles', selected_styles)) for t in tasks_to_run)
        st.info(
            f"📊 {len(tasks_to_run)} videos · "
            f"{total_gens} total caption generations"
        )

        if not check_rate_limit():
            st.markdown("""
            <div class="warning-banner">
                ⚠️ Session limit reached. Refresh to start a new session.
            </div>
            """, unsafe_allow_html=True)
        else:
            run_batch = st.button(
                f"📷 Run Camera Caption on {len(tasks_to_run)} Videos",
                type="primary",
                use_container_width=True
            )

            if run_batch:
                all_results = []
                overall_progress = st.progress(0, text="Starting batch...")

                for task_idx, task in enumerate(tasks_to_run):
                    task_id = task['task_id']
                    styles = task.get('styles', selected_styles)
                    overall_progress.progress(
                        int(task_idx / len(tasks_to_run) * 100),
                        text=f"Processing {task_id} ({task_idx+1}/{len(tasks_to_run)})..."
                    )

                    result = {"task_id": task_id, "captions": {}}

                    with st.expander(f"📹 Task: {task_id}", expanded=True):
                        temp_dir = None
                        try:
                            p = st.progress(0)
                            s = st.empty()

                            s.text("📥 Downloading...")
                            temp_dir = tempfile.mkdtemp()
                            video_path = download_video(task['video_url'], temp_dir)
                            p.progress(25)

                            s.text("🖼️ Extracting frames...")
                            b64_frames, disp_frames, _ = extract_keyframes(
                                video_path, max_frames
                            )

                            fcols = st.columns(len(disp_frames))
                            for i, (c, f) in enumerate(zip(fcols, disp_frames)):
                                with c:
                                    st.image(f, caption=f"Frame {i+1}", use_container_width=True)
                            p.progress(45)

                            s.text("👁️ Analyzing with Qwen...")
                            description = analyze_video_with_qwen(b64_frames)
                            st.info(f"**Scene:** {description}")
                            p.progress(60)

                            for i, style in enumerate(styles):
                                s.text(f"✍️ Generating {style}...")
                                caption = generate_caption_with_glm(description, style)
                                result["captions"][style] = caption
                                increment_request_count()
                                render_caption_card(style, caption)
                                p.progress(60 + int((i+1)/len(styles)*35))

                            p.progress(100)
                            s.empty()
                            st.success(f"✅ Task {task_id} complete!")

                        except Exception as e:
                            st.error(f"❌ Task {task_id} failed: {e}")
                            for style in styles:
                                if style not in result["captions"]:
                                    result["captions"][style] = f"[Error: {str(e)[:80]}]"
                        finally:
                            if temp_dir:
                                shutil.rmtree(temp_dir, ignore_errors=True)

                    all_results.append(result)

                overall_progress.progress(100, text="✅ All tasks complete!")

                st.markdown("---")
                st.markdown("### 📤 Final results.json")
                results_json = json.dumps(all_results, indent=2)
                st.code(results_json, language="json")
                st.download_button(
                    label="📥 Download results.json",
                    data=results_json,
                    file_name="results.json",
                    mime="application/json",
                    use_container_width=True
                )


# ─────────────────────────────────────────────
# TAB 3 — RESULTS VIEWER
# ─────────────────────────────────────────────

with tab3:
    st.markdown("### 📊 View & Inspect Results")

    results_file = st.file_uploader(
        "Upload results.json to inspect",
        type=['json'],
        key="results_uploader"
    )

    if results_file:
        try:
            results = json.load(results_file)
            st.success(f"✅ Loaded {len(results)} results")

            total_captions = sum(len(r.get('captions', {})) for r in results)
            error_count = sum(
                1 for r in results
                for c in r.get('captions', {}).values()
                if '[Error' in str(c)
            )
            success_count = total_captions - error_count

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Videos", len(results))
            m2.metric("Total Captions", total_captions)
            m3.metric("✅ Successful", success_count)
            m4.metric("❌ Errors", error_count)

            st.divider()

            for result in results:
                task_id = result.get('task_id', 'Unknown')
                captions = result.get('captions', {})

                with st.expander(f"📷 Task: {task_id}", expanded=True):
                    for style, caption in captions.items():
                        render_caption_card(style, caption)

            # Download cleaned version
            st.download_button(
                label="📥 Re-download results.json",
                data=json.dumps(results, indent=2),
                file_name="results.json",
                mime="application/json",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ Could not parse file: {e}")
    else:
        st.info("👆 Upload a results.json file to inspect captions")
        st.markdown("#### 📋 Expected Format")
        st.code("""[
  {
    "task_id": "v1",
    "captions": {
      "formal": "The video depicts...",
      "sarcastic": "Oh wow, another...",
      "humorous_tech": "This is running on...",
      "humorous_non_tech": "This has the energy of..."
    }
  }
]""", language="json")


# ─────────────────────────────────────────────
# TAB 4 — HOW TO USE
# ─────────────────────────────────────────────

with tab4:
    st.markdown("### ❓ How to Use Camera Caption")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🚀 Quick Start")
        st.markdown("""
        <div class="howto-card">
            <div class="howto-step">
                <div class="step-num">1</div>
                <div>Go to the <strong>🚀 Try It Now</strong> tab</div>
            </div>
            <div class="howto-step">
                <div class="step-num">2</div>
                <div>Click a <strong>sample video</strong> or paste your own MP4 URL</div>
            </div>
            <div class="howto-step">
                <div class="step-num">3</div>
                <div>Select your <strong>caption styles</strong> in the sidebar</div>
            </div>
            <div class="howto-step">
                <div class="step-num">4</div>
                <div>Click <strong>📷 Generate Captions</strong></div>
            </div>
            <div class="howto-step">
                <div class="step-num">5</div>
                <div>Download your results as <strong>JSON</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎨 Caption Styles Explained")
        for style, icon in STYLE_ICONS.items():
            descriptions = {
                "formal": "Professional, objective, academic tone",
                "sarcastic": "Dry wit, ironic, lightly mocking",
                "humorous_tech": "Programming jokes, coding references",
                "humorous_non_tech": "Universal everyday humor, no tech terms"
            }
            st.markdown(f"""
            <div class="caption-card" style="margin:4px 0">
                <span class="style-badge badge-{style}">{icon} {STYLE_LABELS[style]}</span>
                <p class="caption-text" style="font-size:0.85rem">{descriptions[style]}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📁 Batch Processing")
        st.markdown("""
        <div class="howto-card">
            <div class="howto-step">
                <div class="step-num">1</div>
                <div>Go to the <strong>📁 Batch Process</strong> tab</div>
            </div>
            <div class="howto-step">
                <div class="step-num">2</div>
                <div>Upload a <strong>tasks.json</strong> file or load sample tasks</div>
            </div>
            <div class="howto-step">
                <div class="step-num">3</div>
                <div>Click <strong>Run Camera Caption</strong></div>
            </div>
            <div class="howto-step">
                <div class="step-num">4</div>
                <div>Download the final <strong>results.json</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚙️ Technical Details")
        st.markdown("""
        | Component | Detail |
        |-----------|--------|
        | Vision Model | Qwen3.7 Plus |
        | Text Model | GLM-5.2 |
        | API Provider | Fireworks AI |
        | Max Video Size | 100MB |
        | Keyframes | 2–6 per video |
        | Output Format | JSON |
        """)

        st.markdown("#### 🐳 Docker Deployment")
        st.code("docker pull riyaaisky/video-caption-agent:latest", language="bash")
        st.code("""docker run --rm \\
  -v /path/to/input:/input:ro \\
  -v /path/to/output:/output \\
  -e FIREWORKS_API_KEY=$KEY \\
  riyaaisky/video-caption-agent:latest""", language="bash")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.divider()
st.markdown("""
<div class="footer-text">
    📷 <strong>Camera Caption</strong> · Powered by
    <a href="https://fireworks.ai" style="color:#f59e0b">Fireworks AI</a> ·
    Qwen3.7 Plus + GLM-5.2 · AMD Hackathon 2025 ·
    <a href="https://github.com/Intelli2Byte/Camera_Caption"
       style="color:#f59e0b">GitHub</a> ·
    🐳 riyaaisky/video-caption-agent:latest
</div>
""", unsafe_allow_html=True)