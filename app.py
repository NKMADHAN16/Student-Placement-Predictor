import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# LOAD ARTIFACTS
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("model/placement_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/features.json") as f:
        features = json.load(f)
    with open("model/metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, features, metrics

@st.cache_data
def load_data():
    return pd.read_csv("data/placement_data.csv")

model, scaler, FEATURES, metrics = load_artifacts()
df = load_data()

# ----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------------------------------------------------
if "app_theme" not in st.session_state:
    st.session_state.app_theme = "Dark"

default_profiles = {
    "balanced": {
        "cgpa": 7.5, "iq": 102, "aptitude": 68, "comm": 6.8,
        "internships": 1, "projects": 2, "tech": 6.8, "backlogs": 0,
        "certs": 1, "extra": "Yes"
    },
    "star": {
        "cgpa": 8.9, "iq": 125, "aptitude": 88, "comm": 8.5,
        "internships": 2, "projects": 3, "tech": 8.8, "backlogs": 0,
        "certs": 3, "extra": "Yes"
    },
    "risk": {
        "cgpa": 5.8, "iq": 85, "aptitude": 45, "comm": 4.5,
        "internships": 0, "projects": 1, "tech": 4.0, "backlogs": 2,
        "certs": 0, "extra": "No"
    }
}

for k, v in default_profiles["balanced"].items():
    if f"input_{k}" not in st.session_state:
        st.session_state[f"input_{k}"] = v

def apply_profile(name):
    for k, v in default_profiles[name].items():
        st.session_state[f"input_{k}"] = v

# ----------------------------------------------------------------------------
# SIDEBAR THEME SWITCHER & CONTROLS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span style="font-size: 1.8rem;">🎓</span>
        <div>
            <div class="sidebar-brand-title">Campus Intel</div>
            <div style="font-size: 0.72rem; color: #6366f1; font-weight: 700; letter-spacing: 0.05em;">PREDICTOR SUITE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">🎨 Appearance Mode</div>', unsafe_allow_html=True)
    theme_selection = st.radio(
        "Theme Preference",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        index=0 if st.session_state.app_theme == "Dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    is_dark = "Dark" in theme_selection
    st.session_state.app_theme = "Dark" if is_dark else "Light"

    st.write("")
    st.markdown('<div class="sidebar-title">⚡ Quick Preset Profiles</div>', unsafe_allow_html=True)
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("🌟 Star Candidate", use_container_width=True):
            apply_profile("star")
            st.rerun()
    with p_col2:
        if st.button("⚖️ Average Profile", use_container_width=True):
            apply_profile("balanced")
            st.rerun()
    if st.button("⚠️ High Risk Profile", use_container_width=True):
        apply_profile("risk")
        st.rerun()

    st.write("")

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">🧠 Model Architecture</div>
        <div style="font-size: 0.88rem; font-weight: 700; margin-bottom: 0.4rem;" class="sidebar-card-headline">Random Forest Classifier</div>
        <div style="font-size: 0.8rem; line-height: 1.45;" class="sidebar-card-subtext">
            • 300 Decision Trees<br>
            • Max Depth: 8<br>
            • Balanced Class Weights<br>
            • StandardScaler Feature Norm
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-title">📊 Dataset Overview</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
            <span class="sidebar-card-subtext" style="font-size: 0.82rem;">Total Samples:</span>
            <span class="sidebar-card-accent" style="font-weight: 700; font-size: 0.82rem;">{len(df):,}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
            <span class="sidebar-card-subtext" style="font-size: 0.82rem;">Placement Rate:</span>
            <span style="color: #10b981; font-weight: 700; font-size: 0.82rem;">{df['Placement_Status'].mean()*100:.1f}%</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span class="sidebar-card-subtext" style="font-size: 0.82rem;">Evaluated Features:</span>
            <span style="color: #6366f1; font-weight: 700; font-size: 0.82rem;">10</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.72rem; margin-top: 1rem;">
        Student Placement Predictor v2.0 • Dual Theme Suite
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DYNAMIC CSS THEME GENERATOR (DARK & LIGHT)
# ----------------------------------------------------------------------------
if is_dark:
    # DARK THEME VARIABLES
    theme_css = """
    /* Dark Theme Styles */
    html, body, [class*="css"], .stApp {
        background-color: #080c14 !important;
        background: 
            radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 35%),
            radial-gradient(circle at 85% 15%, rgba(168, 85, 247, 0.10) 0%, transparent 35%),
            radial-gradient(circle at 50% 85%, rgba(14, 165, 233, 0.08) 0%, transparent 45%),
            #080c14 !important;
        color: #f1f5f9 !important;
    }

    .hero-container {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.75) 0%, rgba(15, 23, 42, 0.85) 60%, rgba(49, 46, 129, 0.5) 100%) !important;
        border: 1px solid rgba(129, 140, 248, 0.25) !important;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }
    .hero-badge {
        background: rgba(99, 102, 241, 0.18) !important;
        border: 1px solid rgba(129, 140, 248, 0.4) !important;
        color: #c7d2fe !important;
    }
    .hero-title {
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 40%, #c7d2fe 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    .hero-subtitle { color: #94a3b8 !important; }

    .metric-grid-card {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        box-shadow: 0 8px 24px -8px rgba(0, 0, 0, 0.4) !important;
    }
    .metric-grid-card:hover {
        border-color: rgba(129, 140, 248, 0.35) !important;
        box-shadow: 0 14px 30px -10px rgba(99, 102, 241, 0.25) !important;
    }
    .metric-card-title { color: #94a3b8 !important; }
    .metric-card-pill {
        background: rgba(99, 102, 241, 0.15) !important;
        color: #a5b4fc !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
    }
    .metric-card-value { color: #ffffff !important; }

    .glass-panel {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
    }
    .glass-header {
        color: #ffffff !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07) !important;
    }
    .glass-header span.icon {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #a5b4fc !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    .sidebar-brand-title { color: #ffffff !important; }
    .sidebar-card {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    .sidebar-title { color: #94a3b8 !important; }
    .sidebar-card-headline { color: #f8fafc !important; }
    .sidebar-card-subtext { color: #94a3b8 !important; }
    .sidebar-card-accent { color: #ffffff !important; }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(129, 140, 248, 0.35) !important;
    }

    .empty-state-box {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px dashed rgba(255, 255, 255, 0.12) !important;
    }
    .empty-state-text { color: #94a3b8 !important; }

    .result-card-placed {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.7) 0%, rgba(6, 95, 70, 0.45) 50%, rgba(4, 120, 87, 0.3) 100%) !important;
        border: 1px solid rgba(52, 211, 153, 0.4) !important;
    }
    .result-card-unlikely {
        background: linear-gradient(135deg, rgba(136, 19, 55, 0.7) 0%, rgba(159, 18, 57, 0.45) 50%, rgba(190, 18, 60, 0.3) 100%) !important;
        border: 1px solid rgba(251, 113, 133, 0.4) !important;
    }
    .result-badge-placed {
        background: rgba(16, 185, 129, 0.25) !important;
        color: #6ee7b7 !important;
        border: 1px solid rgba(110, 231, 183, 0.4) !important;
    }
    .result-badge-unlikely {
        background: rgba(244, 63, 94, 0.25) !important;
        color: #fda4af !important;
        border: 1px solid rgba(253, 164, 175, 0.4) !important;
    }
    .result-heading { color: #ffffff !important; }
    .result-subtext { color: rgba(255, 255, 255, 0.85) !important; }

    div[data-testid="stSlider"] > label, div[data-testid="stSelectbox"] > label, div[data-testid="stRadio"] > label {
        color: #cbd5e1 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }

    ul[data-baseweb="menu"] {
        background-color: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    ul[data-baseweb="menu"] li {
        color: #f8fafc !important;
    }
    """
    chart_font_color = "#f8fafc"
    chart_tick_color = "#94a3b8"
    chart_grid_color = "rgba(255,255,255,0.06)"
    gauge_bg = "rgba(15, 23, 42, 0.7)"
    gauge_num_color = "#ffffff"
    pie_border_color = "#080c14"
    box_plot_bg = "rgba(0,0,0,0)"

else:
    # LIGHT THEME VARIABLES
    theme_css = """
    /* Light Theme Styles */
    html, body, [class*="css"], .stApp {
        background-color: #f8fafc !important;
        background: 
            radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 85% 15%, rgba(168, 85, 247, 0.06) 0%, transparent 40%),
            radial-gradient(circle at 50% 85%, rgba(14, 165, 233, 0.05) 0%, transparent 45%),
            #f8fafc !important;
        color: #0f172a !important;
    }

    .hero-container {
        background: linear-gradient(135deg, #3730a3 0%, #4f46e5 55%, #6366f1 100%) !important;
        border: 1px solid rgba(79, 70, 229, 0.3) !important;
        box-shadow: 0 16px 36px -10px rgba(79, 70, 229, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.18) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        color: #ffffff !important;
    }
    .hero-title {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: #ffffff !important;
        color: #ffffff !important;
    }
    .hero-subtitle { color: #e0e7ff !important; }

    .metric-grid-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 6px 20px -6px rgba(15, 23, 42, 0.07) !important;
    }
    .metric-grid-card:hover {
        border-color: #818cf8 !important;
        box-shadow: 0 12px 28px -6px rgba(99, 102, 241, 0.2) !important;
    }
    .metric-card-title { color: #64748b !important; }
    .metric-card-pill {
        background: #eef2ff !important;
        color: #4f46e5 !important;
        border: 1px solid #c7d2fe !important;
    }
    .metric-card-value { color: #0f172a !important; }

    .glass-panel {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 8px 28px -6px rgba(15, 23, 42, 0.06) !important;
    }
    .glass-header {
        color: #0f172a !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    .glass-header span.icon {
        background: #eef2ff !important;
        color: #4f46e5 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    .sidebar-brand-title { color: #0f172a !important; }
    .sidebar-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    .sidebar-title { color: #64748b !important; }
    .sidebar-card-headline { color: #0f172a !important; }
    .sidebar-card-subtext { color: #475569 !important; }
    .sidebar-card-accent { color: #0f172a !important; }

    .stTabs [data-baseweb="tab-list"] {
        background: #e2e8f0 !important;
        border: 1px solid #cbd5e1 !important;
    }
    .stTabs [data-baseweb="tab"] { color: #64748b !important; }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #4f46e5 !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.06) !important;
    }

    .empty-state-box {
        background: #ffffff !important;
        border: 1px dashed #cbd5e1 !important;
    }
    .empty-state-text { color: #64748b !important; }

    .result-card-placed {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%) !important;
        border: 1px solid #34d399 !important;
        box-shadow: 0 10px 30px -5px rgba(16, 185, 129, 0.25) !important;
    }
    .result-card-placed .result-heading { color: #065f46 !important; }
    .result-card-placed .result-subtext { color: #047857 !important; }
    .result-badge-placed {
        background: #d1fae5 !important;
        color: #047857 !important;
        border: 1px solid #6ee7b7 !important;
    }

    .result-card-unlikely {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%) !important;
        border: 1px solid #fb7185 !important;
        box-shadow: 0 10px 30px -5px rgba(244, 63, 94, 0.25) !important;
    }
    .result-card-unlikely .result-heading { color: #9f1239 !important; }
    .result-card-unlikely .result-subtext { color: #be123c !important; }
    .result-badge-unlikely {
        background: #ffe4e6 !important;
        color: #be123c !important;
        border: 1px solid #fda4af !important;
    }

    div[data-testid="stSlider"] > label, div[data-testid="stSelectbox"] > label, div[data-testid="stRadio"] > label {
        color: #334155 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-color: #cbd5e1 !important;
    }

    ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
    }

    ul[data-baseweb="menu"] li {
        color: #0f172a !important;
    }

    div[data-baseweb="radio"] label {
        color: #1e293b !important;
    }
    div[data-baseweb="radio"] div {
        color: #1e293b !important;
    }
    """
    chart_font_color = "#0f172a"
    chart_tick_color = "#475569"
    chart_grid_color = "rgba(0,0,0,0.07)"
    gauge_bg = "rgba(241, 245, 249, 0.95)"
    gauge_num_color = "#0f172a"
    pie_border_color = "#ffffff"
    box_plot_bg = "rgba(0,0,0,0)"

# ----------------------------------------------------------------------------
# CORE STRUCTURAL CSS (COMMON TO BOTH THEMES)
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@600;700;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    
    /* Hero Banner */
    .hero-container {{
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        padding: 2.3rem 2.7rem;
        margin-bottom: 1.8rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }}
    
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.95rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }}
    
    .hero-title {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.35rem !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        margin: 0 0 0.55rem 0 !important;
    }}
    
    .hero-subtitle {{
        font-size: 1.02rem;
        font-weight: 400;
        max-width: 820px;
        margin: 0;
        line-height: 1.5;
    }}

    /* Metric Cards */
    .metric-grid-card {{
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        text-align: left;
        position: relative;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }}
    
    .metric-grid-card:hover {{
        transform: translateY(-2px);
    }}

    .metric-label-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }}
    
    .metric-card-title {{
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0;
    }}
    
    .metric-card-pill {{
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
    }}

    .metric-card-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.95rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
        letter-spacing: -0.02em;
    }}

    /* Panels */
    .glass-panel {{
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.4rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }}

    .glass-header {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
    }}
    
    .glass-header span.icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        font-size: 1rem;
    }}

    /* Result Banners */
    .result-card-placed, .result-card-unlikely {{
        border-radius: 20px;
        padding: 1.8rem;
        text-align: center;
        backdrop-filter: blur(16px);
        margin-bottom: 1.2rem;
    }}

    .result-badge {{
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
    }}

    .result-heading {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
    }}

    .result-subtext {{
        font-size: 1.02rem;
        margin: 0;
    }}

    /* Empty state */
    .empty-state-box {{
        text-align: center;
        padding: 3.5rem 2rem;
        border-radius: 20px;
    }}

    .empty-state-icon {{
        font-size: 3.2rem;
        margin-bottom: 0.8rem;
        filter: drop-shadow(0 6px 12px rgba(99, 102, 241, 0.25));
    }}

    .empty-state-text {{
        font-size: 1rem;
        max-width: 380px;
        margin: 0 auto 1rem auto;
    }}

    .tip-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #d97706;
        font-size: 0.82rem;
        padding: 0.4rem 0.9rem;
        border-radius: 12px;
        font-weight: 600;
    }}

    /* CTA Button Glow */
    .stButton > button {{
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.8rem 1.6rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.45) !important;
        transition: all 0.25s ease !important;
        width: 100% !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 30px -5px rgba(139, 92, 246, 0.55) !important;
        color: #ffffff !important;
    }}

    /* Sidebar Base */
    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 1.2rem;
    }}

    .sidebar-brand-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0;
    }}

    .sidebar-card {{
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }}
    
    .sidebar-title {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.6rem;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        padding: 0.35rem;
        border-radius: 14px;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: auto;
        padding: 0.6rem 1.4rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        border: none !important;
        background: transparent;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}

    {theme_css}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HERO SECTION
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span>✦</span> AI-POWERED RECRUITMENT INTELLIGENCE
    </div>
    <h1 class="hero-title">Student Placement Predictor</h1>
    <p class="hero-subtitle">
        Assess student placement probability based on multi-dimensional academic metrics, technical competence, and cognitive factors powered by an ensemble Random Forest model.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TOP METRICS ROW
# ----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-grid-card">
        <div class="metric-label-row">
            <span class="metric-card-title">Model Accuracy</span>
            <span class="metric-card-pill">Test Set</span>
        </div>
        <div class="metric-card-value" style="color: #6366f1;">{metrics['accuracy']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-grid-card">
        <div class="metric-label-row">
            <span class="metric-card-title">ROC-AUC Score</span>
            <span class="metric-card-pill">Discriminative</span>
        </div>
        <div class="metric-card-value" style="color: #0284c7;">{metrics['roc_auc']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-grid-card">
        <div class="metric-label-row">
            <span class="metric-card-title">F1-Score</span>
            <span class="metric-card-pill">Balanced</span>
        </div>
        <div class="metric-card-value" style="color: #059669;">{metrics['f1_score']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-grid-card">
        <div class="metric-label-row">
            <span class="metric-card-title">Trained Samples</span>
            <span class="metric-card-pill">Stratified</span>
        </div>
        <div class="metric-card-value" style="color: #db2777;">{metrics['n_train']}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# ----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Real-Time Prediction", "📊 Dataset Analytics & Explanations"])

# ============================================================================
# TAB 1: REAL-TIME PREDICTION
# ============================================================================
with tab1:
    col_input, col_result = st.columns([1.18, 1], gap="large")

    with col_input:
        st.markdown("""
        <div class="glass-panel">
            <div class="glass-header">
                <span class="icon">👤</span>
                <span>Student Academic & Skill Profile</span>
            </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<p style='color:#6366f1; font-weight:700; font-size:0.8rem; text-transform:uppercase; margin-bottom:0.2rem;'>Academics & Cognitive</p>", unsafe_allow_html=True)
            cgpa = st.slider("CGPA (Cumulative GPA)", 4.0, 10.0, float(st.session_state.input_cgpa), 0.1, key="slider_cgpa")
            iq = st.slider("IQ Score", 70, 150, int(st.session_state.input_iq), 1, key="slider_iq")
            aptitude = st.slider("Aptitude Exam Score (0-100)", 20, 100, int(st.session_state.input_aptitude), 1, key="slider_aptitude")
            communication = st.slider("Communication Skill (1-10)", 1.0, 10.0, float(st.session_state.input_comm), 0.1, key="slider_comm")
            internships = st.selectbox("Internships Completed", [0, 1, 2, 3], index=int(st.session_state.input_internships), key="select_internships")

        with col_b:
            st.markdown("<p style='color:#6366f1; font-weight:700; font-size:0.8rem; text-transform:uppercase; margin-bottom:0.2rem;'>Technical & Practical</p>", unsafe_allow_html=True)
            technical = st.slider("Technical Skill Rating (1-10)", 1.0, 10.0, float(st.session_state.input_tech), 0.1, key="slider_tech")
            projects = st.selectbox("Projects Completed", [0, 1, 2, 3, 4], index=int(st.session_state.input_projects), key="select_projects")
            backlogs = st.selectbox("Active Backlogs", [0, 1, 2, 3], index=int(st.session_state.input_backlogs), key="select_backlogs")
            certifications = st.selectbox("Certifications Earned", [0, 1, 2, 3, 4], index=int(st.session_state.input_certs), key="select_certs")
            extra = st.radio("Extracurricular Engagement", ["Yes", "No"], index=0 if st.session_state.input_extra == "Yes" else 1, horizontal=True, key="radio_extra")

        st.markdown('</div>', unsafe_allow_html=True)

        predict_clicked = st.button("✨ Evaluate Placement Readiness", use_container_width=True)

    with col_result:
        if predict_clicked:
            input_dict = {
                "CGPA": cgpa,
                "IQ": iq,
                "Aptitude_Score": aptitude,
                "Communication_Skill": communication,
                "Internships": internships,
                "Projects": projects,
                "Technical_Skill": technical,
                "Backlogs": backlogs,
                "Extra_Curricular": 1 if extra == "Yes" else 0,
                "Certifications": certifications,
            }
            X_input = pd.DataFrame([input_dict])[FEATURES]
            X_scaled = scaler.transform(X_input)
            proba = model.predict_proba(X_scaled)[0][1]
            pred = model.predict(X_scaled)[0]

            if pred == 1:
                st.markdown(f"""
                <div class="result-card-placed">
                    <span class="result-badge result-badge-placed">High Probability Outcome</span>
                    <h2 class="result-heading">✅ Likely to be Placed</h2>
                    <p class="result-subtext">Estimated Placement Probability: <b>{proba*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-unlikely">
                    <span class="result-badge result-badge-unlikely">Risk Detected</span>
                    <h2 class="result-heading">⚠️ Placement Unlikely</h2>
                    <p class="result-subtext">Estimated Placement Probability: <b>{proba*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

            # Adaptive Gauge Indicator
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 42, "color": gauge_num_color, "family": "Space Grotesk"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": chart_tick_color, "tickfont": {"size": 12, "color": chart_tick_color}},
                    "bar": {"color": "#6366f1", "thickness": 0.28},
                    "bgcolor": gauge_bg,
                    "borderwidth": 1,
                    "bordercolor": "rgba(148, 163, 184, 0.25)",
                    "steps": [
                        {"range": [0, 45], "color": "rgba(244, 63, 94, 0.45)"},
                        {"range": [45, 70], "color": "rgba(245, 158, 11, 0.45)"},
                        {"range": [70, 100], "color": "rgba(16, 185, 129, 0.5)"},
                    ],
                    "threshold": {
                        "line": {"color": chart_font_color, "width": 4},
                        "thickness": 0.85,
                        "value": proba * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": chart_font_color, "family": "Plus Jakarta Sans"},
                height=260,
                margin=dict(l=30, r=30, t=25, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Relative Impact Analysis for this Profile
            st.markdown(f"""
            <div style="font-weight: 700; color: {chart_font_color}; font-size: 0.95rem; margin-top: 0.5rem; margin-bottom: 0.4rem;">
                🧠 Key Driving Factors in Decision
            </div>
            """, unsafe_allow_html=True)

            imp = pd.DataFrame(
                metrics["feature_importance"].items(), columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)

            fig_bar = px.bar(
                imp.tail(6), x="Importance", y="Feature", orientation="h",
                color="Importance",
                color_continuous_scale=["#4338ca", "#6366f1", "#a855f7"],
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": chart_font_color, "family": "Plus Jakarta Sans"},
                height=230,
                showlegend=False,
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor=chart_grid_color, title="Model Weight", tickfont=dict(color=chart_tick_color)),
                yaxis=dict(title="", tickfont=dict(color=chart_font_color)),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.markdown(f"""
            <div class="empty-state-box">
                <div class="empty-state-icon">🔮</div>
                <h3 style="color:{chart_font_color}; font-family:'Space Grotesk',sans-serif; margin-bottom:0.4rem;">Ready for Inference</h3>
                <p class="empty-state-text">
                    Configure the student's credentials on the left and click <b>Evaluate Placement Readiness</b> to generate an AI placement probability and feature impact breakdown.
                </p>
                <div class="tip-pill">
                    <span>💡</span> <b>Pro Tip:</b> Try preset profiles in the sidebar for quick demonstrations.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: DATASET ANALYTICS & INSIGHTS
# ============================================================================
with tab2:
    row1_c1, row1_c2 = st.columns([1, 1], gap="large")

    with row1_c1:
        st.markdown("""
        <div class="glass-panel">
            <div class="glass-header">
                <span class="icon">🍩</span>
                <span>Placement Outcome Distribution</span>
            </div>
        """, unsafe_allow_html=True)

        counts = df["Placement_Status"].value_counts().rename({0: "Not Placed", 1: "Placed"})
        fig_pie = px.pie(
            values=counts.values,
            names=counts.index,
            hole=0.62,
            color=counts.index,
            color_discrete_map={"Placed": "#10b981", "Not Placed": "#f43f5e"},
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": chart_font_color, "family": "Plus Jakarta Sans"},
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5, font=dict(color=chart_font_color)),
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            textfont=dict(size=14, family="Plus Jakarta Sans", color="white"),
            marker=dict(line=dict(color=pie_border_color, width=3)),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_c2:
        st.markdown("""
        <div class="glass-panel">
            <div class="glass-header">
                <span class="icon">📊</span>
                <span>CGPA vs. Placement Distribution</span>
            </div>
        """, unsafe_allow_html=True)

        fig_box = px.box(
            df,
            x="Placement_Status",
            y="CGPA",
            color="Placement_Status",
            color_discrete_map={0: "#f43f5e", 1: "#10b981"},
            labels={"Placement_Status": "Placement Outcome (0 = Unplaced, 1 = Placed)"},
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": chart_font_color, "family": "Plus Jakarta Sans"},
            height=320,
            showlegend=False,
            xaxis=dict(showgrid=False, tickfont=dict(color=chart_font_color)),
            yaxis=dict(showgrid=True, gridcolor=chart_grid_color, tickfont=dict(color=chart_font_color)),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature Importance (Full Model)
    st.markdown("""
    <div class="glass-panel">
        <div class="glass-header">
            <span class="icon">🏆</span>
            <span>Overall Random Forest Feature Importance</span>
        </div>
    """, unsafe_allow_html=True)

    imp_full = pd.DataFrame(
        metrics["feature_importance"].items(), columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=True)

    fig_full_bar = px.bar(
        imp_full,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#312e81", "#6366f1", "#a855f7", "#ec4899"],
    )
    fig_full_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": chart_font_color, "family": "Plus Jakarta Sans"},
        height=350,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor=chart_grid_color, title="Relative Importance Weight", tickfont=dict(color=chart_tick_color)),
        yaxis=dict(title="", tickfont=dict(color=chart_font_color)),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_full_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Raw Data Explorer
    st.markdown("""
    <div class="glass-panel">
        <div class="glass-header">
            <span class="icon">📁</span>
            <span>Dataset Browser (First 50 Student Records)</span>
        </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=290)
    st.markdown('</div>', unsafe_allow_html=True)
