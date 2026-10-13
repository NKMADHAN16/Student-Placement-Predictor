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
# ADVANCED CUSTOM THEME & STYLES (GLASSMORPHISM & CYBER-ACCENTS)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #080c14;
        color: #f1f5f9;
    }

    /* Ambient Background Glow */
    .stApp {
        background: 
            radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.12) 0%, transparent 35%),
            radial-gradient(circle at 85% 15%, rgba(168, 85, 247, 0.10) 0%, transparent 35%),
            radial-gradient(circle at 50% 85%, rgba(14, 165, 233, 0.08) 0%, transparent 45%),
            #080c14;
    }

    /* Remove default Streamlit top decoration line */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Hero Banner */
    .hero-container {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.75) 0%, rgba(15, 23, 42, 0.85) 60%, rgba(49, 46, 129, 0.5) 100%);
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 24px;
        padding: 2.4rem 2.8rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(129, 140, 248, 0.4);
        padding: 0.35rem 0.95rem;
        border-radius: 9999px;
        color: #c7d2fe;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    
    .hero-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        margin: 0 0 0.6rem 0 !important;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 40%, #c7d2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        max-width: 820px;
        margin: 0;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-grid-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        text-align: left;
        position: relative;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px -8px rgba(0, 0, 0, 0.4);
    }
    
    .metric-grid-card:hover {
        border-color: rgba(129, 140, 248, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 14px 30px -10px rgba(99, 102, 241, 0.25);
    }

    .metric-label-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .metric-card-title {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0;
    }
    
    .metric-card-pill {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    .metric-card-value {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.95rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
        letter-spacing: -0.02em;
    }

    /* Glass Panels */
    .glass-panel {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.4rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .glass-header {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }
    
    .glass-header span.icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        font-size: 1rem;
    }

    /* Result Banners */
    .result-card-placed {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.7) 0%, rgba(6, 95, 70, 0.45) 50%, rgba(4, 120, 87, 0.3) 100%);
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 20px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 12px 35px -8px rgba(16, 185, 129, 0.35);
        backdrop-filter: blur(16px);
        margin-bottom: 1.2rem;
    }
    
    .result-card-unlikely {
        background: linear-gradient(135deg, rgba(136, 19, 55, 0.7) 0%, rgba(159, 18, 57, 0.45) 50%, rgba(190, 18, 60, 0.3) 100%);
        border: 1px solid rgba(251, 113, 133, 0.4);
        border-radius: 20px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 12px 35px -8px rgba(244, 63, 94, 0.35);
        backdrop-filter: blur(16px);
        margin-bottom: 1.2rem;
    }

    .result-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
    }
    .result-badge-placed {
        background: rgba(16, 185, 129, 0.25);
        color: #6ee7b7;
        border: 1px solid rgba(110, 231, 183, 0.4);
    }
    .result-badge-unlikely {
        background: rgba(244, 63, 94, 0.25);
        color: #fda4af;
        border: 1px solid rgba(253, 164, 175, 0.4);
    }

    .result-heading {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
    }

    .result-subtext {
        font-size: 1.02rem;
        color: rgba(255, 255, 255, 0.85);
        margin: 0;
    }

    /* Prompt Box for Empty State */
    .empty-state-box {
        text-align: center;
        padding: 3.5rem 2rem;
        background: rgba(15, 23, 42, 0.5);
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 20px;
    }

    .empty-state-icon {
        font-size: 3.2rem;
        margin-bottom: 0.8rem;
        filter: drop-shadow(0 6px 12px rgba(99, 102, 241, 0.3));
    }

    .empty-state-text {
        color: #94a3b8;
        font-size: 1rem;
        max-width: 380px;
        margin: 0 auto 1rem auto;
    }

    .tip-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #fde68a;
        font-size: 0.82rem;
        padding: 0.4rem 0.9rem;
        border-radius: 12px;
    }

    /* CTA Button Glow */
    .stButton > button {
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
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 30px -5px rgba(139, 92, 246, 0.6) !important;
        color: #ffffff !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.2rem;
    }

    .sidebar-brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }

    .sidebar-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    
    .sidebar-title {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.6rem;
    }

    /* Custom Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.5);
        padding: 0.35rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 0.6rem 1.4rem;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.95rem;
        border: none !important;
        background: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: 1px solid rgba(129, 140, 248, 0.35) !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Sliders & Inputs Subtle Styling */
    div[data-testid="stSlider"] > label, div[data-testid="stSelectbox"] > label, div[data-testid="stRadio"] > label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION FOR PRESET PROFILES
# ----------------------------------------------------------------------------
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
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span style="font-size: 1.8rem;">🎓</span>
        <div>
            <div class="sidebar-brand-title">Campus Intel</div>
            <div style="font-size: 0.72rem; color: #818cf8; font-weight: 600;">PREDICTOR SUITE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        <div style="font-size: 0.88rem; color: #f8fafc; font-weight: 600; margin-bottom: 0.4rem;">Random Forest Classifier</div>
        <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.4;">
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
            <span style="color: #94a3b8; font-size: 0.82rem;">Total Samples:</span>
            <span style="color: #ffffff; font-weight: 700; font-size: 0.82rem;">{len(df):,}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
            <span style="color: #94a3b8; font-size: 0.82rem;">Placement Rate:</span>
            <span style="color: #34d399; font-weight: 700; font-size: 0.82rem;">{df['Placement_Status'].mean()*100:.1f}%</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94a3b8; font-size: 0.82rem;">Evaluated Features:</span>
            <span style="color: #a5b4fc; font-weight: 700; font-size: 0.82rem;">10</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.72rem; margin-top: 1rem;">
        Student Placement Predictor v2.0 • Powered by Streamlit & Scikit-learn
    </div>
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
        <div class="metric-card-value" style="color: #818cf8;">{metrics['accuracy']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-grid-card">
        <div class="metric-label-row">
            <span class="metric-card-title">ROC-AUC Score</span>
            <span class="metric-card-pill">Discriminative</span>
        </div>
        <div class="metric-card-value" style="color: #38bdf8;">{metrics['roc_auc']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-grid-card">
        <div class="metric-label-row">
            <span class="metric-card-title">F1-Score</span>
            <span class="metric-card-pill">Balanced</span>
        </div>
        <div class="metric-card-value" style="color: #34d399;">{metrics['f1_score']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-grid-card">
        <div class="metric-label-row">
            <span class="metric-card-title">Trained Samples</span>
            <span class="metric-card-pill">Stratified</span>
        </div>
        <div class="metric-card-value" style="color: #f472b6;">{metrics['n_train']}</div>
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
            st.markdown("<p style='color:#a5b4fc; font-weight:700; font-size:0.8rem; text-transform:uppercase; margin-bottom:0.2rem;'>Academics & Cognitive</p>", unsafe_allow_html=True)
            cgpa = st.slider("CGPA (Cumulative GPA)", 4.0, 10.0, float(st.session_state.input_cgpa), 0.1, key="slider_cgpa")
            iq = st.slider("IQ Score", 70, 150, int(st.session_state.input_iq), 1, key="slider_iq")
            aptitude = st.slider("Aptitude Exam Score (0-100)", 20, 100, int(st.session_state.input_aptitude), 1, key="slider_aptitude")
            communication = st.slider("Communication Skill (1-10)", 1.0, 10.0, float(st.session_state.input_comm), 0.1, key="slider_comm")
            internships = st.selectbox("Internships Completed", [0, 1, 2, 3], index=int(st.session_state.input_internships), key="select_internships")

        with col_b:
            st.markdown("<p style='color:#a5b4fc; font-weight:700; font-size:0.8rem; text-transform:uppercase; margin-bottom:0.2rem;'>Technical & Practical</p>", unsafe_allow_html=True)
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

            # Modernized Gauge Indicator
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 42, "color": "#ffffff", "family": "Space Grotesk"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8", "tickfont": {"size": 12, "color": "#94a3b8"}},
                    "bar": {"color": "#6366f1", "thickness": 0.28},
                    "bgcolor": "rgba(15, 23, 42, 0.7)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255, 255, 255, 0.1)",
                    "steps": [
                        {"range": [0, 45], "color": "rgba(244, 63, 94, 0.45)"},
                        {"range": [45, 70], "color": "rgba(245, 158, 11, 0.45)"},
                        {"range": [70, 100], "color": "rgba(16, 185, 129, 0.5)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 4},
                        "thickness": 0.85,
                        "value": proba * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f8fafc", "family": "Plus Jakarta Sans"},
                height=260,
                margin=dict(l=30, r=30, t=25, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Relative Impact Analysis for this Profile
            st.markdown("""
            <div style="font-weight: 700; color: #e2e8f0; font-size: 0.95rem; margin-top: 0.5rem; margin-bottom: 0.4rem;">
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
                font={"color": "#cbd5e1", "family": "Plus Jakarta Sans"},
                height=230,
                showlegend=False,
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Model Weight"),
                yaxis=dict(title="", tickfont=dict(color="#e2e8f0")),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.markdown("""
            <div class="empty-state-box">
                <div class="empty-state-icon">🔮</div>
                <h3 style="color:#ffffff; font-family:'Space Grotesk',sans-serif; margin-bottom:0.4rem;">Ready for Inference</h3>
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
            font={"color": "#ffffff", "family": "Plus Jakarta Sans"},
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            textfont=dict(size=14, family="Plus Jakarta Sans", color="white"),
            marker=dict(line=dict(color="#0f172a", width=3)),
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
            font={"color": "#cbd5e1", "family": "Plus Jakarta Sans"},
            height=320,
            showlegend=False,
            xaxis=dict(showgrid=False, tickfont=dict(color="#f8fafc")),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#f8fafc")),
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
        font={"color": "#cbd5e1", "family": "Plus Jakarta Sans"},
        height=350,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Relative Importance Weight"),
        yaxis=dict(title="", tickfont=dict(color="#f8fafc")),
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
