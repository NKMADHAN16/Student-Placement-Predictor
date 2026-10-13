import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Placement Predictor",
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
# STYLING
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .main { background: #0f1420; }

    .hero {
        background: linear-gradient(135deg, #6C63FF 0%, #4834d4 55%, #341f97 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(76, 52, 212, 0.35);
    }
    .hero h1 {
        color: white;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #e2e0ff;
        font-size: 1.02rem;
        margin: 0;
    }

    .metric-card {
        background: #1a2035;
        border: 1px solid #2b3355;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }
    .metric-card h3 {
        color: #9d97ff;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .metric-card p {
        color: #ffffff;
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0;
    }

    .section-card {
        background: #1a2035;
        border: 1px solid #2b3355;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
        border-left: 4px solid #6C63FF;
        padding-left: 0.7rem;
    }

    .result-placed {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(56, 239, 125, 0.25);
    }
    .result-not-placed {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(235, 51, 73, 0.25);
    }
    .result-placed h2, .result-not-placed h2 {
        color: white;
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .result-placed p, .result-not-placed p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #6C63FF 0%, #4834d4 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.2rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 18px rgba(108, 99, 255, 0.45);
        transform: translateY(-1px);
    }

    section[data-testid="stSidebar"] {
        background: #141a2b;
        border-right: 1px solid #2b3355;
    }

    .tip-box {
        background: #232a45;
        border-left: 3px solid #ffb400;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        color: #f0d9a0;
        font-size: 0.88rem;
        margin-top: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎓 Student Placement Prediction System</h1>
    <p>Enter a student's academic and skill profile to predict campus placement outcome, powered by a Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TOP METRIC STRIP
# ----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><h3>Model Accuracy</h3><p>{metrics['accuracy']*100:.1f}%</p></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><h3>ROC-AUC</h3><p>{metrics['roc_auc']:.3f}</p></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card"><h3>F1 Score</h3><p>{metrics['f1_score']:.3f}</p></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card"><h3>Training Records</h3><p>{metrics['n_train']}</p></div>""", unsafe_allow_html=True)

st.write("")

tab1, tab2 = st.tabs(["🔮 Predict Placement", "📊 Dataset Insights"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Student Profile</div>', unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            cgpa = st.slider("CGPA", 4.0, 10.0, 7.5, 0.1)
            iq = st.slider("IQ Score", 70, 150, 100, 1)
            aptitude = st.slider("Aptitude Test Score", 20, 100, 65, 1)
            communication = st.slider("Communication Skill (1-10)", 1.0, 10.0, 6.5, 0.1)
            internships = st.selectbox("Internships Completed", [0, 1, 2, 3], index=1)
        with b:
            projects = st.selectbox("Projects Completed", [0, 1, 2, 3, 4], index=2)
            technical = st.slider("Technical Skill (1-10)", 1.0, 10.0, 6.5, 0.1)
            backlogs = st.selectbox("Active Backlogs", [0, 1, 2, 3], index=0)
            certifications = st.selectbox("Certifications", [0, 1, 2, 3, 4], index=1)
            extra = st.radio("Extra-curricular Activities", ["Yes", "No"], horizontal=True)

        st.markdown('</div>', unsafe_allow_html=True)
        predict_clicked = st.button("🚀 Predict Placement", use_container_width=True)

    with right:
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
                <div class="result-placed">
                    <h2>✅ Likely to be PLACED</h2>
                    <p>Predicted probability: {proba*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-not-placed">
                    <h2>⚠️ Placement Unlikely</h2>
                    <p>Predicted probability: {proba*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 40, "color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": "#6C63FF"},
                    "bgcolor": "#1a2035",
                    "steps": [
                        {"range": [0, 40], "color": "#eb3349"},
                        {"range": [40, 70], "color": "#f5a623"},
                        {"range": [70, 100], "color": "#38ef7d"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": proba * 100,
                    },
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                height=280,
                margin=dict(l=20, r=20, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance for this prediction context
            imp = pd.DataFrame(
                metrics["feature_importance"].items(), columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)
            fig2 = px.bar(
                imp, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale=["#4834d4", "#6C63FF", "#a29bfe"],
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                height=320,
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=30, b=10),
                title="What drives this model's decisions",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown("""
            <div class="section-card" style="text-align:center; padding: 3rem 1.5rem;">
                <div style="font-size: 3rem;">🎯</div>
                <p style="color:#b8bfe0; font-size:1rem; margin-top:0.5rem;">
                    Fill in the student profile and click <b>Predict Placement</b> to see the result here.
                </p>
                <div class="tip-box">💡 Tip: Technical skill, IQ, and communication carry the highest weight in this model.</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: DATASET INSIGHTS
# ============================================================================
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Placement Distribution</div>', unsafe_allow_html=True)
    counts = df["Placement_Status"].value_counts().rename({0: "Not Placed", 1: "Placed"})
    fig3 = px.pie(
        values=counts.values, names=counts.index, hole=0.55,
        color=counts.index,
        color_discrete_map={"Placed": "#38ef7d", "Not Placed": "#eb3349"},
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"}, height=350,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig3.update_traces(textinfo="percent+label", textfont_size=14)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">CGPA vs Placement</div>', unsafe_allow_html=True)
        fig4 = px.box(
            df, x="Placement_Status", y="CGPA", color="Placement_Status",
            color_discrete_map={0: "#eb3349", 1: "#38ef7d"},
            labels={"Placement_Status": "Placement (0=No, 1=Yes)"},
        )
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}, height=340, showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Feature Importance (Model)</div>', unsafe_allow_html=True)
        imp2 = pd.DataFrame(
            metrics["feature_importance"].items(), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        fig5 = px.bar(
            imp2, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#4834d4", "#a29bfe"],
        )
        fig5.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}, height=340, showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Raw Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=300)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write(
        "This app predicts whether a student is likely to be placed based on "
        "academic performance, aptitude, and skill profile, using a "
        "Random Forest classifier trained on historical placement data."
    )
    st.markdown("### 🧠 Model")
    st.write("Algorithm: **Random Forest Classifier**")
    st.write(f"Accuracy: **{metrics['accuracy']*100:.1f}%**")
    st.write(f"ROC-AUC: **{metrics['roc_auc']:.3f}**")
    st.markdown("---")
    st.markdown("### 📁 Dataset")
    st.write(f"Total records: **{len(df)}**")
    st.write(f"Placement rate: **{df['Placement_Status'].mean()*100:.1f}%**")
