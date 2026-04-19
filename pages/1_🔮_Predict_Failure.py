"""
Predict Failure -- FailSafe AI
Enter sensor parameters and get instant failure predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from utils import load_config, pickle_load, resolve_path

st.set_page_config(page_title="Predict Failure | FailSafe AI", page_icon="🛡️", layout="wide")

# Load CNC image as base64 for CSS background
@st.cache_data(show_spinner=False)
def get_bg_image():
    img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "cnc_machine.png")
    if not os.path.exists(img_path):
        return None
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return data

bg_b64 = get_bg_image()

# Premium Enterprise CSS — ULTRA-COMPACT + HIGHER FONTS
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
  .stApp {{
    background: linear-gradient(135deg, #0a0a0a 0%, #171717 50%, #1f1f1f 100%);
  }}
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #000000 0%, #0f0f0f 100%);
    border-right: 1px solid #2a2a2a;
  }}
  #MainMenu {{visibility: hidden;}}
  footer {{visibility: hidden;}}

  /* CNC background — RECOGNIZABLE (reduced blur, increased brightness) */
  [data-testid="stMain"]::before {{
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: url("data:image/png;base64,{bg_b64}") center center / cover no-repeat;
    filter: blur(1.5px) brightness(0.35);
    opacity: 0.65;
    z-index: 0;
    pointer-events: none;
  }}

  [data-testid="stMain"] > div {{
    position: relative;
    z-index: 1;
  }}

  /* COMPACT FONT INCREASES */
  .enterprise-header .title {{
    font-size: 1.8rem !important; /* Increased from 1.4 */
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
  }}
  .enterprise-header .subtitle {{
    font-size: 1rem !important; /* Increased from 0.82 */
    color: #e0e0e0;
    margin-top: 0.1rem;
  }}

  /* ULTRA-COMPACT UI ELEMENTS */
  [data-testid="stSlider"] {{
    margin-bottom: -18px !important; /* Even more compact */
  }}
  [data-testid="stSlider"] label {{
    font-size: 0.95rem !important; /* Increased font */
    font-weight: 600 !important;
    color: #ffffff !important;
    margin-bottom: -2px !important;
  }}

  .stSelectbox label {{
    font-size: 0.95rem !important;
    font-weight: 600 !important;
  }}

  .section-label {{
    font-size: 0.8rem !important;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
    padding-bottom: 0.2rem;
    border-bottom: 2px solid #ff4b4b;
    display: inline-block;
  }}

  /* PANEL SPACING */
  .glass-panel {{
    background: rgba(10, 10, 10, 0.65);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
  }}

  .risk-critical, .risk-high, .risk-medium, .risk-low {{
    padding: 0.8rem !important;
    border-radius: 8px !important;
  }}

  .result-card {{
    padding: 0.6rem 0.8rem !important;
    margin: 0.2rem 0 !important;
  }}

  .block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 0.5rem !important;
  }}

  /* HIDE EXTRA SPACING */
  div[data-testid="stVerticalBlock"] > div {{
    gap: 0.5rem !important;
  }}
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource(show_spinner=False)
def load_models():
    config = load_config()
    best_model = pickle_load(config["models"]["directory"] + config["models"]["best_model"])
    ft_model = pickle_load(config["models"]["directory"] + config["models"]["failure_type_model"])
    ft_metrics = pickle_load(config["models"]["directory"] + config["models"]["failure_type_metrics"])
    return best_model, ft_model, ft_metrics, config

best_model, ft_model, ft_metrics, config = load_models()

# Enterprise Header
st.markdown("""
<div class='enterprise-header'>
  <div class='icon'>🛡️</div>
  <div>
    <div class='title'>Failure Prediction Engine</div>
    <div class='subtitle'>High-Fidelity Machine Hardware Lifecycle Diagnostics</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Main Grid layout to avoid scrolling
col_input, col_result = st.columns([1.1, 1], gap="medium")

with col_input:
    st.markdown("<div class='section-label'>Sensor Configuration</div>", unsafe_allow_html=True)
    
    # Selection
    m_type = st.selectbox("Operating Grade", options=["L", "M", "H"], 
                           help="L=Low (50%), M=Medium (30%), H=High (20%)", index=0)

    # 2 Column Sliders for ultimate compactness
    s1, s2 = st.columns(2)
    with s1:
        air_temp = st.slider("Air Temp (K)", 295.3, 304.5, 300.0, 0.1)
        proc_temp = st.slider("Process Temp (K)", 305.7, 313.8, 310.0, 0.1)
        rot_speed = st.slider("Rot. Speed (rpm)", 1168, 2886, 1500, 10)
    with s2:
        torque = st.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.1)
        tool_wear = st.slider("Tool Wear (min)", 0, 253, 100, 1)

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🛡️  RUN REAL-TIME DIAGNOSTIC", use_container_width=True, type="primary")

with col_result:
    st.markdown("<div class='section-label'>Diagnostic Result</div>", unsafe_allow_html=True)

    if predict_btn:
        # Features & Engineering
        type_encoded = config["dataset"]["type_encoding"][m_type]
        power = torque * rot_speed * (2 * np.pi / 60)
        temp_diff = proc_temp - air_temp
        overstrain = tool_wear * torque

        features = pd.DataFrame([{
            "Type": type_encoded, "Air_Temp_K": air_temp,
            "Process_Temp_K": proc_temp, "Rotational_Speed_rpm": rot_speed,
            "Torque_Nm": torque, "Tool_Wear_min": tool_wear,
            "Power_W": power, "Temp_Diff_K": temp_diff,
            "Overstrain_Indicator": overstrain,
        }])

        prob = best_model.predict_proba(features)[0][1]
        pred = best_model.predict(features)[0]
        ft_pred_enc = ft_model.predict(features)[0]
        le = ft_metrics["label_encoder"]
        ft_pred = le.inverse_transform([ft_pred_enc])[0]

        # Risk UI
        if prob >= 0.75: r_l, r_c, r_i = "CRITICAL", "risk-critical", "🔴"
        elif prob >= 0.50: r_l, r_c, r_i = "HIGH", "risk-high", "🟠"
        elif prob >= 0.25: r_l, r_c, r_i = "MEDIUM", "risk-medium", "🟡"
        else: r_l, r_c, r_i = "LOW", "risk-low", "🟢"

        st.markdown(f"""
        <div class='{r_c}'>
          <div style='font-size: 1.2rem; font-weight: 800;'>{r_i} {r_l} RISK DETECTED</div>
          <div style='font-size: 0.85rem; opacity: 0.9;'>Failure Probability: {prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""<div class='result-card'><div style='font-size:0.6rem; color:#8b949e;'>STATUS</div>
            <div style='font-size:0.95rem; font-weight:700; color:{"#f85149" if pred==1 else "#3fb950"};'>{"FAILURE" if pred==1 else "NORMAL"}</div></div>""", 
            unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class='result-card'><div style='font-size:0.6rem; color:#8b949e;'>TYPE</div>
            <div style='font-size:0.95rem; font-weight:700; color:{"#f85149" if ft_pred!="No Failure" else "#3fb950"};'>{ft_pred}</div></div>""", 
            unsafe_allow_html=True)

        # Compact Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 24, "color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e", "tickfont": {"size": 8}},
                "bar": {"color": "#ff4b4b"},
                "bgcolor": "rgba(255,255,255,0.05)",
                "steps": [
                    {"range": [0, 25], "color": "rgba(63, 185, 80, 0.15)"},
                    {"range": [25, 50], "color": "rgba(255, 200, 0, 0.1)"},
                    {"range": [50, 75], "color": "rgba(245, 166, 35, 0.15)"},
                    {"range": [75, 100], "color": "rgba(248, 81, 73, 0.15)"},
                ],
                "threshold": {"line": {"color": "#f85149", "width": 2}, "thickness": 0.8, "value": 50},
            },
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=150, margin=dict(t=10, b=10, l=20, r=20))
        st.plotly_chart(fig_g, use_container_width=True)

        # Critical Recommendation
        recs = {
            "No Failure": ("System stable. Routine maintenance.", "#3fb950"),
            "Heat Dissipation Failure": ("Heat alarm! Check cooling/airflow.", "#f85149"),
            "Power Failure": ("Power anomaly! Spindle overload risk.", "#f5a623"),
            "Overstrain Failure": ("Mechanical strain! Reduce torque load.", "#f5a623"),
            "Tool Wear Failure": ("Tool life critical! Replace immediately.", "#f85149"),
            "Random Failures": ("Spontaneous error. System audit required.", "#ffc800"),
        }
        txt, clr = recs.get(ft_pred, ("Consult logs.", "#8b949e"))
        st.markdown(f"""
        <div style='background: rgba(10, 10, 10, 0.6); border-left: 4px solid {clr};
                    border-radius: 0 8px 8px 0; padding: 0.6rem 0.8rem;'>
          <div style='font-size: 0.6rem; color: #8b949e; text-transform: uppercase; margin-bottom: 0.1rem;'>RECOMMENDATION</div>
          <div style='color: #ffffff; font-size: 0.82rem; line-height: 1.3;'>{txt}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='placeholder-card' style='padding: 2rem 1rem;'>
          <div style='font-size: 1.8rem; margin-bottom: 0.5rem;'>🛡️</div>
          <div style='font-size: 1rem; color: #ffffff; font-weight: 600;'>System Standby</div>
          <div style='font-size: 0.8rem; color: #8b949e;'>Adjust parameters and trigger diagnostics.</div>
        </div>
        """, unsafe_allow_html=True)
