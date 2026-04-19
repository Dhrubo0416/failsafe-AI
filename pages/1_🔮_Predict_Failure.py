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
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return data

bg_b64 = get_bg_image()

# Premium Enterprise CSS — Compact + Prominent Background
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

  /* CNC background — More prominent (reduced blur, increased brightness/opacity) */
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

  /* Make all main content sit above the background */
  [data-testid="stMain"] > div {{
    position: relative;
    z-index: 1;
  }}

  /* Compact spacing for sliders (Reset styles for Sliders) */
  [data-testid="stSlider"] {{
    margin-bottom: -15px !important;
  }}
  [data-testid="stSlider"] label {{
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #e6edf3 !important;
  }}

  /* Semi-transparent panels so background shows through */
  .glass-panel {{
    background: rgba(10, 10, 10, 0.6);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
  }}

  /* Risk cards */
  .risk-critical {{
    background: rgba(248,81,73,0.18);
    border: 1px solid rgba(248,81,73,0.6);
    color: #f85149;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }}
  .risk-high {{
    background: rgba(245,166,35,0.18);
    border: 1px solid rgba(245,166,35,0.6);
    color: #f5a623;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }}
  .risk-medium {{
    background: rgba(255,200,0,0.15);
    border: 1px solid rgba(255,200,0,0.5);
    color: #ffc800;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }}
  .risk-low {{
    background: rgba(35,134,54,0.18);
    border: 1px solid rgba(35,134,54,0.6);
    color: #3fb950;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }}
  .result-card {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.3rem 0;
  }}

  /* Enterprise header bar */
  .enterprise-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 0.8rem;
  }}
  .enterprise-header .icon {{
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #ff4b4b, #b71c1c);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
  }}
  .enterprise-header .title {{
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.3px;
  }}
  .enterprise-header .subtitle {{
    font-size: 0.82rem;
    color: #8b949e;
    margin-top: 0.1rem;
  }}

  /* Breadcrumb */
  .breadcrumb {{
    font-size: 0.75rem;
    color: #6e7681;
    margin-bottom: 0.4rem;
    letter-spacing: 0.04em;
  }}
  .breadcrumb span {{
    color: #ff4b4b;
  }}

  /* Section label */
  .section-label {{
    font-size: 0.68rem;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #ff4b4b;
    display: inline-block;
  }}

  /* Placeholder card */
  .placeholder-card {{
    text-align: center;
    padding: 3rem 1.5rem;
    background: rgba(255,255,255,0.05);
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 12px;
  }}

  /* Reduce top padding of main block */
  .block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
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

# Breadcrumb
st.markdown("<div class='breadcrumb'>FailSafe AI &nbsp;/&nbsp; <span>Predict Failure</span></div>", unsafe_allow_html=True)

# Enterprise Header
st.markdown("""
<div class='enterprise-header'>
  <div class='icon'>🛡️</div>
  <div>
    <div class='title'>Failure Prediction Engine</div>
    <div class='subtitle'>Real-time CNC milling machine diagnostics &mdash; High-Fidelity Diagnostics</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Input form — with Sliders in 2 columns
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("<div class='section-label'>Sensor Input Parameters</div>", unsafe_allow_html=True)
    
    # Machine Type wrapped in its own row for better vertical control
    machine_type = st.selectbox("Machine Quality Variant", options=["L", "M", "H"], 
                                 help="L=Low (50%), M=Medium (30%), H=High (20%)", index=0)

    # Sliders in 2-column layout for compactness
    c1, c2 = st.columns(2)
    with c1:
        air_temp = st.slider("Air Temp (K)", 295.3, 304.5, 300.0, 0.1)
        process_temp = st.slider("Process Temp (K)", 305.7, 313.8, 310.0, 0.1)
        rot_speed = st.slider("Rotational Speed (rpm)", 1168, 2886, 1500, 10)
    with c2:
        torque = st.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.1)
        tool_wear = st.slider("Tool Wear (min)", 0, 253, 100, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🛡️  Analyze Hardware Health", use_container_width=True, type="primary")

with col_result:
    st.markdown("<div class='section-label'>System Diagnostic Output</div>", unsafe_allow_html=True)

    if predict_btn:
        # Encode type
        type_map = config["dataset"]["type_encoding"]
        type_encoded = type_map[machine_type]

        # Engineer features
        power = torque * rot_speed * (2 * np.pi / 60)
        temp_diff = process_temp - air_temp
        overstrain = tool_wear * torque

        # Build feature vector
        features = pd.DataFrame([{
            "Type": type_encoded, "Air_Temp_K": air_temp,
            "Process_Temp_K": process_temp, "Rotational_Speed_rpm": rot_speed,
            "Torque_Nm": torque, "Tool_Wear_min": tool_wear,
            "Power_W": power, "Temp_Diff_K": temp_diff,
            "Overstrain_Indicator": overstrain,
        }])

        # Binary prediction
        failure_prob = best_model.predict_proba(features)[0][1]
        failure_pred = best_model.predict(features)[0]

        # Failure type prediction
        ft_pred_encoded = ft_model.predict(features)[0]
        le = ft_metrics["label_encoder"]
        ft_pred = le.inverse_transform([ft_pred_encoded])[0]

        # Risk level
        if failure_prob >= 0.75:
            risk_level, risk_class, risk_icon = "CRITICAL", "risk-critical", "🔴"
        elif failure_prob >= 0.50:
            risk_level, risk_class, risk_icon = "HIGH", "risk-high", "🟠"
        elif failure_prob >= 0.25:
            risk_level, risk_class, risk_icon = "MEDIUM", "risk-medium", "🟡"
        else:
            risk_level, risk_class, risk_icon = "LOW", "risk-low", "🟢"

        # Risk level card
        st.markdown(f"""
        <div class='{risk_class}'>
          <div style='font-size: 1.5rem;'>{risk_icon}</div>
          <div style='font-size: 1.3rem; font-weight: 700; margin: 0.2rem 0;'>{risk_level} RISK</div>
          <div style='font-size: 0.85rem; opacity: 0.9;'>Failure Probability: {failure_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

        # Status and failure type
        r1, r2 = st.columns(2)
        with r1:
            status = "FAILURE PREDICTED" if failure_pred == 1 else "MACHINE NORMAL"
            status_color = "#f85149" if failure_pred == 1 else "#3fb950"
            st.markdown(f"""
            <div class='result-card'>
              <div style='font-size: 0.65rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em;'>Status</div>
              <div style='font-size: 1rem; font-weight: 700; color: {status_color}; margin-top: 0.15rem;'>{status}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            ft_color = "#f85149" if ft_pred != "No Failure" else "#3fb950"
            st.markdown(f"""
            <div class='result-card'>
              <div style='font-size: 0.65rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em;'>Predicted Type</div>
              <div style='font-size: 1rem; font-weight: 700; color: {ft_color}; margin-top: 0.15rem;'>{ft_pred}</div>
            </div>
            """, unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=failure_prob * 100,
            number={"suffix": "%", "font": {"size": 30, "color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e", "size": 10}},
                "bar": {"color": "#ff4b4b"},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(63, 185, 80, 0.2)"},
                    {"range": [25, 50], "color": "rgba(255, 200, 0, 0.15)"},
                    {"range": [50, 75], "color": "rgba(245, 166, 35, 0.2)"},
                    {"range": [75, 100], "color": "rgba(248, 81, 73, 0.2)"},
                ],
                "threshold": {"line": {"color": "#f85149", "width": 2}, "thickness": 0.8, "value": 50},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=200, margin=dict(t=30, b=10, l=25, r=25),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Maintenance recommendation
        recommendations = {
            "No Failure": ("Machine operating within normal parameters. Routine checks advised.", "#3fb950"),
            "Heat Dissipation Failure": ("Overheating risk! Inspect heat exchangers and airflow.", "#f85149"),
            "Power Failure": ("Power anomaly! Verify spindle torque and operational power usage.", "#f5a623"),
            "Overstrain Failure": ("Mechanical stress! Reduce processing torque or tool load.", "#f5a623"),
            "Tool Wear Failure": ("Tool life end! Immediate bit replacement required.", "#f85149"),
            "Random Failures": ("Spontaneous error risk. Run secondary system calibration.", "#ffc800"),
        }
        rec_text, rec_color = recommendations.get(ft_pred, ("Consult engineering logs.", "#8b949e"))
        st.markdown(f"""
        <div style='background: rgba(10, 10, 10, 0.6); border-left: 4px solid {rec_color};
                    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;'>
          <div style='font-size: 0.62rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;'>Recommended Action</div>
          <div style='color: #ffffff; font-size: 0.85rem; line-height: 1.4;'>{rec_text}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Placeholder
        st.markdown("""
        <div class='placeholder-card'>
          <div style='font-size: 2.2rem; margin-bottom: 0.8rem;'>🛡️</div>
          <div style='font-size: 1.05rem; color: #ffffff; font-weight: 600;'>System Ready for Diagnostic</div>
          <div style='font-size: 0.8rem; color: #8b949e; margin-top: 0.4rem;'>Adjust sliders to specific sensor levels and trigger hardware analysis.</div>
        </div>
        """, unsafe_allow_html=True)
