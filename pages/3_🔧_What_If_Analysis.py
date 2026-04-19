"""
What-If Analysis -- FailSafe AI
Simulate scenarios by adjusting parameters in real-time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from utils import load_config, pickle_load, resolve_path

st.set_page_config(page_title="What-If Analysis | FailSafe AI", page_icon="🛡️", layout="wide")

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

# Premium Enterprise CSS
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
  .stApp {{ background: linear-gradient(135deg, #0a0a0a 0%, #171717 50%, #1f1f1f 100%); }}
  [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #000000 0%, #0f0f0f 100%); border-right: 1px solid #2a2a2a; }}
  #MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}}

  [data-testid="stMain"]::before {{
    content: ""; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: url("data:image/png;base64,{bg_b64}") center center / cover no-repeat;
    filter: blur(1.5px) brightness(0.35); opacity: 0.55; z-index: 0; pointer-events: none;
  }}

  [data-testid="stMain"] > div {{ position: relative; z-index: 1; }}

  .enterprise-header {{ display: flex; align-items: center; gap: 0.75rem; padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 0.8rem; }}
  .enterprise-header .title {{ font-size: 1.8rem !important; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; line-height: 1.4; padding-top: 0.5rem; padding-bottom: 0.5rem; }}
  .enterprise-header .subtitle {{ font-size: 1rem !important; color: #e0e0e0; }}
  .enterprise-header .icon {{ width: 40px; height: 40px; background: linear-gradient(135deg, #ff4b4b, #b71c1c); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0; }}

  .section-label {{ font-size: 0.75rem !important; font-weight: 700; color: #ffffff; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.4rem; padding-bottom: 0.2rem; border-bottom: 2px solid #ff4b4b; display: inline-block; }}
  
  [data-testid="stSlider"] {{ margin-bottom: -15px !important; }}
  [data-testid="stSlider"] label {{ font-size: 0.85rem !important; color: #ffffff !important; }}
  
  .block-container {{ padding-top: 2.5rem !important; padding-bottom: 2rem !important; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_models():
    config = load_config()
    best_model = pickle_load(config["models"]["directory"] + config["models"]["best_model"])
    ft_model = pickle_load(config["models"]["directory"] + config["models"]["failure_type_model"])
    ft_metrics = pickle_load(config["models"]["directory"] + config["models"]["failure_type_metrics"])
    return best_model, ft_model, ft_metrics, config

best_model, ft_model, ft_metrics, config = load_models()

def predict_single(type_val, air_temp, proc_temp, rot_speed, torque, tool_wear):
    type_encoded = config["dataset"]["type_encoding"][type_val]
    power = torque * rot_speed * (2 * np.pi / 60)
    temp_diff = proc_temp - air_temp
    overstrain = tool_wear * torque
    features = pd.DataFrame([{
        "Type": type_encoded, "Air_Temp_K": air_temp, "Process_Temp_K": proc_temp,
        "Rotational_Speed_rpm": rot_speed, "Torque_Nm": torque, "Tool_Wear_min": tool_wear,
        "Power_W": power, "Temp_Diff_K": temp_diff, "Overstrain_Indicator": overstrain,
    }])
    prob = best_model.predict_proba(features)[0][1]
    ft_pred_enc = ft_model.predict(features)[0]
    ft_pred = ft_metrics["label_encoder"].inverse_transform([ft_pred_enc])[0]
    return prob, ft_pred

# Enterprise Header
st.markdown(f"""
<div class='enterprise-header'>
  <div class='icon'>🛡️</div>
  <div>
    <div class='title'>What-If Simulation Lab</div>
    <div class='subtitle'>Stress-test physical thresholds and calibrate safety margins with real-time feedback</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Main structure
col_ctrl, col_viz = st.columns([1, 2.2], gap="large")

with col_ctrl:
    st.markdown("<div class='section-label'>Simulation Controls</div>", unsafe_allow_html=True)
    sim_type = st.selectbox("Machine Grade", ["L", "M", "H"], index=0)
    sim_air = st.slider("Air Temp (K)", 295.3, 304.5, 300.0, 0.1)
    sim_proc = st.slider("Process Temp (K)", 305.7, 313.8, 310.0, 0.1)
    sim_speed = st.slider("Rot. Speed (rpm)", 1168, 2886, 1500, 10)
    sim_torque = st.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.1)
    sim_wear = st.slider("Tool Wear (min)", 0, 253, 100, 1)

with col_viz:
    prob, ft = predict_single(sim_type, sim_air, sim_proc, sim_speed, sim_torque, sim_wear)
    risk_color = "#f85149" if prob >= 0.75 else ("#f5a623" if prob >= 0.50 else ("#ffc800" if prob >= 0.25 else "#3fb950"))
    
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.06); border:2px solid {risk_color}; border-radius:12px; padding:1rem; display:flex; justify-content:space-between; align-items:center;'>
        <div>
            <div style='font-size:0.7rem; color:#8b949e;'>CURRENT PROBABILITY</div>
            <div style='font-size:2.2rem; font-weight:800; color:{risk_color};'>{prob:.1%}</div>
        </div>
        <div>
            <div style='font-size:0.7rem; color:#8b949e;'>PREDICTED ROOT CAUSE</div>
            <div style='font-size:1.2rem; font-weight:700; color:{risk_color};'>{ft.upper()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    
    tab_sens, tab_safety = st.tabs(["📉 Parameter Sensitivity", "🛡️ Safety Margin Map"])
    
    with tab_sens:
        params = {
            "Tool Wear": ("tool_wear", np.linspace(0, 253, 30)),
            "Torque": ("torque", np.linspace(3.8, 76.6, 30)),
            "Rot. Speed": ("rot_speed", np.linspace(1168, 2886, 30)),
            "Air Temp": ("air_temp", np.linspace(295.3, 304.5, 30))
        }
        fig_sens = make_subplots(rows=2, cols=2, subplot_titles=list(params.keys()), vertical_spacing=0.2, horizontal_spacing=0.1)
        
        for idx, (title, (p_name, vals)) in enumerate(params.items()):
            row, col = idx // 2 + 1, idx % 2 + 1
            y_vals = []
            for v in vals:
                k = {"type_val": sim_type, "air_temp": sim_air, "proc_temp": sim_proc, "rot_speed": sim_speed, "torque": sim_torque, "tool_wear": sim_wear}
                k[p_name] = int(v) if p_name in ["rot_speed", "tool_wear"] else v
                p, _ = predict_single(**k)
                y_vals.append(p * 100)
            
            fig_sens.add_trace(go.Scatter(x=vals, y=y_vals, mode="lines", line=dict(color="#ff4b4b", width=2), showlegend=False), row=row, col=col)
            # Current value marker
            curr_v = {"tool_wear": sim_wear, "torque": sim_torque, "rot_speed": sim_speed, "air_temp": sim_air}[p_name]
            fig_sens.add_trace(go.Scatter(x=[curr_v], y=[prob*100], mode="markers", marker=dict(color="#f5a623", size=10, symbol="diamond"), showlegend=False), row=row, col=col)
            fig_sens.add_hline(y=50, line=dict(color="#f85149", width=1, dash="dash"), row=row, col=col)

        fig_sens.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=30, b=30, l=40, r=10), font=dict(size=9))
        fig_sens.update_yaxes(range=[0, 105])
        st.plotly_chart(fig_sens, use_container_width=True)

    with tab_safety:
        # Threshold Logic
        power = sim_torque * sim_speed * (2 * np.pi / 60)
        temp_diff = sim_proc - sim_air
        overstrain = sim_wear * sim_torque
        s_data = {
            "Power (W)": (power, 3500, 9000), "Temp Diff (K)": (temp_diff, 8.6, 12),
            "Overstrain": (overstrain, 0, 11000), "Tool Wear": (float(sim_wear), 0, 200)
        }
        fig_s = go.Figure()
        for i, (name, (val, s_min, s_max)) in enumerate(s_data.items()):
            color = "#3fb950" if s_min <= val <= s_max else "#f85149"
            fig_s.add_trace(go.Bar(x=[s_max-s_min], y=[name], base=[s_min], orientation="h", marker_color="rgba(35,134,54,0.2)", showlegend=False, width=0.4))
            fig_s.add_trace(go.Scatter(x=[val], y=[name], mode="markers", marker=dict(color=color, size=14, symbol="diamond", line=dict(color="#ffffff", width=1)), showlegend=False))
        fig_s.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=10, b=20, l=100, r=20))
        st.plotly_chart(fig_s, use_container_width=True)
