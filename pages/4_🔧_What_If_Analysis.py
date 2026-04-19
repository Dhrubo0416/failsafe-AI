"""
What-If Analysis -- FailSafe AI
Simulate scenarios by adjusting parameters in real-time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from utils import load_config, pickle_load

st.set_page_config(page_title="What-If Analysis | FailSafe AI", page_icon="🔧", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); border-right: 1px solid #30363d; }
  #MainMenu {visibility: hidden;} footer {visibility: hidden;}
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
    """Make a prediction for a single set of parameters."""
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

# Header
st.markdown("""
<h1 style='background: linear-gradient(135deg, #00d2ff, #3a7bd5); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-size: 2rem;'>
  🔧 What-If Analysis
</h1>
<p style='color: #8b949e; font-size: 0.9rem; margin-bottom: 1rem;'>
  Adjust sensor parameters and watch the failure probability change in real-time
</p>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("### Simulation Controls")
    sim_type = st.selectbox("Machine Type", ["L", "M", "H"], index=0)
    sim_air = st.slider("Air Temp (K)", 295.3, 304.5, 300.0, 0.1)
    sim_proc = st.slider("Process Temp (K)", 305.7, 313.8, 310.0, 0.1)
    sim_speed = st.slider("Rotational Speed (rpm)", 1168, 2886, 1500, 10)
    sim_torque = st.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.1)
    sim_wear = st.slider("Tool Wear (min)", 0, 253, 100, 1)

# Real-time prediction
prob, ft = predict_single(sim_type, sim_air, sim_proc, sim_speed, sim_torque, sim_wear)

# Display current prediction
if prob >= 0.75:
    risk_color, risk = "#f85149", "CRITICAL"
elif prob >= 0.50:
    risk_color, risk = "#f5a623", "HIGH"
elif prob >= 0.25:
    risk_color, risk = "#ffc800", "MEDIUM"
else:
    risk_color, risk = "#3fb950", "LOW"

rc1, rc2, rc3 = st.columns(3)
with rc1:
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.04); border:1px solid {risk_color}; border-radius:12px; padding:1rem; text-align:center;'>
      <div style='font-size:2rem; font-weight:700; color:{risk_color};'>{prob:.1%}</div>
      <div style='font-size:0.75rem; color:#8b949e; text-transform:uppercase;'>Failure Probability</div>
    </div>""", unsafe_allow_html=True)
with rc2:
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.04); border:1px solid {risk_color}; border-radius:12px; padding:1rem; text-align:center;'>
      <div style='font-size:1.5rem; font-weight:700; color:{risk_color};'>{risk}</div>
      <div style='font-size:0.75rem; color:#8b949e; text-transform:uppercase;'>Risk Level</div>
    </div>""", unsafe_allow_html=True)
with rc3:
    ft_c = "#f85149" if ft != "No Failure" else "#3fb950"
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.04); border:1px solid {ft_c}; border-radius:12px; padding:1rem; text-align:center;'>
      <div style='font-size:1rem; font-weight:700; color:{ft_c};'>{ft}</div>
      <div style='font-size:0.75rem; color:#8b949e; text-transform:uppercase;'>Failure Type</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -- Sensitivity charts: how each param affects failure prob --
st.markdown("### Parameter Sensitivity Analysis")
st.markdown("<p style='color:#8b949e; font-size:0.85rem;'>Each chart shows how changing one parameter (while keeping others fixed) affects failure probability</p>", unsafe_allow_html=True)

# Generate sensitivity data
params = {
    "Tool Wear (min)": ("tool_wear", np.linspace(0, 253, 50)),
    "Torque (Nm)": ("torque", np.linspace(3.8, 76.6, 50)),
    "Rotational Speed (rpm)": ("rot_speed", np.linspace(1168, 2886, 50)),
    "Air Temperature (K)": ("air_temp", np.linspace(295.3, 304.5, 50)),
}

fig_sens = make_subplots(rows=2, cols=2, subplot_titles=list(params.keys()),
                         vertical_spacing=0.15, horizontal_spacing=0.1)

for idx, (title, (param_name, values)) in enumerate(params.items()):
    row = idx // 2 + 1
    col = idx % 2 + 1
    probs = []

    for v in values:
        kwargs = {
            "type_val": sim_type, "air_temp": sim_air, "proc_temp": sim_proc,
            "rot_speed": sim_speed, "torque": sim_torque, "tool_wear": sim_wear
        }
        kwargs[param_name] = v
        # Ensure integer for rpm and tool_wear
        if param_name == "rot_speed":
            kwargs[param_name] = int(v)
        elif param_name == "tool_wear":
            kwargs[param_name] = int(v)
        p, _ = predict_single(**kwargs)
        probs.append(p * 100)

    # Current value marker
    current_vals = {
        "tool_wear": sim_wear, "torque": sim_torque,
        "rot_speed": sim_speed, "air_temp": sim_air,
    }
    current = current_vals[param_name]

    fig_sens.add_trace(go.Scatter(
        x=values, y=probs, mode="lines",
        line=dict(color="#00d2ff", width=2.5),
        showlegend=False,
    ), row=row, col=col)

    # Add danger zone
    fig_sens.add_hline(y=50, line=dict(color="#f85149", width=1, dash="dash"),
                       row=row, col=col)

    # Mark current position
    curr_prob, _ = predict_single(**{
        "type_val": sim_type, "air_temp": sim_air, "proc_temp": sim_proc,
        "rot_speed": sim_speed, "torque": sim_torque, "tool_wear": sim_wear
    })
    fig_sens.add_trace(go.Scatter(
        x=[current], y=[curr_prob * 100], mode="markers",
        marker=dict(color="#f5a623", size=12, symbol="diamond",
                    line=dict(color="#e6edf3", width=2)),
        showlegend=False,
    ), row=row, col=col)

fig_sens.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=600, font=dict(family="Inter", size=11),
    margin=dict(t=40, b=40),
)
fig_sens.update_yaxes(title_text="Failure Prob. (%)", range=[0, 100])
st.plotly_chart(fig_sens, use_container_width=True)

# -- Safety zones visualization --
st.markdown("---")
st.markdown("### Safety Zone Map")
st.markdown("<p style='color:#8b949e; font-size:0.85rem;'>Green = safe operating range, Red = danger zone based on current parameters</p>", unsafe_allow_html=True)

# Compute safety metrics
power = sim_torque * sim_speed * (2 * np.pi / 60)
temp_diff = sim_proc - sim_air
overstrain = sim_wear * sim_torque

# Thresholds from domain knowledge
safety_data = {
    "Power (W)": {"value": power, "safe_min": 3500, "safe_max": 9000, "actual_min": 400, "actual_max": 23000},
    "Temp Diff (K)": {"value": temp_diff, "safe_min": 8.6, "safe_max": 20, "actual_min": 5, "actual_max": 15},
    "Overstrain": {"value": overstrain, "safe_min": 0, "safe_max": 11000, "actual_min": 0, "actual_max": 19000},
    "Tool Wear (min)": {"value": float(sim_wear), "safe_min": 0, "safe_max": 200, "actual_min": 0, "actual_max": 253},
}

fig_safety = go.Figure()

for i, (name, d) in enumerate(safety_data.items()):
    in_safe = d["safe_min"] <= d["value"] <= d["safe_max"]
    marker_color = "#3fb950" if in_safe else "#f85149"

    # Safe zone background
    fig_safety.add_trace(go.Bar(
        x=[d["safe_max"] - d["safe_min"]], y=[name], base=[d["safe_min"]],
        orientation="h", marker_color="rgba(35,134,54,0.25)",
        showlegend=(i == 0), name="Safe Zone",
        width=0.5,
    ))

    # Current value marker
    fig_safety.add_trace(go.Scatter(
        x=[d["value"]], y=[name], mode="markers",
        marker=dict(color=marker_color, size=16, symbol="diamond",
                    line=dict(color="#e6edf3", width=2)),
        showlegend=(i == 0), name="Current Value",
    ))

fig_safety.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=300, font=dict(family="Inter"),
    xaxis_title="Value", margin=dict(t=20, b=40, l=120),
    legend=dict(orientation="h", y=-0.2),
    barmode="overlay",
)
st.plotly_chart(fig_safety, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #484f58; font-size: 0.78rem;'>
  FailSafe AI &nbsp;|&nbsp; Built by Dhrubo
</div>
""", unsafe_allow_html=True)
