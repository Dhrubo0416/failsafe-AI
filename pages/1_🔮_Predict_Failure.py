"""
Predict Failure -- FailSafe AI
Enter sensor parameters and get instant failure predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from utils import load_config, pickle_load, resolve_path

st.set_page_config(page_title="Predict Failure | FailSafe AI", page_icon="🔮", layout="wide")

# Reuse CSS from main app
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #171717 50%, #1f1f1f 100%); }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #000000 0%, #0f0f0f 100%); border-right: 1px solid #2a2a2a; }
  #MainMenu {visibility: hidden;} footer {visibility: hidden;}
  .risk-critical { background: rgba(248,81,73,0.15); border: 1px solid rgba(248,81,73,0.5); color: #f85149; border-radius: 12px; padding: 1.5rem; text-align: center; }
  .risk-high { background: rgba(245,166,35,0.15); border: 1px solid rgba(245,166,35,0.5); color: #f5a623; border-radius: 12px; padding: 1.5rem; text-align: center; }
  .risk-medium { background: rgba(255,200,0,0.12); border: 1px solid rgba(255,200,0,0.4); color: #ffc800; border-radius: 12px; padding: 1.5rem; text-align: center; }
  .risk-low { background: rgba(35,134,54,0.15); border: 1px solid rgba(35,134,54,0.5); color: #3fb950; border-radius: 12px; padding: 1.5rem; text-align: center; }
  .result-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 1.5rem; margin: 0.5rem 0; }
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
display_names = config["dataset"]["display_names"]

# Header
st.markdown("""
<div style='padding: 1rem 0;'>
  <h1 style='background: linear-gradient(135deg, #e0e0e0, #ff4b4b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem;'>
    ⚙️ Predict Equipment Failure
  </h1>
  <p style='color: #8b949e; font-size: 0.9rem;'>
    Enter CNC milling machine sensor readings below to predict failure probability and type
  </p>
</div>
""", unsafe_allow_html=True)

# Input form
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### Sensor Parameters")

    machine_type = st.selectbox(
        "Machine Type (Quality Variant)",
        options=["L", "M", "H"],
        help="L = Low quality (50%), M = Medium (30%), H = High (20%)",
        index=0,
    )

    air_temp = st.slider(
        "Air Temperature (K)",
        min_value=295.3, max_value=304.5, value=300.0, step=0.1,
        help="Ambient factory temperature"
    )

    process_temp = st.slider(
        "Process Temperature (K)",
        min_value=305.7, max_value=313.8, value=310.0, step=0.1,
        help="Machine operating temperature"
    )

    rot_speed = st.slider(
        "Rotational Speed (rpm)",
        min_value=1168, max_value=2886, value=1500, step=10,
        help="Spindle rotation speed"
    )

    torque = st.slider(
        "Torque (Nm)",
        min_value=3.8, max_value=76.6, value=40.0, step=0.1,
        help="Cutting force applied"
    )

    tool_wear = st.slider(
        "Tool Wear (min)",
        min_value=0, max_value=253, value=100, step=1,
        help="Cumulative tool usage time"
    )

    predict_btn = st.button("🔮 Predict Now", use_container_width=True, type="primary")

with col_result:
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
            "Type": type_encoded,
            "Air_Temp_K": air_temp,
            "Process_Temp_K": process_temp,
            "Rotational_Speed_rpm": rot_speed,
            "Torque_Nm": torque,
            "Tool_Wear_min": tool_wear,
            "Power_W": power,
            "Temp_Diff_K": temp_diff,
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
            risk_level = "CRITICAL"
            risk_class = "risk-critical"
            risk_icon = "🔴"
        elif failure_prob >= 0.50:
            risk_level = "HIGH"
            risk_class = "risk-high"
            risk_icon = "🟠"
        elif failure_prob >= 0.25:
            risk_level = "MEDIUM"
            risk_class = "risk-medium"
            risk_icon = "🟡"
        else:
            risk_level = "LOW"
            risk_class = "risk-low"
            risk_icon = "🟢"

        # Display results
        st.markdown("### Prediction Results")

        # Risk level card
        st.markdown(f"""
        <div class='{risk_class}'>
          <div style='font-size: 2rem;'>{risk_icon}</div>
          <div style='font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;'>{risk_level} RISK</div>
          <div style='font-size: 0.9rem; opacity: 0.8;'>Failure Probability: {failure_prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Status and failure type
        r1, r2 = st.columns(2)
        with r1:
            status = "FAILURE PREDICTED" if failure_pred == 1 else "MACHINE NORMAL"
            status_color = "#f85149" if failure_pred == 1 else "#3fb950"
            st.markdown(f"""
            <div class='result-card'>
              <div style='font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em;'>Status</div>
              <div style='font-size: 1.2rem; font-weight: 700; color: {status_color}; margin-top: 0.3rem;'>{status}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            ft_color = "#f85149" if ft_pred != "No Failure" else "#3fb950"
            st.markdown(f"""
            <div class='result-card'>
              <div style='font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em;'>Failure Type</div>
              <div style='font-size: 1.2rem; font-weight: 700; color: {ft_color}; margin-top: 0.3rem;'>{ft_pred}</div>
            </div>
            """, unsafe_allow_html=True)

        # Failure probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=failure_prob * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "#e6edf3"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
                "bar": {"color": "#ff4b4b"},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(35,134,54,0.2)"},
                    {"range": [25, 50], "color": "rgba(255,200,0,0.15)"},
                    {"range": [50, 75], "color": "rgba(245,166,35,0.2)"},
                    {"range": [75, 100], "color": "rgba(248,81,73,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#f85149", "width": 3},
                    "thickness": 0.8,
                    "value": 50
                },
            },
            title={"text": "Failure Probability", "font": {"size": 14, "color": "#8b949e"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(t=50, b=20, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Engineered features display
        st.markdown("#### Computed Sensor Metrics")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Power (W)", f"{power:.1f}")
        with e2:
            st.metric("Temp Difference (K)", f"{temp_diff:.1f}")
        with e3:
            st.metric("Overstrain Indicator", f"{overstrain:.1f}")

        # Maintenance recommendation
        st.markdown("<br>", unsafe_allow_html=True)
        recommendations = {
            "No Failure": ("✅ Machine is operating normally. Continue regular monitoring.", "#3fb950"),
            "Heat Dissipation Failure": ("🌡️ **Heat issue detected!** Check cooling system, verify ambient air flow, and ensure rotational speed is above 1380 rpm.", "#f85149"),
            "Power Failure": ("⚡ **Power anomaly detected!** Verify torque-speed combination. Power should be between 3500W and 9000W.", "#f5a623"),
            "Overstrain Failure": ("💪 **Overstrain risk!** Reduce tool wear or lower torque. Replace cutting tool immediately.", "#f5a623"),
            "Tool Wear Failure": ("🔧 **Tool wear critical!** Replace the cutting tool now. Tool has exceeded safe operating hours.", "#f85149"),
            "Random Failures": ("🎲 **Random failure risk.** No specific root cause. Perform general equipment inspection.", "#ffc800"),
        }
        rec_text, rec_color = recommendations.get(ft_pred, ("Consult maintenance team.", "#8b949e"))
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.04); border-left: 4px solid {rec_color};
                    border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin-top: 0.5rem;'>
          <div style='font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;'>
            Recommended Action
          </div>
          <div style='color: #e6edf3; font-size: 0.9rem; line-height: 1.5;'>{rec_text}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Placeholder when no prediction yet
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem;
                    background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.1);
                    border-radius: 16px; margin-top: 2rem;'>
          <div style='font-size: 3rem; margin-bottom: 1rem;'>⚙️</div>
          <div style='font-size: 1.1rem; color: #8b949e; font-weight: 500;'>
            Adjust the sensor parameters and click Predict
          </div>
          <div style='font-size: 0.85rem; color: #484f58; margin-top: 0.5rem;'>
            The model will analyze the readings and predict failure risk
          </div>
        </div>
        """, unsafe_allow_html=True)
