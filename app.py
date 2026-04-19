"""
app.py -- FailSafe AI
========================
Predictive Maintenance for Smart Manufacturing
Powered by XGBoost + Random Forest + Streamlit

Deploy on Streamlit Community Cloud for free remote access.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from utils import load_config, pickle_load, resolve_path

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="FailSafe AI | Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "FailSafe AI -- Predict equipment failures before they happen. Built by Dhrubo.",
    }
)

# ================================================================
# CUSTOM CSS -- Premium Dark Theme
# ================================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  .stApp {
    background: linear-gradient(135deg, #0a0a0a 0%, #171717 50%, #1f1f1f 100%);
    min-height: 100vh;
  }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #000000 0%, #0f0f0f 100%);
    border-right: 1px solid #2a2a2a;
  }

  .gradient-title {
    background: linear-gradient(135deg, #e0e0e0 0%, #ff4b4b 50%, #e0e0e0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1.2;
  }

  .subtitle {
    color: #8b949e;
    font-size: 0.95rem;
    margin-top: 0.25rem;
    font-weight: 400;
  }

  .metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(224,224,224,0.15);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
  }
  .metric-card:hover {
    border-color: rgba(255,75,75,0.6);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(255,75,75,0.15);
  }
  .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ff4b4b;
  }
  .metric-label {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .sidebar-section {
    color: #e0e0e0;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1rem 0 0.5rem 0;
    border-bottom: 1px solid rgba(224,224,224,0.2);
    padding-bottom: 0.25rem;
  }

  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(35,134,54,0.15);
    border: 1px solid rgba(35,134,54,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    color: #3fb950;
    font-weight: 500;
  }

  .risk-critical {
    background: rgba(248,81,73,0.15);
    border: 1px solid rgba(248,81,73,0.5);
    color: #f85149;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    font-weight: 600;
    font-size: 1.1rem;
  }
  .risk-high {
    background: rgba(245,166,35,0.15);
    border: 1px solid rgba(245,166,35,0.5);
    color: #f5a623;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    font-weight: 600;
    font-size: 1.1rem;
  }
  .risk-medium {
    background: rgba(255,200,0,0.12);
    border: 1px solid rgba(255,200,0,0.4);
    color: #ffc800;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    font-weight: 600;
    font-size: 1.1rem;
  }
  .risk-low {
    background: rgba(35,134,54,0.15);
    border: 1px solid rgba(35,134,54,0.5);
    color: #3fb950;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    font-weight: 600;
    font-size: 1.1rem;
  }

  .info-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
  }

  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}

  [data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
  }

  [data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
  }

  hr {
    border-color: rgba(255,255,255,0.08) !important;
  }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# LOAD RESOURCES
# ================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    config = load_config()
    best_model = pickle_load(config["models"]["directory"] + config["models"]["best_model"])
    ft_model = pickle_load(config["models"]["directory"] + config["models"]["failure_type_model"])
    metrics = pickle_load(config["models"]["directory"] + config["models"]["metrics"])
    ft_metrics = pickle_load(config["models"]["directory"] + config["models"]["failure_type_metrics"])
    return best_model, ft_model, metrics, ft_metrics, config

@st.cache_data(show_spinner=False)
def load_dataset():
    config = load_config()
    path = resolve_path(config["dataset"]["data_directory"] + config["dataset"]["file_name"])
    return pd.read_csv(path)

try:
    best_model, ft_model, metrics, ft_metrics, config = load_models()
    raw_df = load_dataset()
    models_loaded = True
except Exception as e:
    st.error(f"**Error loading models:** {e}")
    st.info("Please run `python src/data_pipeline.py` then `python src/model_training.py` first.")
    models_loaded = False
    st.stop()

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
      <div style='font-size: 2.5rem;'>⚙️</div>
      <div style='font-size: 1.1rem; font-weight: 700; color: #ff4b4b;'>FailSafe AI</div>
      <div style='font-size: 0.75rem; color: #a0a0a0; margin-top: 0.25rem;'>Predictive Maintenance Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin: 0.5rem 0 1rem 0;'>
      <span class='status-badge'>&#x25CF; Models Loaded</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("<div class='sidebar-section'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem; color:#8b949e; line-height:1.7;'>
    FailSafe AI uses <b style='color:#e0e0e0;'>machine learning</b> to predict
    equipment failures in CNC milling machines <b>before they happen</b>,
    analyzing 6 sensor parameters + 3 engineered features.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("<div class='sidebar-section'>Tech Stack</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.8rem; color:#8b949e; line-height:1.8;'>
      🤖 <b style='color:#ff4b4b;'>XGBoost + RF</b> -- ML Models<br>
      📊 <b style='color:#e0e0e0;'>Plotly</b> -- Visualizations<br>
      🔬 <b style='color:#ff4b4b;'>SMOTE</b> -- Class Balancing<br>
      🎈 <b style='color:#e0e0e0;'>Streamlit</b> -- Frontend<br>
      📦 <b style='color:#ff4b4b;'>scikit-learn</b> -- ML Pipeline<br>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("<div class='sidebar-section'>Dataset</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.8rem; color:#8b949e; line-height:1.7;'>
    <b style='color:#e0e0e0;'>AI4I 2020</b> Predictive Maintenance Dataset<br>
    UCI Machine Learning Repository<br>
    10,000 data points | 5 failure modes
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='text-align:center; font-size:0.75rem; color:#484f58; padding:0.5rem 0;'>
      Built by Dhrubo
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# MAIN CONTENT -- HOME PAGE
# ================================================================

# Header
st.markdown("""
<div style='padding: 1.5rem 0 1rem 0;'>
  <div class='gradient-title'>FailSafe AI</div>
  <div class='subtitle'>
    Predict equipment failures before they happen -- powered by XGBoost, Random Forest & SMOTE
  </div>
</div>
""", unsafe_allow_html=True)

# KPI strip
best_name = metrics["best_model_name"]
best_acc = metrics[best_name]["accuracy"]
best_f1 = metrics[best_name]["f1_score"]
best_auc = metrics[best_name]["auc_roc"]
failure_rate = raw_df["Target"].mean() * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value'>{best_acc:.1%}</div>
      <div class='metric-label'>Model Accuracy</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value'>{best_f1:.2f}</div>
      <div class='metric-label'>F1 Score</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value'>{best_auc:.2f}</div>
      <div class='metric-label'>AUC-ROC</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-value'>{failure_rate:.1f}%</div>
      <div class='metric-label'>Failure Rate</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Project description cards
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    <div class='info-card'>
      <h4 style='color:#e0e0e0; margin-top:0;'>🏭 The Problem</h4>
      <p style='color:#8b949e; font-size:0.88rem; line-height:1.6;'>
        Unplanned equipment failures cost the manufacturing industry
        <b style='color:#ff4b4b;'>$50 billion annually</b>. A single hour of
        downtime on a CNC milling machine costs $10,000-$50,000. Traditional
        reactive maintenance is too late; preventive maintenance on fixed schedules
        is wasteful.
      </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class='info-card'>
      <h4 style='color:#e0e0e0; margin-top:0;'>⚙️ The Solution</h4>
      <p style='color:#8b949e; font-size:0.88rem; line-height:1.6;'>
        FailSafe AI uses <b style='color:#e0e0e0;'>real-time sensor data</b> --
        temperature, speed, torque, tool wear -- to predict failures
        <b style='color:#ff4b4b;'>before they happen</b>. It identifies the
        <b>specific failure type</b> (Heat, Power, Overstrain, Tool Wear, Random),
        enabling targeted preventive action.
      </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Feature cards
st.markdown("""
<div style='color:#e6edf3; font-size:1.1rem; font-weight:600; margin-bottom:0.75rem;'>
  Navigate the Pages
</div>
""", unsafe_allow_html=True)

col_p1, col_p2, col_p3, col_p4 = st.columns(4)

with col_p1:
    st.markdown("""
    <div class='info-card' style='text-align:center; min-height: 140px;'>
      <div style='font-size:1.8rem; margin-bottom:0.5rem;'>🔮</div>
      <div style='color:#e6edf3; font-weight:600; font-size:0.9rem;'>Predict Failure</div>
      <div style='color:#8b949e; font-size:0.78rem; margin-top:0.3rem;'>Enter sensor data and get instant failure predictions with risk levels</div>
    </div>
    """, unsafe_allow_html=True)

with col_p2:
    st.markdown("""
    <div class='info-card' style='text-align:center; min-height: 140px;'>
      <div style='font-size:1.8rem; margin-bottom:0.5rem;'>📊</div>
      <div style='color:#e6edf3; font-weight:600; font-size:0.9rem;'>Analytics</div>
      <div style='color:#8b949e; font-size:0.78rem; margin-top:0.3rem;'>Explore interactive charts, distributions, and failure pattern insights</div>
    </div>
    """, unsafe_allow_html=True)

with col_p3:
    st.markdown("""
    <div class='info-card' style='text-align:center; min-height: 140px;'>
      <div style='font-size:1.8rem; margin-bottom:0.5rem;'>🤖</div>
      <div style='color:#e6edf3; font-weight:600; font-size:0.9rem;'>Model Performance</div>
      <div style='color:#8b949e; font-size:0.78rem; margin-top:0.3rem;'>Compare Decision Tree vs Random Forest vs XGBoost side by side</div>
    </div>
    """, unsafe_allow_html=True)

with col_p4:
    st.markdown("""
    <div class='info-card' style='text-align:center; min-height: 140px;'>
      <div style='font-size:1.8rem; margin-bottom:0.5rem;'>🔧</div>
      <div style='color:#e6edf3; font-weight:600; font-size:0.9rem;'>What-If Analysis</div>
      <div style='color:#8b949e; font-size:0.78rem; margin-top:0.3rem;'>Simulate scenarios by adjusting sensor parameters in real-time</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #484f58; font-size: 0.78rem; padding: 0.5rem 0;'>
  FailSafe AI &nbsp;|&nbsp; Built by Dhrubo &nbsp;|&nbsp; Powered by XGBoost + scikit-learn + Streamlit
  &nbsp;|&nbsp; <a href='https://github.com/Dhrubo0416/failsafe-AI' style='color: #ff4b4b; text-decoration: none;'>GitHub</a>
</div>
""", unsafe_allow_html=True)
