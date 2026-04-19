"""
Model Performance -- FailSafe AI
Compare Decision Tree vs Random Forest vs XGBoost.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from utils import load_config, pickle_load, resolve_path

st.set_page_config(page_title="Model Performance | FailSafe AI", page_icon="🛡️", layout="wide")

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
  .enterprise-header .icon {{ width: 55px; height: 55px; background: linear-gradient(135deg, #ff4b4b, #b71c1c); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; flex-shrink: 0; }}

  .section-label {{ font-size: 0.8rem !important; font-weight: 700; color: #ffffff; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.5rem; padding-bottom: 0.2rem; border-bottom: 2px solid #ff4b4b; display: inline-block; }}

  .metric-mini-card {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 0.8rem; text-align: center; }}
  
  .block-container {{ padding-top: 2.5rem !important; padding-bottom: 2rem !important; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_metrics():
    config = load_config()
    metrics = pickle_load(config["models"]["directory"] + config["models"]["metrics"])
    ft_metrics = pickle_load(config["models"]["directory"] + config["models"]["failure_type_metrics"])
    return metrics, ft_metrics, config

metrics, ft_metrics, config = load_metrics()
best_name = metrics["best_model_name"]
model_names = ["Decision Tree", "Random Forest", "XGBoost"]
COLORS_3 = {"Decision Tree": "#e0e0e0", "Random Forest": "#b71c1c", "XGBoost": "#ff4b4b"}

# Enterprise Header
st.markdown(f"""
<div class='enterprise-header'>
  <div class='icon'>📊</div>
  <div>
    <div class='title'>Algorithmic Benchmarks</div>
    <div class='subtitle'>Comparative analysis across model architectures &mdash; Current Production: {best_name}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Tabs for compactness
tab_metrics, tab_binary, tab_multiclass = st.tabs(["📊 Performance Metrics", "🔬 Binary Diagnostics", "🏷️ Failure Type Detail"])

with tab_metrics:
    st.markdown("<div class='section-label'>Summary Statistics</div>", unsafe_allow_html=True)
    m_cols = st.columns(3)
    for i, name in enumerate(model_names):
        m = metrics[name]
        is_best = (name == best_name)
        border = "#ff4b4b" if is_best else "rgba(255,255,255,0.1)"
        with m_cols[i]:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.05); border:2px solid {border}; border-radius:12px; padding:1rem; text-align:center;'>
                <div style='font-size:0.9rem; font-weight:700; color:{COLORS_3[name]};'>{name.upper()} {"🏆" if is_best else ""}</div>
                <div style='display:flex; justify-content:space-around; margin-top:0.8rem;'>
                    <div><div style='font-size:1.4rem; font-weight:800;'>{m["accuracy"]:.1%}</div><div style='font-size:0.6rem; color:#8b949e;'>ACCURACY</div></div>
                    <div><div style='font-size:1.4rem; font-weight:800;'>{m["f1_score"]:.3f}</div><div style='font-size:0.6rem; color:#8b949e;'>F1 SCORE</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    
    col_roc, col_fi = st.columns([1, 1.2])
    with col_roc:
        st.markdown("<div class='section-label'>ROC Curves</div>", unsafe_allow_html=True)
        fig_roc = go.Figure()
        for name in model_names:
            m = metrics[name]
            fig_roc.add_trace(go.Scatter(x=m["fpr"], y=m["tpr"], mode="lines", name=f"{name}", line=dict(color=COLORS_3[name], width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, line=dict(color="#484f58", width=1, dash="dash")))
        fig_roc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(t=10, b=10, l=10, r=10), legend=dict(x=0.6, y=0.08, font=dict(size=9)))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_fi:
        st.markdown("<div class='section-label'>Feature Importance</div>", unsafe_allow_html=True)
        display_names_map = config["dataset"]["display_names"]
        feature_names = metrics["feature_names"]
        display_labels = [display_names_map.get(f, f) for f in feature_names]
        
        fig_fi = go.Figure()
        for name in model_names:
            importance = metrics[name]["feature_importance"]
            fig_fi.add_trace(go.Bar(x=display_labels, y=importance, name=name, marker_color=COLORS_3[name]))
        
        fig_fi.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, barmode='group', margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=1.2, font=dict(size=9)))
        st.plotly_chart(fig_fi, use_container_width=True)

with tab_binary:
    st.markdown("<div class='section-label'>Binary Confusion Matrices</div>", unsafe_allow_html=True)
    cm_cols = st.columns(3)
    labels = ["Normal", "Failure"]
    for i, name in enumerate(model_names):
        cm = metrics[name]["confusion_matrix"]
        with cm_cols[i]:
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels, colorscale=[[0, "#0d1117"], [1, COLORS_3[name]]], text=cm, texttemplate="%{text}", textfont={"size": 14}, showscale=False))
            fig_cm.update_layout(title=dict(text=name, font=dict(size=12)), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(t=30, b=30, l=30, r=30))
            st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("<div class='section-label'>Classification Reports</div>", unsafe_allow_html=True)
    tab_reports = st.tabs(model_names)
    for i, name in enumerate(model_names):
        with tab_reports[i]:
            st.dataframe(pd.DataFrame(metrics[name]["report"]).transpose().round(3), use_container_width=True)

with tab_multiclass:
    st.markdown("<div class='section-label'>Failure Type Classifier</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.metric("Aggregate Accuracy", f"{ft_metrics['accuracy']:.1%}")
        st.dataframe(pd.DataFrame(ft_metrics["report"]).transpose().round(3), use_container_width=True)
    with col2:
        ft_classes = [c.replace(" Failure", "").replace("Heat Dissipation", "Heat").replace("Random Failures", "Random") for c in ft_metrics["classes"]]
        fig_ft_cm = go.Figure(data=go.Heatmap(z=ft_metrics["confusion_matrix"], x=ft_classes, y=ft_classes, colorscale="Greys", text=ft_metrics["confusion_matrix"], texttemplate="%{text}", showscale=False))
        fig_ft_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=10, b=40, l=40, r=10))
        st.plotly_chart(fig_ft_cm, use_container_width=True)
