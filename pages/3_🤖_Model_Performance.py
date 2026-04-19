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
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from utils import load_config, pickle_load

st.set_page_config(page_title="Model Performance | FailSafe AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #171717 50%, #1f1f1f 100%); }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #000000 0%, #0f0f0f 100%); border-right: 1px solid #2a2a2a; }
  #MainMenu {visibility: hidden;} footer {visibility: hidden;}
  .winner-badge { display: inline-block; background: rgba(255,75,75,0.15); border: 1px solid rgba(255,75,75,0.4);
    border-radius: 20px; padding: 0.3rem 0.8rem; font-size: 0.82rem; color: #ff4b4b; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

COLORS_3 = {"Decision Tree": "#e0e0e0", "Random Forest": "#b71c1c", "XGBoost": "#ff4b4b"}

@st.cache_resource(show_spinner=False)
def load_metrics():
    config = load_config()
    metrics = pickle_load(config["models"]["directory"] + config["models"]["metrics"])
    ft_metrics = pickle_load(config["models"]["directory"] + config["models"]["failure_type_metrics"])
    return metrics, ft_metrics, config

metrics, ft_metrics, config = load_metrics()
best_name = metrics["best_model_name"]
feature_names = metrics["feature_names"]
model_names = ["Decision Tree", "Random Forest", "XGBoost"]

# Header
st.markdown(f"""
<h1 style='background: linear-gradient(135deg, #e0e0e0, #ff4b4b); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-size: 2rem;'>
  🤖 Model Performance Comparison
</h1>
<p style='color: #8b949e; font-size: 0.9rem;'>
  Comparing 3 classifiers -- Best model: <span class='winner-badge'>🏆 {best_name}</span>
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -- Metrics comparison cards --
cols = st.columns(3)
for i, name in enumerate(model_names):
    m = metrics[name]
    is_best = (name == best_name)
    border_color = "rgba(255,75,75,0.6)" if is_best else "rgba(255,255,255,0.08)"
    badge = " 🏆" if is_best else ""

    with cols[i]:
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.04); border: 2px solid {border_color};
                    border-radius: 12px; padding: 1.2rem; text-align: center;'>
          <div style='font-size: 1rem; font-weight: 600; color: {COLORS_3[name]};'>{name}{badge}</div>
          <div style='display: flex; justify-content: space-around; margin-top: 1rem;'>
            <div>
              <div style='font-size: 1.5rem; font-weight: 700; color: #e6edf3;'>{m["accuracy"]:.1%}</div>
              <div style='font-size: 0.7rem; color: #8b949e; text-transform: uppercase;'>Accuracy</div>
            </div>
            <div>
              <div style='font-size: 1.5rem; font-weight: 700; color: #e6edf3;'>{m["f1_score"]:.3f}</div>
              <div style='font-size: 0.7rem; color: #8b949e; text-transform: uppercase;'>F1 Score</div>
            </div>
            <div>
              <div style='font-size: 1.5rem; font-weight: 700; color: #e6edf3;'>{m["auc_roc"]:.3f}</div>
              <div style='font-size: 0.7rem; color: #8b949e; text-transform: uppercase;'>AUC-ROC</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -- ROC Curves --
st.markdown("### ROC Curves")
fig_roc = go.Figure()

for name in model_names:
    m = metrics[name]
    fig_roc.add_trace(go.Scatter(
        x=m["fpr"], y=m["tpr"],
        mode="lines", name=f"{name} (AUC={m['auc_roc']:.3f})",
        line=dict(color=COLORS_3[name], width=2.5),
    ))

fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
    line=dict(color="#484f58", width=1, dash="dash"),
))

fig_roc.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=420, font=dict(family="Inter"),
    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    legend=dict(x=0.55, y=0.05, bgcolor="rgba(0,0,0,0.5)", bordercolor="rgba(255,255,255,0.1)"),
)
st.plotly_chart(fig_roc, use_container_width=True)

# -- Confusion Matrices --
st.markdown("---")
st.markdown("### Confusion Matrices")

cm_cols = st.columns(3)
labels = ["Normal", "Failure"]

for i, name in enumerate(model_names):
    cm = metrics[name]["confusion_matrix"]
    with cm_cols[i]:
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, "#0d1117"], [1, COLORS_3[name]]],
            text=cm, texttemplate="%{text}",
            textfont={"size": 18, "color": "#e6edf3"},
            showscale=False,
        ))
        fig_cm.update_layout(
            title=dict(text=name, font=dict(size=14, color=COLORS_3[name])),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=300, font=dict(family="Inter"),
            xaxis_title="Predicted", yaxis_title="Actual",
            margin=dict(t=40, b=60),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# -- Feature Importance --
st.markdown("---")
st.markdown("### Feature Importance Comparison")

display_names_map = config["dataset"]["display_names"]
display_labels = [display_names_map.get(f, f) for f in feature_names]

fig_fi = make_subplots(rows=1, cols=3, subplot_titles=model_names, shared_yaxes=True)

for i, name in enumerate(model_names):
    importance = metrics[name]["feature_importance"]
    sorted_idx = np.argsort(importance)

    fig_fi.add_trace(go.Bar(
        x=importance[sorted_idx],
        y=[display_labels[j] for j in sorted_idx],
        orientation="h",
        marker_color=COLORS_3[name],
        name=name,
        showlegend=False,
    ), row=1, col=i+1)

fig_fi.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=450, font=dict(family="Inter", size=10),
    margin=dict(t=40, b=30, l=140),
)
st.plotly_chart(fig_fi, use_container_width=True)

# -- Detailed Classification Reports --
st.markdown("---")
st.markdown("### Detailed Classification Reports")

tabs = st.tabs(model_names)
for i, name in enumerate(model_names):
    with tabs[i]:
        report = metrics[name]["report"]
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(3)
        st.dataframe(report_df, use_container_width=True)

# -- Failure Type Model --
st.markdown("---")
st.markdown("### Failure Type Classifier (Multi-class)")

ft_col1, ft_col2 = st.columns(2)

with ft_col1:
    st.metric("Accuracy", f"{ft_metrics['accuracy']:.1%}")
    ft_report = pd.DataFrame(ft_metrics["report"]).transpose().round(3)
    st.dataframe(ft_report, use_container_width=True)

with ft_col2:
    ft_cm = ft_metrics["confusion_matrix"]
    ft_classes = ft_metrics["classes"]
    short_classes = [c.replace(" Failure", "").replace("Heat Dissipation", "Heat").replace("Random Failures", "Random") for c in ft_classes]

    fig_ft_cm = go.Figure(data=go.Heatmap(
        z=ft_cm, x=short_classes, y=short_classes,
        colorscale=[[0, "#0d1117"], [1, "#757575"]],
        text=ft_cm, texttemplate="%{text}",
        textfont={"size": 12, "color": "#e6edf3"},
        showscale=False,
    ))
    fig_ft_cm.update_layout(
        title="Failure Type Confusion Matrix",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, font=dict(family="Inter", size=10),
        xaxis_title="Predicted", yaxis_title="Actual",
        margin=dict(t=40, b=80, l=80),
    )
    st.plotly_chart(fig_ft_cm, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #484f58; font-size: 0.78rem;'>
  FailSafe AI &nbsp;|&nbsp; Built by Dhrubo
</div>
""", unsafe_allow_html=True)
