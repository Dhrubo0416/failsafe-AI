"""
Analytics Dashboard -- FailSafe AI
Interactive EDA with Plotly charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from utils import load_config, resolve_path

st.set_page_config(page_title="Analytics | FailSafe AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #171717 50%, #1f1f1f 100%); }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #000000 0%, #0f0f0f 100%); border-right: 1px solid #2a2a2a; }
  #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

DARK_TEMPLATE = "plotly_dark"
COLORS = ["#ff4b4b", "#e0e0e0", "#b71c1c", "#f5a623", "#3fb950", "#757575"]

@st.cache_data(show_spinner=False)
def load_data():
    config = load_config()
    path = resolve_path(config["dataset"]["data_directory"] + config["dataset"]["file_name"])
    df = pd.read_csv(path)
    df["Power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["Temp_Diff_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Overstrain_Indicator"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    return df

df = load_data()

# Header
st.markdown("""
<h1 style='background: linear-gradient(135deg, #e0e0e0, #ff4b4b); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-size: 2rem;'>
  📊 Analytics Dashboard
</h1>
<p style='color: #8b949e; font-size: 0.9rem; margin-bottom: 1.5rem;'>
  Explore patterns, distributions, and correlations in the predictive maintenance dataset
</p>
""", unsafe_allow_html=True)

# -- Row 1: Target distribution + Failure type --
col1, col2 = st.columns(2)

with col1:
    target_counts = df["Target"].value_counts().reset_index()
    target_counts.columns = ["Target", "Count"]
    target_counts["Label"] = target_counts["Target"].map({0: "Normal", 1: "Failure"})

    fig1 = px.pie(target_counts, values="Count", names="Label",
                  color_discrete_sequence=["#3fb950", "#f85149"],
                  title="Machine Status Distribution",
                  hole=0.5)
    fig1.update_layout(template=DARK_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", height=380,
                       font=dict(family="Inter"))
    fig1.update_traces(textinfo="percent+label", textfont_size=13)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    ft_counts = df["Failure Type"].value_counts().reset_index()
    ft_counts.columns = ["Failure Type", "Count"]
    ft_counts = ft_counts[ft_counts["Failure Type"] != "No Failure"]

    fig2 = px.bar(ft_counts, x="Count", y="Failure Type", orientation="h",
                  color="Failure Type", color_discrete_sequence=COLORS,
                  title="Failure Type Breakdown")
    fig2.update_layout(template=DARK_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", height=380,
                       showlegend=False, font=dict(family="Inter"),
                       yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig2, use_container_width=True)

# -- Row 2: Failure rate by type + Sensor distributions --
col3, col4 = st.columns(2)

with col3:
    type_failure = df.groupby("Type")["Target"].agg(["sum", "count"]).reset_index()
    type_failure.columns = ["Type", "Failures", "Total"]
    type_failure["Failure Rate (%)"] = (type_failure["Failures"] / type_failure["Total"] * 100).round(2)
    type_failure["Type Label"] = type_failure["Type"].map({"L": "Low Quality", "M": "Medium Quality", "H": "High Quality"})

    fig3 = px.bar(type_failure, x="Type Label", y="Failure Rate (%)",
                  color="Type Label", color_discrete_sequence=["#e0e0e0", "#f5a623", "#ff4b4b"],
                  title="Failure Rate by Machine Type",
                  text="Failure Rate (%)")
    fig3.update_layout(template=DARK_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", height=380,
                       showlegend=False, font=dict(family="Inter"))
    fig3.update_traces(textposition="outside", textfont_size=14)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    sensor_col = st.selectbox("Select Sensor Parameter",
                              ["Air temperature [K]", "Process temperature [K]",
                               "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                               "Power_W", "Temp_Diff_K"])

    df_plot = df.copy()
    df_plot["Status"] = df_plot["Target"].map({0: "Normal", 1: "Failure"})

    fig4 = px.histogram(df_plot, x=sensor_col, color="Status", barmode="overlay",
                        color_discrete_map={"Normal": "#3fb950", "Failure": "#f85149"},
                        title=f"Distribution of {sensor_col}",
                        opacity=0.7, nbins=50)
    fig4.update_layout(template=DARK_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", height=380,
                       font=dict(family="Inter"))
    st.plotly_chart(fig4, use_container_width=True)

# -- Row 3: Correlation heatmap --
st.markdown("---")
st.markdown("### Feature Correlation Heatmap")

numeric_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
                "Torque [Nm]", "Tool wear [min]", "Power_W", "Temp_Diff_K",
                "Overstrain_Indicator", "Target"]
corr = df[numeric_cols].corr()

short_labels = ["Air Temp", "Process Temp", "Rot. Speed", "Torque",
                "Tool Wear", "Power", "Temp Diff", "Overstrain", "Target"]

fig5 = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=short_labels,
    y=short_labels,
    colorscale=[[0, "#0d1117"], [0.5, "#b71c1c"], [1, "#ff4b4b"]],
    text=np.round(corr.values, 2),
    texttemplate="%{text}",
    textfont={"size": 11},
    zmin=-1, zmax=1,
))
fig5.update_layout(template=DARK_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)", height=500,
                   font=dict(family="Inter"),
                   xaxis=dict(side="bottom"),
                   margin=dict(t=30, b=80))
st.plotly_chart(fig5, use_container_width=True)

# -- Row 4: Scatter plots --
st.markdown("---")
st.markdown("### Sensor Relationships")

col5, col6 = st.columns(2)

with col5:
    df_scatter = df.copy()
    df_scatter["Status"] = df_scatter["Target"].map({0: "Normal", 1: "Failure"})

    fig6 = px.scatter(df_scatter, x="Rotational speed [rpm]", y="Torque [Nm]",
                      color="Status", opacity=0.5,
                      color_discrete_map={"Normal": "#3fb950", "Failure": "#f85149"},
                      title="Rotational Speed vs Torque")
    fig6.update_layout(template=DARK_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", height=400,
                       font=dict(family="Inter"))
    st.plotly_chart(fig6, use_container_width=True)

with col6:
    fig7 = px.scatter(df_scatter, x="Tool wear [min]", y="Torque [Nm]",
                      color="Status", opacity=0.5,
                      color_discrete_map={"Normal": "#3fb950", "Failure": "#f85149"},
                      title="Tool Wear vs Torque (Overstrain Zone)")
    fig7.update_layout(template=DARK_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", height=400,
                       font=dict(family="Inter"))
    st.plotly_chart(fig7, use_container_width=True)

# -- Row 5: Box plots --
st.markdown("---")
st.markdown("### Sensor Readings: Normal vs Failure")

sensors = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
           "Torque [Nm]", "Tool wear [min]"]

fig8 = make_subplots(rows=1, cols=5, subplot_titles=[s.split("[")[0].strip() for s in sensors])

for i, sensor in enumerate(sensors):
    for status, color, name in [(0, "#3fb950", "Normal"), (1, "#f85149", "Failure")]:
        fig8.add_trace(
            go.Box(y=df[df["Target"] == status][sensor], name=name,
                   marker_color=color, showlegend=(i == 0)),
            row=1, col=i+1
        )

fig8.update_layout(template=DARK_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)", height=350,
                   font=dict(family="Inter", size=10),
                   margin=dict(t=50, b=30))
st.plotly_chart(fig8, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #484f58; font-size: 0.78rem;'>
  FailSafe AI &nbsp;|&nbsp; Built by Dhrubo
</div>
""", unsafe_allow_html=True)
