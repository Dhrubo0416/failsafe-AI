# 🔧 FailSafe AI — Predictive Maintenance Platform

![Project Architecture](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-FF4B4B?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-blue?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn)

**FailSafe AI** is an advanced machine learning platform designed to predict equipment failures in CNC milling machines **before they happen**. By transitioning from reactive maintenance to predictive maintenance, manufacturing plants can reduce unplanned downtime by up to 70% and optimize operational efficiency (OEE).

## 🚀 Live Demo
Access the live application deployed on Streamlit Community Cloud: **[FailSafe AI Live Demo](https://failsafe-ai-amhhbjkacjg4r86i55godh.streamlit.app/)**

---

## 🏭 The Business Problem
Unplanned equipment failures cost the manufacturing industry an estimated **$50 billion annually**. A single hour of downtime for heavy machinery can cost between $10,000 and $50,000. Traditional maintenance strategies fall into two categories:
1. **Reactive (Fix when broken):** Leads to catastrophic failures, safety risks, and expensive emergency repairs.
2. **Preventive (Fixed schedule):** Often results in changing perfectly good parts, wasting resources and reducing machine availability.

**The Solution:** FailSafe AI actively monitors sensor readings (temperatures, speed, torque, tool wear) to predict imminent machine failures, allowing maintenance teams to intervene exactly when needed.

---

## ✨ Key Features
- **Binary Failure Prediction:** Predicts whether a machine will fail based on current operating parameters.
- **Root Cause Classification:** Identifies the *type* of impending failure:
  - 🌡️ Heat Dissipation Failure (HDF)
  - ⚡ Power Failure (PWF)
  - 💪 Overstrain Failure (OSF)
  - 🔧 Tool Wear Failure (TWF)
  - 🎲 Random Failures (RNF)
- **Interactive Analytics Dashboard:** Explore sensor distributions, correlations, and historical failure patterns.
- **What-If Analysis Simulator:** Real-time parameter tweaking to visualize the impact on failure probability and identify safety margins.

---

## 🧠 Machine Learning Engine

### Models
The system evaluates three algorithms to ensure maximum predictive accuracy. **XGBoost** was selected as the final production model due to its superior performance on our highly imbalanced dataset.
- **Decision Tree** (Baseline)
- **Random Forest** (Ensemble bagging)
- **XGBoost** (Gradient boosting) - *Winner (97.95% Accuracy, F1: 0.732)*

### Feature Engineering
To capture complex physical interactions, the original 6 features were augmented with 3 domain-specific engineered features:
1. **Power [W]:** Computed taking Torque and Rotational Speed. Crucial for detecting Power Failures.
2. **Temperature Difference [K]:** Process Temperature minus Air Temperature. Critical for Heat Dissipation.
3. **Overstrain Indicator:** Tool Wear multiplied by Torque. Important for detecting mechanical strain.

### Class Imbalance Handling
The dataset exhibits extreme class imbalance (96.6% Normal vs 3.4% Failure). To prevent the model from simply predicting "Normal" for everything, **SMOTE (Synthetic Minority Over-sampling Technique)** was utilized exclusively on the training set to synthetically generate minority class variants.

---

## 📊 Dataset Reference
The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.
- **Instances:** 10,000
- **Features:** 6 raw sensor inputs + 3 engineered
- Contains 5 distinct failure modes recorded from operations.

---

## 💻 Local Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dhrubo0416/failsafe-AI.git
   cd failsafe-AI
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Data Pipeline and Model Training:**
   ```bash
   python src/data_pipeline.py
   python src/model_training.py
   ```

5. **Start the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

---

## 🔮 Future Roadmap
- Integration with live IoT sensor streaming.
- Deep Learning (LSTM) approaches for multi-step time-series forecasting.
- Alerting mechanisms (Email/SMS) using AWS SNS or Twilio.

---

*Developed by Dhrubo*
