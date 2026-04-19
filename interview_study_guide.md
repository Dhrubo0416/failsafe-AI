# 🎯 FailSafe AI — Interview Study Guide & Cheat Sheet

Use this guide to confidently present FailSafe AI in your technical interviews and resume reviews.

## 🎤 The 2-Minute Elevator Pitch
*"For my portfolio, I built **FailSafe AI**, an end-to-end predictive maintenance platform. I tackled a major manufacturing problem: unplanned downtime, which costs the industry billions annually. Using the AI4I predictive maintenance dataset, I built a pipeline that ingests CNC machine sensor readings—like temperature, torque, and rotational speed—and predicts an impending machine failure with over 97% accuracy.*

*I engineered three domain-specific features—such as detecting the Overstrain Indicator—and handled extreme class imbalance (96% normal vs 3% failure) using SMOTE. Finally, I trained multiple models, deployed the highest performing **XGBoost** model into a production-grade Streamlit application, and included a secondary Random Forest classifier to pinpoint the exact **type** of failure so maintenance teams know exactly what to fix."*

---

## 🛠️ Common Interview Questions & Answers

### 1. Why did you choose XGBoost over Decision Tree?
**Answer:** "The original baseline only used a single Decision Tree. While interpretable, it's highly prone to overfitting and struggled heavily with the minority failure class (F1 score was ~0.49). I upgraded to Random Forest and XGBoost. XGBoost leverages Gradient Boosting (building trees sequentially to correct the errors of the previous ones). It performed the best handling the non-linear relationships of the engineered physics features, bumping the F1 score to 0.73 and overall accuracy to 97.9%."

### 2. How did you handle the imbalanced dataset? Why not use Undersampling?
**Answer:** "The data was extremely imbalanced (96.6% no failure vs 3.4% failure). The previous iteration used Random Undersampling, which meant throwing away 90% of the valuable normal operating data. Instead, I used **SMOTE (Synthetic Minority Over-sampling Technique)** on the *training set only*. It synthetically generated new failure examples by interpolating between existing minority instances, allowing the model to learn the decision boundary without discarding the majority class information."

### 3. What new features did you engineer, and why?
**Answer:** "Instead of relying purely on the raw sensor inputs, I calculated mathematically derived features simulating actual mechanical physics:
1. **Power [W]:** By multiplying Torque and Rotational Speed (converted to rad/s). This was massive for detecting Power Failures.
2. **Temperature Difference [K]:** Process Temperature minus Air Temperature. Direct trigger for Heat Dissipation failures.
3. **Overstrain Indicator:** Tool wear multiplied by torque, directly targeting mechanical strain over time."

### 4. What is the difference between Predictive and Preventive Maintenance?
**Answer:**
- **Preventive Maintenance** is based on time or usage (e.g., change oil every 3 months). It's safer than reactive, but often results in replacing parts that are still perfectly good.
- **Predictive Maintenance** (what FailSafe AI does) monitors the exact condition of the machine in real-time. It only triggers an alert when the data shows an anomaly, drastically saving resource costs and maximizing tool utilization.

### 5. If I am the factory floor manager, how do I actually use this app?
**Answer:** "The application has a 'What-If Analysis' simulator exactly for this. The manager inputs the current sensor readings. The app outputs two things: the instantaneous **Probability of Failure (Risk Level)**, and the **Specific Failure Type** (e.g., 'Tool Wear Failure'). The application even outputs a recommended action—if it sees 'Tool Wear Failure', it pushes a warning to physically replace the cutting attachment immediately, preventing an unplanned halt to the assembly line."

### 6. You mentioned OEE, what is that?
**Answer:** "OEE stands for **Overall Equipment Effectiveness**, the gold standard for measuring manufacturing productivity. It comprises Availability, Performance, and Quality. By predicting failures before they happen, FailSafe AI directly targets the 'Availability' metric, cutting down the downtime that plummets OEE scores."

---

## 📈 Important Metrics to Memorize
If an interviewer asks for exact figures, have these ready:
- **Baseline Accuracy:** ~93% (Decision Tree, weak F1)
- **FailSafe AI Accuracy:** ~97.9% (XGBoost)
- **F1 Score Improved:** Jumped from 0.49 → 0.73
- **Dataset Size:** 10,000 instances
- **Imbalance Ratio:** 96.6% vs 3.4%

---

## 🚀 Scenario / Follow-up Questions

**Q: How would you deploy this in a real factory?**
*A:* "In real production, the sensors on the CNC machines (IoT) stream data via MQTT or Kafka. The ML pipeline would be hosted on AWS SageMaker or an endpoint service. The model would run inferences every minute, and if the probability exceeds a threshold (e.g. 75%), it would trigger an SNS alert directly to the maintenance technician's dashboard or pager."

**Q: What is concept drift and how do you monitor it here?**
*A:* "Over time, machines age naturally—what represents 'normal' torque output today might be slightly different in 2 years. I would set up a monitoring loop that tracks the distribution of the incoming inference data against the original training distribution. If they diverge significantly, it's an automated signal to retrain the model."
