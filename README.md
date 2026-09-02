# 🛡️ Real-Time AI Network Attack Forecasting System

An end-to-end, proactive cyber defense platform built with PyTorch and Streamlit. Unlike traditional reactive Intrusion Detection Systems (IDS), this project uses an **LSTM World Model** to forecast future network states ($S_{t+n}$), calculate reconstruction loss deviations ($D_t$), and map anomalies to MITRE ATT&CK stages in real time.

---

## 🌟 Key Features

* **Proactive Forecasting:** Multi-step lookahead horizon ($S_{t+1}, S_{t+2}, S_{t+3}$) to catch threats before full execution.
* **10-Second Temporal Windowing:** High-resolution temporal aggregation of flow metrics, IAT dynamics, and TCP flag counts.
* **Calibrated Anomaly Scoring:** Trained on benign baseline traffic with dynamic deviation thresholds ($D_t$) for low false-positive rates.
* **MITRE ATT&CK Stage Mapping:** Real-time behavioral mapping to Reconnaissance, Execution, and Lateral Movement.
* **Explainable AI (XAI):** Feature attribution highlights top driving anomaly indicators.
* **Streamlit Live Dashboard:** High-contrast, interactive real-time monitoring dashboard with CSV file streaming.

---

## 🏗️ System Architecture