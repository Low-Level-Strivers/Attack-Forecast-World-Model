# 🛡️ Real-Time AI Network Attack Forecasting System

An end-to-end, proactive cyber defense platform built with **PyTorch** and **Streamlit**. Unlike traditional reactive Intrusion Detection Systems (IDS) that flag malicious traffic after execution, this system uses an **LSTM World Model** to forecast future network states ($S_{t+n}$), evaluate reconstruction loss deviations ($D_t$), and map anomalies to **MITRE ATT&CK** stages in real time.

---

## 🌟 Key Features

* **Proactive State Forecasting:** Multi-step lookahead horizon ($S_{t+1}, S_{t+2}, S_{t+3}$) to detect threats before full execution.
* **10-Second Temporal Windowing:** Aggregates packet metrics, flow ratios, TCP flag distributions, and IAT dynamics into structured time-series state vectors.
* **Calibrated Anomaly Scoring:** Trained strictly on benign baseline traffic (Monday dataset) to set baseline deviation thresholds ($D_t$) and keep false positives minimal.
* **MITRE ATT&CK Mapping:** Classifies anomalous states directly into threat stages such as *Reconnaissance*, *Execution*, and *Lateral Movement*.
* **Explainable AI (XAI):** Provides SHAP-inspired feature attribution scores highlighting top anomaly drivers per window.
* **Real-Time Streamlit Dashboard:** Dark-mode dashboard featuring live streaming metrics, progress tracking, interactive risk horizon charts, and flagged suspicious flow logs.

---

## 🏗️ System Architecture

```text
Network Traffic (CSV) 
    │
    ▼
[Feature Extraction & 10s Windowing] ──► Temporal State Sequences
    │
    ▼
[Min-Max Scaling & Tensor Formatting] ──► X_seq_train.npy
    │
    ▼
[Autoregressive LSTM World Model] ──► Future State Predictions (S_t+1..3)
    │
    ▼
[Deviation Assessment (D_t)] ──► Baseline Threshold Calibration
    │
    ▼
[MITRE ATT&CK & SHAP Attribution]
    │
    ▼
[Streamlit Live Streaming Dashboard]

```

## 📂 Project Structure

```text
Cyber-World-Model/
├── app.py                      # Streamlit Dashboard Application
├── requirements.txt            # Python Dependencies
├── .gitignore                  # Git Exclusion Rules
├── README.md                   # Project Documentation
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Feature Extraction & Scaling
│   ├── train_world_model.py    # PyTorch LSTM Model Training
│   ├── train_baseline.py       # Baseline Model Benchmark
│   ├── calibrate.py            # Anomaly Threshold Calibration
│   └── inference.py            # Threat Engine & ATT&CK Mapping
└── data/                       # CIC-IDS2017 Dataset Files
    └── README.md
```
## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Cyber-World-Model.git](https://github.com/YOUR_USERNAME/Cyber-World-Model.git)
cd Cyber-World-Model
```

### 2. Set Up Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Execution Pipeline

Follow this sequence to preprocess dataset features, train the LSTM neural network, calibrate decision thresholds, and launch the streaming interface:

### Step 1: Preprocess Dataset
Processes raw network flow records into 10-second temporal state windows and fits feature scalers:
```bash
python src/preprocess.py
```

```text
Inputs: data/Monday-WorkingHours.pcap_ISCX.csv

Output Artifacts: scaler.pkl, X_seq_train.npy, y_next_train.npy
```

