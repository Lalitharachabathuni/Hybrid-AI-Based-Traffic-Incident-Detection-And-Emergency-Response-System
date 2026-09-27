# Hybrid AI-Based Traffic Incident Detection and Emergency Response System

**Department of Allied Computer Science & Engineering — Batch CSM-03**  
Fully localized for Indian roads (NH16 Highway Corridor, Andhra Pradesh), units (km/h), and emergency response structures (Police 100, 108 Ambulance, Fire 101, National Emergency 112).

---

## 🚀 Quick Start (Windows)

### 1. One-Time Setup
Double-click **`setup.bat`** (or run `setup.bat` from terminal) to:
- Create Python virtual environment (`.venv`)
- Install all dependencies (`requirements.txt`)
- Build synthesized feature dataset for the NH16 corridor
- Train the XGBoost incident severity classifier

### 2. Run Everything
Double-click **`run_all.bat`** (or run `run_all.bat` from terminal). This launches:
1. **FastAPI Backend & REST API**: `http://127.0.0.1:8000` (Swagger UI: `http://127.0.0.1:8000/docs`)
2. **Control Room Dashboard (React + Tailwind)**: `http://127.0.0.1:8000/app/index.html`
3. **Citizen Accident Reporting Page (Mobile/Web)**: `http://127.0.0.1:8000/app/report.html`
4. **Streamlit Analytics Dashboard**: `http://localhost:8501`

---

## 📁 Repository Structure

```
├── .gitignore               <- Git ignore patterns for Python, caches & data
├── README.md                <- Quickstart and project guide
├── setup.bat                <- One-time setup launcher
├── run_all.bat              <- Full system launcher
├── start_backend.bat        <- Backend API launcher
├── start_dashboard.bat      <- Streamlit dashboard launcher
├── start_frontend.bat       <- Web UI launcher
├── start_report_page.bat    <- Citizen reporting page launcher
└── hybrid_system/
    ├── backend/
    │   ├── main.py              <- FastAPI REST API server
    │   ├── database.py          <- SQLite database layer
    │   ├── notification.py      <- Multi-agency emergency alert dispatcher
    │   ├── report_generator.py  <- AI Incident Analysis & report generator
    │   ├── geocode.py           <- OpenStreetMap reverse geocoding
    │   ├── simulate_stream.py   <- Real-time sensor replay engine
    │   ├── video_detection.py   <- Optional YOLOv11 + ByteTrack + Optical Flow CV pipeline
    │   ├── models/              <- Trained XGBoost model & evaluation metrics
    │   └── ml/
    │       ├── india_stations.py <- NH16 highway stations (Vijayawada to Kavali)
    │       ├── preprocess.py     <- Feature engineering & data synthesis
    │       ├── train_model.py    <- Model training & evaluation
    │       └── predict.py        <- Model inference pipeline
    ├── dashboard/
    │   └── app.py               <- Streamlit interactive monitoring dashboard
    ├── frontend/
    │   ├── index.html           <- Control-room live incident & alert dashboard
    │   └── report.html          <- Public roadside citizen crash reporting page
    ├── data/
    │   ├── processed/           <- Generated feature table
    │   └── citizen_uploads/     <- Uploaded incident photos
    └── requirements.txt         <- Python package dependencies
```

---

## 🧪 Testing

To run backend tests:
```powershell
.\hybrid_system\.venv\Scripts\python.exe -m pytest
```

---

## 🌐 Git Setup & Push Instructions

To push this repository to GitHub / GitLab:
```bash
git init
git add .
git commit -m "Initial commit: Hybrid AI Traffic Incident Detection and Emergency Response System"
git branch -M main
git remote add origin <YOUR_REMOTE_REPOSITORY_URL>
git push -u origin main
```
