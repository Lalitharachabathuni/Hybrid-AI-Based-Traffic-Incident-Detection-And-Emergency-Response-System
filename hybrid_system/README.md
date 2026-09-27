# Hybrid AI-Based Traffic Incident Detection and Emergency Response System

Department of Allied Computer Science & Engineering — Batch CSM-03
Built strictly following the accompanying project-review PPT (`Hybrid_AI.pptx`),
fully localized for Indian roads (locations, units, emergency numbers).

---

## 1. What this project actually does

This is a full, runnable implementation of the system architecture described
in the PPT:

```
Data Collection → Data Preprocessing → Vehicle Detection/Tracking →
Speed Estimation → Accident Detection (severity) → Information Extraction →
AI Incident Analysis (report generation) → Emergency Notification →
Live Dashboard & Database
```

with a working ML model, a REST API, a live dashboard, and a public web
front-end for citizen accident reporting — everything running end-to-end.

### About the sensor dataset (India, NH16 corridor)

The sensor-based half of the pipeline (Accident Detection module) needs a
per-minute speed/occupancy/flow feed to train and demo against. No public
Indian highway sensor dataset with that kind of granularity was supplied, so
`backend/ml/preprocess.py` **synthesizes** one instead: realistic diurnal
traffic (morning/evening rush-hour slowdowns) with randomly injected
incident-like events (sudden speed drops + occupancy build-up), generated
for a real, named highway corridor — **NH16, through Andhra Pradesh**,
covering Vijayawada → Mangalagiri → Guntur → Tenali → Ponnur → Chinaganjam →
Bapatla → Chirala → Ongole → Kavali (see `backend/ml/india_stations.py` for
the full list with real coordinates, district and state).

This means every incident the sensor pipeline detects — on the dashboard map,
in the incident table, and in every generated report — is tied to a real
Indian place name and district, not just a bare coordinate or a US postmile
marker. The generator is fully documented in `preprocess.py`, and swapping
it for a real Indian sensor feed (NHAI / state highway ITS data) later
requires no changes anywhere else in the project — every other module only
depends on the column names in `FEATURE_COLUMNS` plus
`station_pm`/`lat`/`lon`/`location_name`/`district`/`state`/`road`/`direction`.

The PPT's Technologies slide also lists a computer-vision stack (YOLOv11 +
ByteTrack + Optical Flow) for a *video* input. That's included as a
**bonus, optional script** (`backend/video_detection.py`) that works on any
traffic-camera clip you supply — it's not on the critical path since the
core pipeline works directly from sensor/flow data (the same approach used
by the papers cited in the Literature Survey slide: LSTM/RF-RFE and
FA/Weighted-Random-Forest models that detect incidents from flow data, not
video).

There is also no ground-truth accident/incident label available (real or
synthetic). Severity labels (Minor / Moderate / Major) are generated with a
transparent, documented anomaly-detection heuristic (sudden speed drop +
occupancy build-up relative to a rolling baseline) — the same style of
labeling used in the traffic-incident-detection literature. This is clearly
marked in `backend/ml/preprocess.py` (`label_incidents()`), and can be
swapped for real incident logs (state police / NHAI accident data) if you
obtain them.

**Everything is localized for India:** speeds are in km/h throughout (no
mph anywhere), locations are shown as real place/district/state names (no
US postmile markers or coordinates-only), and every generated report and
notification uses Indian emergency numbers and department names (Police
100, 108 Ambulance Service, Fire & Rescue 101, National Emergency 112,
Police Control Room, Nearest Government Hospital).

### New: citizen photo + GPS reporting, for areas with no traffic sensors

Most rural Indian roads have neither loop detectors nor cameras, so the
sensor pipeline above can't see accidents there at all. `frontend/report.html`
is a plain, dependency-light, mobile-first public web page (English /
తెలుగు / हिंदी) that lets *anyone standing near an accident* report it
directly:

1. Take or choose a photo of the accident.
2. Share their phone's GPS location (auto reverse-geocoded to
   village/mandal/district/state/PIN via OpenStreetMap Nominatim — free,
   no API key — with a manual-entry fallback if GPS/network isn't available).
3. Pick how serious it looks (Minor / Moderate / Severe / Not sure).
4. Optionally add a description and a callback number.

Submitting calls `POST /api/citizen/report`, which saves the photo, resolves
the address, generates a clear structured accident report, stores it, and
dispatches it to the Indian responders mapped to that severity (Local
Traffic Police / Police Control Room 100 / 108 Ambulance / Fire & Rescue 101
/ Nearest Government Hospital) — the same notification pipeline the sensor
side uses, plus a reference number (e.g. `CR-14`) the reporter can quote.
There is deliberately **no automatic AI severity detection from the photo
alone** (a single image isn't a reliable basis for that) — severity is
whatever the reporter selects, defaulting to "Not sure", which is treated as
seriously as "Moderate" so it's never under-escalated. All citizen reports
(with photo thumbnails) also show up in the control-room dashboard
(`frontend/index.html`) alongside the sensor-detected incidents.

Because both pages are served straight from the FastAPI backend (see
`app.mount("/app", ...)` in `backend/main.py`), a deployment only needs one
running server:
- Public reporting page: `http://<server>:8000/app/report.html`
- Control-room dashboard: `http://<server>:8000/app/index.html`

---

## 2. Project structure

```
traffic_incident_system/
├── data/
│   ├── MobileCentury/        <- your dataset (already included)
│   └── processed/            <- generated feature table (created by setup)
├── backend/
│   ├── main.py                <- FastAPI app (the REST API)
│   ├── database.py             <- SQLite storage layer
│   ├── notification.py         <- Emergency Notification module
│   ├── report_generator.py     <- AI report generation (template + optional Gemini)
│   ├── simulate_stream.py       <- Real-time dataset replay engine
│   ├── video_detection.py       <- OPTIONAL: YOLOv11+ByteTrack+OpticalFlow (needs a video file)
│   ├── models/                  <- trained model (xgb_incident_model.joblib) + metrics
│   └── ml/
│       ├── preprocess.py         <- Data Collection + Preprocessing + feature engineering
│       ├── train_model.py        <- Trains the XGBoost severity classifier
│       └── predict.py            <- Loads the model for inference
├── dashboard/
│   └── app.py                 <- Streamlit live monitoring dashboard
├── frontend/
│   └── index.html             <- React + Tailwind web UI (no Node/npm needed)
├── requirements.txt
├── .env.example
├── setup.bat            <- one-time Windows setup
├── start_backend.bat
├── start_dashboard.bat
├── start_frontend.bat
└── run_all.bat          <- starts everything at once
```

---

## 3. How to run it on Windows

### Prerequisites
- **Windows 10/11**
- **Python 3.10, 3.11, or 3.12** installed from https://www.python.org/downloads/
  — during installation, check **"Add python.exe to PATH"**.
- Internet connection for the one-time `pip install` step (and for the
  frontend page, which loads React/Tailwind from a CDN).

### Step-by-step

1. **Unzip** this project anywhere, e.g. `C:\Users\<you>\Desktop\traffic_incident_system`.

2. **Double-click `setup.bat`** (or run it from a Command Prompt inside the
   project folder). This will:
   - create a Python virtual environment (`.venv`)
   - install all required packages
   - build the synthesized NH16 (Andhra Pradesh) traffic feature dataset
   - train the XGBoost severity-classification model

   This only needs to be done once. It takes a few minutes.

3. **Double-click `run_all.bat`**. This opens three things:
   - a window running the **FastAPI backend** at `http://127.0.0.1:8000`
     (interactive API docs at `http://127.0.0.1:8000/docs`)
   - a window running the **Streamlit dashboard**, which will open your
     browser at `http://localhost:8501`
   - the **React + Tailwind frontend** (`frontend/index.html`) in your
     default browser

   You can also start each piece individually with `start_backend.bat`,
   `start_dashboard.bat`, and `start_frontend.bat` — just make sure the
   backend is running before the dashboard/frontend, since they both call it.

4. **In the dashboard (or the frontend page), click "Start"** under
   *Dataset Replay Simulation*. This replays the synthesized NH16 sensor data
   in timestamp order (sped up) through the full pipeline: every reading is
   scored by the XGBoost model, and any Minor/Moderate/Major detection
   automatically generates an AI report and dispatches simulated emergency
   notifications to the correct departments, all of which appear live on
   the map, KPI cards, and incident feed.

5. **To stop everything**, just close the backend/dashboard windows (or
   press `Ctrl+C` inside them).

### Re-running later
You don't need to run `setup.bat` again unless you delete `.venv` or want to
retrain the model. Just use `run_all.bat` (or the individual `start_*.bat`
scripts) going forward.

---

## 4. Using the system directly (API)

The FastAPI backend exposes a full REST API (see `http://127.0.0.1:8000/docs`
for interactive Swagger UI):

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/predict` | POST | Run one traffic reading through Accident Detection → Report → Notification |
| `/api/incidents` | GET | List detected incidents |
| `/api/incidents/{id}` | GET | Incident detail + AI report + notifications |
| `/api/incidents` | DELETE | Clear all stored incidents |
| `/api/simulation/start` | POST | Start replaying the dataset as a live feed |
| `/api/simulation/stop` | POST | Stop the replay |
| `/api/simulation/status` | GET | Replay progress |
| `/api/stats` | GET | Aggregate counts by severity |

Example manual test (from the API docs page or curl):
```json
POST /api/predict
{
  "speed_kmph": 5.0,
  "occupancy": 0.5,
  "flow": 60,
  "speed_roll_mean": 80,
  "speed_drop_pct": 0.9,
  "occ_spike": 0.3,
  "station_pm": 27.0,
  "location_name": "Guntur",
  "district": "Guntur",
  "state": "Andhra Pradesh",
  "road": "NH16",
  "lat": 16.3067,
  "lon": 80.4365,
  "direction": "NB"
}
```
This simulates a sudden, severe speed drop near Guntur and will be
classified **Major**, triggering notifications to Police Control Room (100),
108 Ambulance Service, Fire & Rescue Service (101), Nearest Government
Hospital, and Traffic Control Room — exactly the escalation rule defined in
the "Proposed System" slide.

---

## 5. Mapping this implementation back to the PPT

| PPT module | Implementation |
|---|---|
| Data Collection | `backend/ml/preprocess.py` — synthesizes NH16 (Andhra Pradesh) sensor data across named locations in `india_stations.py` |
| Data Preprocessing | same file — cleans, resamples, aligns on time/location |
| Vehicle Detection & Tracking | `backend/video_detection.py` (bonus, video-based, YOLOv11 + ByteTrack) |
| Speed Estimation | GPS-derived speed (core pipeline) + Optical Flow (bonus video module) |
| Accident Detection | `backend/ml/train_model.py` + `predict.py` — XGBoost severity classifier |
| Information Extraction | `backend/report_generator.py` — structured report fields |
| AI Incident Analysis | same file — optional Gemini narrative rewrite |
| Emergency Notification | `backend/notification.py` — severity → department rules |
| Dashboard & Database | `dashboard/app.py` (Streamlit) + `backend/database.py` (SQLite) + `frontend/index.html` (React/Tailwind) |

### Tech-stack substitutions (and why)
- **MongoDB / Firebase Firestore → SQLite**: keeps the project 100%
  self-contained on Windows with zero external services to install or
  configure. `backend/database.py` is a thin, isolated data-access layer —
  swapping in MongoDB only requires reimplementing the functions in that one
  file.
- **Gemini 2.5 / EasyOCR**: implemented as optional integrations
  (`report_generator.py`) that activate automatically if you install the
  relevant package and set an API key in `.env` — otherwise the system uses
  a clear, structured template report so it always runs without any paid
  API key.
- **React + Tailwind**: implemented as a single static page using the React
  and Tailwind CDN builds, so there is no Node.js/npm install step — just
  open the HTML file. If you'd like a full compiled React app with routing,
  component splitting, etc. for a more "production" front-end, that can be
  built out from this page.

---

## 6. Retraining / regenerating data

If you want to rebuild the feature dataset or retrain the model manually:

```bat
call .venv\Scripts\activate.bat
cd backend\ml
python preprocess.py
python train_model.py
```

Metrics from the last training run are saved to `backend/models/metrics.json`.

---

## 7. Troubleshooting

- **"python is not recognized"** — reinstall Python and check "Add to PATH",
  or open a new Command Prompt after installing.
- **Dashboard/frontend shows "Backend not reachable"** — make sure the
  backend window (FastAPI/uvicorn) is running and shows
  `Uvicorn running on http://127.0.0.1:8000`.
- **Port already in use** — close any previous backend/dashboard windows,
  or edit the port in `start_backend.bat` / `start_dashboard.bat`.
- **`pip install` fails / slow** — check your internet connection; corporate
  proxies sometimes need `pip install --proxy <proxy> -r requirements.txt`.
