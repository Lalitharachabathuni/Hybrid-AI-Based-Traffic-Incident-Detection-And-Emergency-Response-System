"""
main.py
--------
FastAPI backend for the Hybrid AI-Based Traffic Incident Detection and
Emergency Response System.

Run:
    uvicorn main:app --reload --port 8000        (from inside backend/)

Docs / test UI: http://127.0.0.1:8000/docs
"""
import datetime
import os
import shutil
import sys
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
ML_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "ml"))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from database import (  # noqa: E402
    init_db, list_incidents, get_incident, list_notifications, stats_summary, reset_db,
    insert_citizen_report, list_citizen_reports, get_citizen_report, update_citizen_report_fields,
)
from notification import dispatch_alert, dispatch_citizen_alert  # noqa: E402
from report_generator import generate_report, build_citizen_report  # noqa: E402
from geocode import reverse_geocode  # noqa: E402
from ml.predict import predict_row  # noqa: E402
from ml.preprocess import FEATURE_COLUMNS  # noqa: E402
import simulate_stream  # noqa: E402

app = FastAPI(
    title="Hybrid AI Traffic Incident Detection & Emergency Response API",
    description="Backend for detecting traffic incidents, classifying severity, "
                "generating AI reports, and dispatching emergency notifications "
                "for Indian roads -- covering both sensor-based detection and "
                "citizen photo/GPS reporting from rural areas.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# ---------------------------------------------------------------------------
# Static files: uploaded accident photos + the frontend (dashboard + the
# public rural reporting page) are served straight off this same backend, so
# a deployment only needs ONE running server.
#   http://<server>:8000/app/report.html   -> public citizen reporting page
#   http://<server>:8000/app/index.html    -> control-room dashboard
# ---------------------------------------------------------------------------
DATA_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", "data"))
UPLOAD_DIR = os.path.join(DATA_DIR, "citizen_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
FRONTEND_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", "frontend"))

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/app/index.html")


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    speed_kmph: float
    occupancy: float
    flow: float
    speed_roll_mean: float | None = None
    speed_roll_std: float | None = None
    speed_drop_pct: float | None = None
    occ_roll_mean: float | None = None
    occ_spike: float | None = None
    flow_roll_mean: float | None = None
    flow_drop_pct: float | None = None
    hour: int | None = None
    station_pm: float | None = None
    location_name: str | None = None
    district: str | None = None
    state: str | None = None
    road: str | None = None
    lat: float | None = None
    lon: float | None = None
    direction: str | None = "NB"
    notify: bool = True


class SimulationRequest(BaseModel):
    speed: float = 500.0
    limit: int | None = 2000


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/stats")
def get_stats():
    return stats_summary()


# ---------------------------------------------------------------------------
# Accident Detection + Information Extraction + Emergency Notification
# ---------------------------------------------------------------------------
@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    Runs a single reading through the Accident Detection module (XGBoost).
    If the predicted severity is Minor/Moderate/Major, also generates an
    AI report and (optionally) dispatches emergency notifications, exactly
    like the live simulation does for each dataset row.
    """
    features = req.model_dump()
    for col in FEATURE_COLUMNS:
        if features.get(col) is None:
            features[col] = 12 if col == "hour" else 0.0

    try:
        result = predict_row({c: features[c] for c in FEATURE_COLUMNS})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    response = {**result}

    if result["severity_label"] != "Normal":
        from database import insert_incident
        import pandas as pd

        record = {
            "detected_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "station_pm": features.get("station_pm"),
            "location_name": features.get("location_name"),
            "district": features.get("district"),
            "state": features.get("state"),
            "road": features.get("road"),
            "lat": features.get("lat"),
            "lon": features.get("lon"),
            "direction": features.get("direction"),
            "speed_kmph": features["speed_kmph"],
            "occupancy": features["occupancy"],
            "flow": features["flow"],
            "severity_code": result["severity_code"],
            "severity_label": result["severity_label"],
            "confidence": result["confidence"],
            "report": "",
            "source": "manual_api",
        }
        record["report"] = generate_report(record)
        incident_id = insert_incident(record)
        response["incident_id"] = incident_id
        response["report"] = record["report"]

        if req.notify:
            response["notifications"] = dispatch_alert(
                incident_id, result["severity_label"], record["report"], record
            )

    return response


# ---------------------------------------------------------------------------
# Incidents & notifications (read access for dashboard / frontend)
# ---------------------------------------------------------------------------
@app.get("/api/incidents")
def get_incidents(limit: int = 200):
    return list_incidents(limit)


@app.get("/api/incidents/{incident_id}")
def get_incident_detail(incident_id: int):
    incident = get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident["notifications"] = list_notifications(incident_id)
    return incident


@app.delete("/api/incidents")
def clear_incidents():
    reset_db()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Citizen reporting: rural / roadside users upload an accident photo + their
# phone's GPS location directly through the public web page (frontend/report.html),
# no traffic camera or sensor required.
# ---------------------------------------------------------------------------
@app.get("/api/geocode")
def geocode(lat: float, lon: float):
    """Resolve GPS coordinates to a readable Indian address (village/mandal/
    district/state/PIN). Used by the reporting page to show the reporter
    what location will be sent, before they submit."""
    return reverse_geocode(lat, lon)


@app.post("/api/citizen/report")
async def submit_citizen_report(
    image: UploadFile = File(...),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
    village: str | None = Form(None),
    town: str | None = Form(None),
    district: str | None = Form(None),
    state: str | None = Form(None),
    pincode: str | None = Form(None),
    description: str | None = Form(None),
    severity_label: str = Form("Not sure"),
    reporter_name: str | None = Form(None),
    reporter_phone: str | None = Form(None),
):
    """Accepts a photo of an accident plus location details from a citizen,
    saves it, resolves/fills in the address, generates a clear accident
    report, stores it, and dispatches it to the emergency responders mapped
    to the reported severity -- all in one call so the rural reporting page
    only needs a single request."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a photo (image file).")

    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        ext = ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        shutil.copyfileobj(image.file, f)

    geo = reverse_geocode(lat, lon) if (lat is not None and lon is not None) else {}

    record = {
        "reported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reporter_name": reporter_name,
        "reporter_phone": reporter_phone,
        "lat": lat,
        "lon": lon,
        "village": village or geo.get("village"),
        "town": town or geo.get("town"),
        "mandal": geo.get("mandal"),
        "district": district or geo.get("district"),
        "state": state or geo.get("state"),
        "pincode": pincode or geo.get("pincode"),
        "road": geo.get("road"),
        "display_address": geo.get("display_address"),
        "description": description,
        "severity_label": severity_label or "Not sure",
        "image_path": f"/uploads/{fname}",
        "report": "",
        "status": "Reported",
    }

    report_id = insert_citizen_report(record)
    record["id"] = report_id
    report_text = build_citizen_report(record)
    update_citizen_report_fields(report_id, report=report_text)
    record["report"] = report_text

    notifications = dispatch_citizen_alert(report_id, record["severity_label"], report_text, record)

    address = record["display_address"] or ", ".join(
        p for p in [record["village"], record["district"], record["state"]] if p
    )

    return {
        "id": report_id,
        "reference_no": f"CR-{report_id}",
        "address": address or "Not resolved",
        "report": report_text,
        "notifications": notifications,
        "notified_departments": [n["department"] for n in notifications],
    }


@app.get("/api/citizen/reports")
def get_citizen_reports(limit: int = 200):
    return list_citizen_reports(limit)


@app.get("/api/citizen/reports/{report_id}")
def get_citizen_report_detail(report_id: int):
    report = get_citizen_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    report["notifications"] = list_notifications(report_id, source_type="citizen_report")
    return report


# ---------------------------------------------------------------------------
# Real-time dataset replay simulation (Data Collection -> ... -> Dashboard)
# ---------------------------------------------------------------------------
@app.post("/api/simulation/start")
def start_simulation(req: SimulationRequest):
    started = simulate_stream.start_background(speed=req.speed, limit=req.limit)
    if not started:
        raise HTTPException(status_code=409, detail="Simulation already running")
    return {"status": "started", **req.model_dump()}


@app.post("/api/simulation/stop")
def stop_simulation():
    simulate_stream.stop_background()
    return {"status": "stopping"}


@app.get("/api/simulation/status")
def simulation_status():
    return simulate_stream.get_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
