"""
main.py
--------
FastAPI backend for the Hybrid AI-Based Traffic Incident Detection and
Emergency Response System (India).

Features:
- Full-stack unified backend serving REST APIs + static assets (Control Room & Citizen App)
- Admin Login & Authentication Security
- Sensor Analytics & XGBoost Severity Classification (NH16 Highway Corridor)
- Multi-Agency Emergency Alert Dispatch (Police 100, Ambulance 108, Fire 101, NHAI 1033)
- Public Citizen Incident Reporting with GPS & Photo Upload + Tracking
- Real-time Analytics & Export
"""
import csv
import datetime
import io
import os
import shutil
import sys
import uuid
import secrets

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
ML_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "ml"))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from database import (  # noqa: E402
    init_db, init_default_admin, verify_admin_credentials, change_admin_password,
    list_incidents, get_incident, update_incident_status, list_notifications, stats_summary, reset_db,
    insert_citizen_report, list_citizen_reports, get_citizen_report, update_citizen_report_fields,
    get_analytics,
)
from notification import dispatch_alert, dispatch_citizen_alert  # noqa: E402
from report_generator import generate_report, build_citizen_report  # noqa: E402
from geocode import reverse_geocode  # noqa: E402
from ml.predict import predict_row  # noqa: E402
from ml.preprocess import FEATURE_COLUMNS  # noqa: E402
from ml.india_stations import STATIONS  # noqa: E402
import simulate_stream  # noqa: E402

app = FastAPI(
    title="Hybrid AI Traffic Incident Detection & Emergency Response System",
    description="Unified Full-Stack API for Traffic Incident Detection, AI Analysis, "
                "Citizen Accident Reporting, and Multi-Agency Emergency Response.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database & Default Admin (admin / admin123)
init_db()
init_default_admin()

# ---------------------------------------------------------------------------
# Static files: uploaded accident photos + frontend application
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

# In-memory Active Sessions
# token -> {username, role, expires_at}
ACTIVE_SESSIONS: dict[str, dict] = {}


def get_current_admin(authorization: str | None = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    
    token = authorization.replace("Bearer ", "").strip()
    session = ACTIVE_SESSIONS.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please log in again.")
    
    if session["expires_at"] < datetime.datetime.now(datetime.timezone.utc):
        ACTIVE_SESSIONS.pop(token, None)
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    
    return session


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class StatusUpdateRequest(BaseModel):
    status: str


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
# Authentication Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = verify_admin_credentials(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    
    token = secrets.token_urlsafe(32)
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=12)
    ACTIVE_SESSIONS[token] = {
        "username": user["username"],
        "role": user["role"],
        "expires_at": expires,
    }
    return {
        "success": True,
        "token": token,
        "username": user["username"],
        "role": user["role"],
        "expires_at": expires.isoformat(),
    }


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(None)):
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        ACTIVE_SESSIONS.pop(token, None)
    return {"success": True, "message": "Logged out successfully"}


@app.get("/api/auth/me")
def get_me(admin: dict = Depends(get_current_admin)):
    return {
        "authenticated": True,
        "username": admin["username"],
        "role": admin["role"],
        "expires_at": admin["expires_at"].isoformat(),
    }


@app.post("/api/auth/change-password")
def change_password(req: ChangePasswordRequest, admin: dict = Depends(get_current_admin)):
    user = verify_admin_credentials(admin["username"], req.old_password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect current password.")
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters long.")
    
    success = change_admin_password(admin["username"], req.new_password)
    if not success:
        raise HTTPException(status_code=500, detail="Could not update password.")
    return {"success": True, "message": "Password updated successfully."}


# ---------------------------------------------------------------------------
# Corridor Stations & Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "system": "Hybrid AI Traffic Incident Detection & Emergency Response",
        "corridor": "NH16 Highway (Andhra Pradesh)",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@app.get("/api/stations")
def get_stations():
    """Returns the NH16 corridor monitoring stations (Vijayawada to Kavali)."""
    return STATIONS


@app.get("/api/stats")
def get_stats():
    return stats_summary()


@app.get("/api/analytics")
def get_analytics_data():
    return get_analytics()


# ---------------------------------------------------------------------------
# Accident Detection + AI Report + Emergency Notification
# ---------------------------------------------------------------------------
@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    Evaluates real sensor telemetry through the trained XGBoost model.
    Calculates severity, extracts AI incident parameters, and triggers
    emergency dispatch according to real input values and timestamp.
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

        record = {
            "detected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "station_pm": features.get("station_pm"),
            "location_name": features.get("location_name"),
            "district": features.get("district"),
            "state": features.get("state"),
            "road": features.get("road") or "NH16",
            "lat": features.get("lat"),
            "lon": features.get("lon"),
            "direction": features.get("direction") or "NB",
            "speed_kmph": features["speed_kmph"],
            "occupancy": features["occupancy"],
            "flow": features["flow"],
            "severity_code": result["severity_code"],
            "severity_label": result["severity_label"],
            "confidence": result["confidence"],
            "report": "",
            "source": "manual_sensor_test",
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
# Incidents & Notifications Management
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


@app.patch("/api/incidents/{incident_id}/status")
def patch_incident_status(incident_id: int, req: StatusUpdateRequest, admin: dict = Depends(get_current_admin)):
    incident = get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    update_incident_status(incident_id, req.status)
    return {"success": True, "id": incident_id, "status": req.status}


@app.delete("/api/incidents")
def clear_incidents(admin: dict = Depends(get_current_admin)):
    reset_db()
    return {"status": "cleared", "message": "Incident records reset."}


# ---------------------------------------------------------------------------
# Citizen Reporting (Public & Admin)
# ---------------------------------------------------------------------------
@app.get("/api/geocode")
def geocode(lat: float, lon: float):
    """Resolve GPS coordinates to a readable Indian address."""
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
    """Public endpoint for roadside accident photo & GPS submissions."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

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
        p for p in [record["village"] or record["town"], record["district"], record["state"]] if p
    )

    return {
        "id": report_id,
        "reference_no": f"CR-{report_id}",
        "reported_at": record["reported_at"],
        "address": address or "Not resolved",
        "report": report_text,
        "notifications": notifications,
        "notified_departments": [n["department"] for n in notifications],
        "status": "Reported",
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


@app.get("/api/citizen/track/{ref_no}")
def track_citizen_report(ref_no: str):
    """Allows any citizen to track status of their report using Reference ID (e.g. CR-1 or 1)."""
    clean_id = ref_no.upper().replace("CR-", "").strip()
    try:
        r_id = int(clean_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Reference Number format. Example: CR-101")
    
    report = get_citizen_report(r_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"No report found for Reference Number: {ref_no}")
    
    report["notifications"] = list_notifications(r_id, source_type="citizen_report")
    return report


@app.patch("/api/citizen/reports/{report_id}/status")
def patch_citizen_report_status(report_id: int, req: StatusUpdateRequest, admin: dict = Depends(get_current_admin)):
    report = get_citizen_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Citizen report not found")
    update_citizen_report_fields(report_id, status=req.status)
    return {"success": True, "id": report_id, "status": req.status}


# ---------------------------------------------------------------------------
# Data Exports (CSV)
# ---------------------------------------------------------------------------
@app.get("/api/export/incidents/csv")
def export_incidents_csv():
    incidents = list_incidents(1000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Incident ID", "Detected At", "Severity", "Confidence", "Speed (km/h)",
        "Occupancy", "Flow (veh/h)", "Location", "District", "State", "Road",
        "Direction", "Lat", "Lon", "Source", "Status"
    ])
    for r in incidents:
        writer.writerow([
            r.get("id"), r.get("detected_at"), r.get("severity_label"),
            r.get("confidence"), r.get("speed_kmph"), r.get("occupancy"),
            r.get("flow"), r.get("location_name"), r.get("district"),
            r.get("state"), r.get("road"), r.get("direction"),
            r.get("lat"), r.get("lon"), r.get("source"), r.get("status")
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=nh16_incident_logs.csv"},
    )


@app.get("/api/export/citizen/csv")
def export_citizen_csv():
    reports = list_citizen_reports(1000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Report ID", "Reference No", "Reported At", "Reporter Name", "Reporter Phone",
        "Severity", "Status", "Address", "Village/Town", "District", "State",
        "Pincode", "Lat", "Lon", "Description", "Image Path"
    ])
    for r in reports:
        writer.writerow([
            r.get("id"), f"CR-{r.get('id')}", r.get("reported_at"),
            r.get("reporter_name"), r.get("reporter_phone"), r.get("severity_label"),
            r.get("status"), r.get("display_address"), r.get("village") or r.get("town"),
            r.get("district"), r.get("state"), r.get("pincode"),
            r.get("lat"), r.get("lon"), r.get("description"), r.get("image_path")
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=citizen_accident_reports.csv"},
    )


# ---------------------------------------------------------------------------
# Real-time Sensor Replay Simulation
# ---------------------------------------------------------------------------
@app.post("/api/simulation/start")
def start_simulation(req: SimulationRequest, admin: dict = Depends(get_current_admin)):
    started = simulate_stream.start_background(speed=req.speed, limit=req.limit)
    if not started:
        raise HTTPException(status_code=409, detail="Simulation already running")
    return {"status": "started", **req.model_dump()}


@app.post("/api/simulation/stop")
def stop_simulation(admin: dict = Depends(get_current_admin)):
    simulate_stream.stop_background()
    return {"status": "stopping"}


@app.get("/api/simulation/status")
def simulation_status():
    return simulate_stream.get_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
