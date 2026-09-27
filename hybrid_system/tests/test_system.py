import io
import os
import sys
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
ML_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "ml"))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from main import app
from database import list_incidents, stats_summary
from predict import predict_row
from geocode import reverse_geocode

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json().get("status") == "ok"

def test_stats(client):
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_incidents" in data
    assert "by_severity" in data

def test_predict_incident(client):
    payload = {
        "speed_kmph": 8.0,
        "occupancy": 0.65,
        "flow": 55.0,
        "speed_roll_mean": 82.0,
        "speed_drop_pct": 0.9,
        "occ_spike": 0.45,
        "station_pm": 27.0,
        "location_name": "Guntur",
        "district": "Guntur",
        "state": "Andhra Pradesh",
        "road": "NH16",
        "lat": 16.3067,
        "lon": 80.4365,
        "direction": "NB",
        "notify": True
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["severity_label"] in ["Minor", "Moderate", "Major"]
    assert "report" in data
    assert "notifications" in data

def test_citizen_report_submission(client):
    img_bytes = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    files = {"image": ("accident.png", img_bytes, "image/png")}
    data = {
        "village": "Mangalagiri",
        "district": "Guntur",
        "state": "Andhra Pradesh",
        "severity_label": "Severe",
        "description": "Vehicle skid near bypass",
        "reporter_name": "Test Reporter",
        "reporter_phone": "9876543210"
    }
    res = client.post("/api/citizen/report", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()
    assert "reference_no" in res_data
    assert "report" in res_data
    assert len(res_data.get("notifications", [])) > 0

def test_simulation_status(client):
    res = client.get("/api/simulation/status")
    assert res.status_code == 200
    data = res.json()
    assert "running" in data
    assert "processed" in data
