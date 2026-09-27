"""
simulate_stream.py
--------------------
Since real live sensor hardware isn't available for this demo, this module
"replays" the synthesized traffic dataset (see preprocess.py) in timestamp
order at an adjustable speed to emulate the real-time monitoring pipeline
described in the PPT (Vehicle Detection & Tracking -> Speed Estimation ->
Accident Detection -> Information Extraction -> Emergency Notification ->
Dashboard).

Every row that the XGBoost model classifies as Minor/Moderate/Major is:
  1. written to the `incidents` table,
  2. turned into an AI report (report_generator),
  3. escalated to the correct departments (notification).

Run standalone:
    python backend/simulate_stream.py --speed 200 --limit 500

Or import `run_simulation()` / use the FastAPI endpoints to start it as a
background thread from the web UI / dashboard.
"""
import argparse
import os
import sys
import threading
import time

import pandas as pd

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
ML_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "ml"))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from database import init_db, insert_incident  # noqa: E402
from notification import dispatch_alert  # noqa: E402
from report_generator import generate_report  # noqa: E402
from predict import predict_row  # noqa: E402
from preprocess import FEATURE_COLUMNS, PROCESSED_DIR, build_dataset  # noqa: E402

_stop_event = threading.Event()
_thread = None
_status = {"running": False, "processed": 0, "incidents_found": 0}


def _process_row(row) -> dict | None:
    features = {c: float(row[c]) for c in FEATURE_COLUMNS}
    result = predict_row(features)

    if result["severity_label"] == "Normal":
        return None

    record = {
        "detected_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "station_pm": float(row["station_pm"]),
        "location_name": row.get("location_name"),
        "district": row.get("district"),
        "state": row.get("state"),
        "road": row.get("road"),
        "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
        "lon": float(row["lon"]) if pd.notna(row["lon"]) else None,
        "direction": row["direction"],
        "speed_kmph": float(row["speed_kmph"]),
        "occupancy": float(row["occupancy"]),
        "flow": float(row["flow"]),
        "severity_code": result["severity_code"],
        "severity_label": result["severity_label"],
        "confidence": result["confidence"],
        "report": "",
        "source": "sensor_simulation",
    }
    record["report"] = generate_report(record)
    incident_id = insert_incident(record)
    dispatch_alert(incident_id, result["severity_label"], record["report"], record)
    return {**record, "id": incident_id}


def run_simulation(speed: float = 500.0, limit: int = None, on_incident=None):
    """
    speed: playback multiplier (e.g. 500 = 500x faster than real time)
    limit: optional cap on number of rows processed (useful for a quick demo)
    on_incident: optional callback(incident_dict) invoked whenever an incident is stored
    """
    init_db()
    path = os.path.join(PROCESSED_DIR, "traffic_features.csv")
    if not os.path.exists(path):
        build_dataset()
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
    if limit:
        df = df.iloc[:limit]

    _status.update(running=True, processed=0, incidents_found=0)
    _stop_event.clear()

    prev_ts = None
    for _, row in df.iterrows():
        if _stop_event.is_set():
            break
        if prev_ts is not None:
            delta = (row["timestamp"] - prev_ts).total_seconds()
            time.sleep(max(0.0, min(delta, 60) / speed))
        prev_ts = row["timestamp"]

        incident = _process_row(row)
        _status["processed"] += 1
        if incident:
            _status["incidents_found"] += 1
            if on_incident:
                on_incident(incident)

    _status["running"] = False


def start_background(speed: float = 500.0, limit: int = None):
    global _thread
    if _status["running"]:
        return False
    _thread = threading.Thread(target=run_simulation, kwargs={"speed": speed, "limit": limit}, daemon=True)
    _thread.start()
    return True


def stop_background():
    _stop_event.set()


def get_status():
    return dict(_status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=500.0, help="Playback speed multiplier")
    parser.add_argument("--limit", type=int, default=2000, help="Number of rows to replay")
    args = parser.parse_args()
    run_simulation(speed=args.speed, limit=args.limit)
    print(get_status())
