"""
database.py
------------
Dashboard & Database module. Uses a local SQLite file so the whole project
runs on Windows with zero external services. (The PPT lists MongoDB /
Firebase Firestore as the production choice -- swapping the storage backend
only requires re-implementing the functions in this file; every other module
talks to the database exclusively through this interface.)
"""
import datetime
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "app.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS admin_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    role TEXT DEFAULT 'admin',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    station_pm REAL,
    location_name TEXT,
    district TEXT,
    state TEXT,
    road TEXT,
    lat REAL,
    lon REAL,
    direction TEXT,
    speed_kmph REAL,
    occupancy REAL,
    flow REAL,
    severity_code INTEGER,
    severity_label TEXT,
    confidence REAL,
    report TEXT,
    source TEXT DEFAULT 'sensor',
    status TEXT DEFAULT 'Active'
);

CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER,
    source_type TEXT DEFAULT 'incident',  -- 'incident' (sensor) or 'citizen_report'
    department TEXT,
    message TEXT,
    sent_at TEXT
);

-- Citizen-submitted reports: photos uploaded directly by the public
-- (mainly intended for rural users without access to traffic cameras/sensors)
-- via the /app/report.html web page, along with the GPS location captured
-- from their phone browser.
CREATE TABLE IF NOT EXISTS citizen_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reported_at TEXT NOT NULL,
    reporter_name TEXT,
    reporter_phone TEXT,
    lat REAL,
    lon REAL,
    village TEXT,
    town TEXT,
    mandal TEXT,
    district TEXT,
    state TEXT,
    pincode TEXT,
    road TEXT,
    display_address TEXT,
    description TEXT,
    severity_label TEXT,
    image_path TEXT,
    report TEXT,
    status TEXT DEFAULT 'Reported'
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_incident(record: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO incidents
               (detected_at, station_pm, location_name, district, state, road, lat, lon,
                direction, speed_kmph, occupancy, flow, severity_code, severity_label,
                confidence, report, source)
               VALUES (:detected_at, :station_pm, :location_name, :district, :state, :road,
                       :lat, :lon, :direction, :speed_kmph, :occupancy, :flow, :severity_code,
                       :severity_label, :confidence, :report, :source)""",
            record,
        )
        return cur.lastrowid


def insert_notification(incident_id: int, department: str, message: str, source_type: str = "incident"):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO notifications (incident_id, source_type, department, message, sent_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (incident_id, source_type, department, message, datetime.datetime.now(datetime.timezone.utc).isoformat()),
        )


def list_incidents(limit: int = 200):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_incident(incident_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        return dict(row) if row else None


def list_notifications(incident_id: int = None, source_type: str = "incident"):
    with get_conn() as conn:
        if incident_id is not None:
            rows = conn.execute(
                "SELECT * FROM notifications WHERE incident_id = ? AND source_type = ? ORDER BY id DESC",
                (incident_id, source_type),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM notifications ORDER BY id DESC LIMIT 200").fetchall()
        return [dict(r) for r in rows]


def stats_summary():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM incidents").fetchone()["c"]
        by_sev = conn.execute(
            "SELECT severity_label, COUNT(*) c FROM incidents GROUP BY severity_label"
        ).fetchall()
        return {"total_incidents": total, "by_severity": {r["severity_label"]: r["c"] for r in by_sev}}


def reset_db():
    with get_conn() as conn:
        conn.executescript("DROP TABLE IF EXISTS incidents; DROP TABLE IF EXISTS notifications;")
    init_db()


# ---------------------------------------------------------------------------
# Citizen-submitted accident reports (photo upload + GPS, for rural users)
# ---------------------------------------------------------------------------
def insert_citizen_report(record: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO citizen_reports
               (reported_at, reporter_name, reporter_phone, lat, lon, village, town, mandal,
                district, state, pincode, road, display_address, description, severity_label,
                image_path, report, status)
               VALUES (:reported_at, :reporter_name, :reporter_phone, :lat, :lon, :village, :town,
                       :mandal, :district, :state, :pincode, :road, :display_address, :description,
                       :severity_label, :image_path, :report, :status)""",
            record,
        )
        return cur.lastrowid


def update_citizen_report_fields(report_id: int, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k} = :{k}" for k in fields)
    fields["id"] = report_id
    with get_conn() as conn:
        conn.execute(f"UPDATE citizen_reports SET {cols} WHERE id = :id", fields)


def list_citizen_reports(limit: int = 200):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM citizen_reports ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_citizen_report(report_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM citizen_reports WHERE id = ?", (report_id,)).fetchone()
        return dict(row) if row else None


def reset_citizen_reports():
    with get_conn() as conn:
        conn.executescript("DROP TABLE IF EXISTS citizen_reports;")
    init_db()


# ---------------------------------------------------------------------------
# Admin User & Authentication Management
# ---------------------------------------------------------------------------
import hashlib
import secrets


def _hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
    return pw_hash, salt


def init_default_admin(default_username: str = "admin", default_password: str = "admin123"):
    with get_conn() as conn:
        user = conn.execute("SELECT * FROM admin_users WHERE username = ?", (default_username,)).fetchone()
        if not user:
            pw_hash, salt = _hash_password(default_password)
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO admin_users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (default_username, pw_hash, salt, "admin", now),
            )


def verify_admin_credentials(username: str, password: str) -> dict | None:
    with get_conn() as conn:
        user = conn.execute("SELECT * FROM admin_users WHERE username = ?", (username,)).fetchone()
        if not user:
            return None
        user_dict = dict(user)
        pw_hash, _ = _hash_password(password, user_dict["salt"])
        if secrets.compare_digest(pw_hash, user_dict["password_hash"]):
            return {"id": user_dict["id"], "username": user_dict["username"], "role": user_dict["role"]}
        return None


def change_admin_password(username: str, new_password: str) -> bool:
    with get_conn() as conn:
        user = conn.execute("SELECT * FROM admin_users WHERE username = ?", (username,)).fetchone()
        if not user:
            return False
        pw_hash, salt = _hash_password(new_password)
        conn.execute(
            "UPDATE admin_users SET password_hash = ?, salt = ? WHERE username = ?",
            (pw_hash, salt, username),
        )
        return True


def update_incident_status(incident_id: int, status: str):
    with get_conn() as conn:
        conn.execute("UPDATE incidents SET status = ? WHERE id = ?", (status, incident_id))


def get_analytics():
    with get_conn() as conn:
        total_inc = conn.execute("SELECT COUNT(*) c FROM incidents").fetchone()["c"]
        total_citizen = conn.execute("SELECT COUNT(*) c FROM citizen_reports").fetchone()["c"]
        by_sev = conn.execute(
            "SELECT severity_label, COUNT(*) c FROM incidents GROUP BY severity_label"
        ).fetchall()
        by_location = conn.execute(
            "SELECT location_name, COUNT(*) c FROM incidents WHERE location_name IS NOT NULL GROUP BY location_name ORDER BY c DESC LIMIT 10"
        ).fetchall()
        by_status = conn.execute(
            "SELECT status, COUNT(*) c FROM citizen_reports GROUP BY status"
        ).fetchall()
        recent_notifs = conn.execute(
            "SELECT department, COUNT(*) c FROM notifications GROUP BY department"
        ).fetchall()
        
        return {
            "total_incidents": total_inc,
            "total_citizen_reports": total_citizen,
            "by_severity": {r["severity_label"]: r["c"] for r in by_sev},
            "by_location": {r["location_name"]: r["c"] for r in by_location},
            "by_citizen_status": {r["status"]: r["c"] for r in by_status},
            "by_department_alerts": {r["department"]: r["c"] for r in recent_notifs},
        }


if __name__ == "__main__":
    init_db()
    init_default_admin()
    print(f"Initialized database at {DB_PATH}")
