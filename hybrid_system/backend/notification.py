"""
notification.py
-----------------
Emergency Notification module.

Implements the severity -> department escalation rules defined in the
"Proposed System" slide, using the Indian emergency-response structure:
    Minor    -> Local Traffic Police
    Moderate -> Local Police Station + Traffic Control Room
    Major    -> Police Control Room (100) + 108 Ambulance Service +
                Fire & Rescue Service (101) + Nearest Government Hospital +
                Traffic Control Room

By default this "sends" alerts by logging them to the console, to a local
log file, and to the notifications table in the database, which is enough
to demo and grade the module end-to-end with zero configuration. If SMTP
environment variables are provided (see .env.example) it will also send a
real email.
"""
import datetime
import os
import smtplib
from email.mime.text import MIMEText

from database import insert_notification

LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "notifications.log"))
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

SEVERITY_DEPARTMENTS = {
    "Minor": ["Local Traffic Police"],
    "Moderate": ["Local Police Station", "Traffic Control Room"],
    "Major": [
        "Police Control Room (100)",
        "108 Ambulance Service",
        "Fire & Rescue Service (101)",
        "Nearest Government Hospital",
        "Traffic Control Room",
    ],
}

# Citizen photo/GPS reports skip straight to responders who can act on the
# ground, since there's no sensor confirmation step for these -- and default
# to a safe (ambulance + police) response whenever the reporter isn't sure
# how serious it is, rather than under-reacting.
CITIZEN_REPORT_DEPARTMENTS = {
    "Minor": ["Local Traffic Police"],
    "Moderate": ["Local Police Station", "108 Ambulance Service"],
    "Major": [
        "Police Control Room (100)",
        "108 Ambulance Service",
        "Fire & Rescue Service (101)",
        "Nearest Government Hospital",
    ],
    "Not sure": ["Local Police Station", "108 Ambulance Service"],
}


def _send_email_if_configured(subject: str, body: str):
    host = os.environ.get("SMTP_HOST")
    if not host:
        return False
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    to_addr = os.environ.get("ALERT_EMAIL_TO", user)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr

    with smtplib.SMTP(host, port, timeout=10) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(user, [to_addr], msg.as_string())
    return True


def dispatch_alert(incident_id: int, severity_label: str, report: str, location: dict):
    """Notify every department mapped to this severity level."""
    if severity_label not in SEVERITY_DEPARTMENTS:
        return []

    departments = SEVERITY_DEPARTMENTS[severity_label]
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    results = []

    for dept in departments:
        location_str = ", ".join(
            b for b in [location.get("location_name"), location.get("district"), location.get("state")] if b
        ) or f"({location.get('lat')}, {location.get('lon')})"
        message = (
            f"[{severity_label.upper()} INCIDENT #{incident_id}] "
            f"Location: {location_str} | {location.get('road', '')} km {location.get('station_pm')} "
            f"| Direction: {location.get('direction')}\n{report}"
        )

        # 1. Console
        print(f"ALERT -> {dept}: {message}")

        # 2. Log file
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}\t{dept}\t{message}\n")

        # 3. Database
        insert_notification(incident_id, dept, message)

        # 4. Optional real email
        emailed = _send_email_if_configured(
            subject=f"[{severity_label} Incident #{incident_id}] Notify {dept}",
            body=message,
        )

        results.append({"department": dept, "message": message, "emailed": emailed})

    return results


def dispatch_citizen_alert(report_id: int, severity_label: str, report: str, location: dict):
    """Notify every responder mapped to a citizen-submitted photo/GPS report."""
    departments = CITIZEN_REPORT_DEPARTMENTS.get(
        severity_label, CITIZEN_REPORT_DEPARTMENTS["Not sure"]
    )
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    results = []

    address = location.get("display_address") or ", ".join(
        p for p in [location.get("village") or location.get("town"),
                    location.get("district"), location.get("state")] if p
    ) or f"GPS {location.get('lat')}, {location.get('lon')}"

    for dept in departments:
        message = (
            f"[CITIZEN REPORT #{report_id} - {severity_label.upper()}] "
            f"Location: {address} ({location.get('lat')}, {location.get('lon')})\n{report}"
        )

        print(f"ALERT -> {dept}: {message}")

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}\t{dept}\t{message}\n")

        insert_notification(report_id, dept, message, source_type="citizen_report")

        emailed = _send_email_if_configured(
            subject=f"[Citizen Report #{report_id} - {severity_label}] Notify {dept}",
            body=message,
        )

        results.append({"department": dept, "message": message, "emailed": emailed})

    return results
