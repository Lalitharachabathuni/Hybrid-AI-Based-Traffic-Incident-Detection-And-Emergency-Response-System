"""
report_generator.py
---------------------
Information Extraction module: generates an AI-based accident report
(as required by Objective #4 in the PPT: "Generate an AI-based accident
report").

Two modes:
  1. Template mode (default, no setup required): builds a structured,
     human-readable report from the detected features.
  2. LLM mode (optional): if GEMINI_API_KEY is set in the environment and the
     'google-generativeai' package is installed, the template report is
     passed to Gemini to be rewritten as a natural-language incident
     narrative, mirroring the "Generative AI (LLM): Gemini 2.5" entry in the
     Technologies slide. If the key/package isn't available, this silently
     falls back to the template report so the system always runs.
"""
import datetime
import os

SEVERITY_LABELS = {0: "Normal", 1: "Minor", 2: "Moderate", 3: "Major"}

INDIA_EMERGENCY_NUMBERS_LINE = (
    "Emergency numbers: Police 100 | Ambulance 108 | Fire 101 | "
    "National Emergency Number 112"
)


def build_template_report(record: dict) -> str:
    ts = record.get("detected_at", datetime.datetime.now(datetime.timezone.utc).isoformat())

    location_bits = [record.get("location_name"), record.get("district"), record.get("state")]
    location_str = ", ".join(b for b in location_bits if b) or "Location not resolved"
    road = record.get("road")
    marker = record.get("station_pm")
    road_marker = f"{road} km {marker}" if road and marker is not None else (f"km {marker}" if marker is not None else "")

    return (
        f"ROAD TRAFFIC INCIDENT REPORT\n"
        f"Generated: {ts}\n"
        f"Location: {location_str}"
        + (f" ({road_marker})" if road_marker else "")
        + f"\nGPS Coordinates: {record.get('lat')}, {record.get('lon')}\n"
        f"Direction: {record.get('direction')}\n"
        f"Detected Severity: {record.get('severity_label')} (confidence {record.get('confidence', 0):.2f})\n"
        f"Traffic Conditions at Detection:\n"
        f"  - Speed: {record.get('speed_kmph', 0):.1f} km/h\n"
        f"  - Occupancy: {record.get('occupancy'):.3f}\n"
        f"  - Flow: {record.get('flow'):.1f} vehicles/interval\n"
        f"Summary: A {record.get('severity_label', 'Normal').lower()} traffic anomaly consistent with an "
        f"incident was detected by the hybrid AI pipeline (sensor analytics + XGBoost severity "
        f"classifier) near {record.get('location_name', 'the reported location')}. Recommended action: "
        f"dispatch the departments listed in the emergency notification for this severity level.\n"
        f"{INDIA_EMERGENCY_NUMBERS_LINE}"
    )


def build_citizen_report(record: dict) -> str:
    """Builds a clear, human-readable accident report from a citizen's photo
    upload + GPS location submitted via the public rural reporting website
    (frontend/report.html). Unlike the sensor pipeline above, there is no
    automatic AI severity classification here -- severity is whatever the
    reporter selected (or 'Not sure'), since a single photo alone isn't a
    reliable basis for an automated severity call. The report exists to get
    a complete, well-organised account of what was reported to the right
    responders as fast as possible.
    """
    ts = record.get("reported_at", datetime.datetime.now(datetime.timezone.utc).isoformat())

    address = record.get("display_address")
    if not address:
        parts = [
            record.get("village") or record.get("town"),
            record.get("mandal"),
            record.get("district"),
            record.get("state"),
            record.get("pincode"),
        ]
        address = ", ".join(p for p in parts if p)
    if not address:
        address = "Not resolved -- see GPS coordinates below"

    lat, lon = record.get("lat"), record.get("lon")
    gps_line = f"{lat}, {lon}" if lat is not None and lon is not None else "Not shared by reporter"

    severity = record.get("severity_label") or "Not sure"

    return (
        f"ROAD ACCIDENT REPORT (Citizen-Submitted)\n"
        f"Reference No: CR-{record.get('id', '—')}\n"
        f"Reported: {ts} (as captured by device)\n"
        f"Location: {address}\n"
        f"GPS Coordinates: {gps_line}\n"
        f"Reported Severity: {severity}\n"
        f"Description by reporter: {record.get('description') or 'Not provided'}\n"
        f"Photo evidence attached: {'Yes' if record.get('image_path') else 'No'}\n"
        f"Reporter contact: {record.get('reporter_phone') or 'Not shared'}\n\n"
        f"{INDIA_EMERGENCY_NUMBERS_LINE}\n\n"
        "This report has been forwarded to the responders listed for this severity "
        "level. If you are at the scene: stay safe, do not stand in the middle of "
        "the road, switch on hazard lights if a vehicle is involved, and if it is "
        "safe to do so, help guide traffic around the site until help arrives."
    )


def _try_gemini_rewrite(template_report: str) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "Rewrite the following structured traffic-incident report as a concise, "
            "professional narrative suitable for an emergency dispatcher. Keep all "
            "numeric facts unchanged.\n\n" + template_report
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as exc:  # network / quota / API errors -> fall back gracefully
        print(f"[report_generator] Gemini call failed, using template report instead: {exc}")
        return None


def generate_report(record: dict) -> str:
    template_report = build_template_report(record)
    llm_report = _try_gemini_rewrite(template_report)
    return llm_report if llm_report else template_report
