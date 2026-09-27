"""
dashboard/app.py
------------------
Live Dashboard module (Streamlit), as specified in the Technologies slide
and the "Live Dashboard & Database" box in the System Architecture diagram.

Shows:
  - Live incidents on a map (color-coded by severity)
  - KPI cards (total incidents, breakdown by severity)
  - Incident feed table with AI-generated reports
  - Severity distribution + incident timeline charts
  - Controls to start/stop the real-time dataset-replay simulation via the
    FastAPI backend

Run (from the project root, with the backend already running):
    streamlit run dashboard/app.py
"""
import os
import sys
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

SEVERITY_COLORS = {
    "Normal": "#4CAF50",
    "Minor": "#FFC107",
    "Moderate": "#FF9800",
    "Major": "#F44336",
}

st.set_page_config(
    page_title="Hybrid AI Traffic Incident Detection & Emergency Response",
    page_icon="🚦",
    layout="wide",
)


def api_get(path, **kwargs):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Backend not reachable at {API_BASE}{path} ({exc}). "
                 f"Start it with: uvicorn main:app --port 8000 (from the backend/ folder).")
        return None


def api_post(path, json=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=json, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Request to {path} failed: {exc}")
        return None


def api_delete(path):
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Request to {path} failed: {exc}")
        return None


st.title("🚦 Hybrid AI-Based Traffic Incident Detection & Emergency Response System")
st.caption("Live monitoring dashboard — sensor analytics → XGBoost severity "
           "classification → AI report generation → automated emergency notification "
           "(Police 100 · Ambulance 108 · Fire 101)")

# ---------------------------------------------------------------------------
# Sidebar: simulation controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    st.write("Replay a live sensor feed along the NH16 (Andhra Pradesh) corridor.")

    speed = st.slider("Playback speed (x real-time)", min_value=10, max_value=5000, value=1000, step=10)
    limit = st.number_input("Rows to replay (0 = all)", min_value=0, value=3000, step=500)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True):
            api_post("/api/simulation/start", json={"speed": speed, "limit": limit or None})
    with col2:
        if st.button("⏹ Stop", use_container_width=True):
            api_post("/api/simulation/stop")

    status = api_get("/api/simulation/status")
    if status:
        st.metric("Rows processed", status.get("processed", 0))
        st.write("🟢 Running" if status.get("running") else "🔴 Stopped")

    st.divider()
    if st.button("🗑️ Clear all incidents", use_container_width=True):
        res = api_delete("/api/incidents")
        if res:
            st.success("Incidents cleared")

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh every 5s", value=True)

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
stats = api_get("/api/stats") or {"total_incidents": 0, "by_severity": {}}
by_sev = stats.get("by_severity", {})

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Incidents", stats.get("total_incidents", 0))
k2.metric("Minor", by_sev.get("Minor", 0))
k3.metric("Moderate", by_sev.get("Moderate", 0))
k4.metric("Major", by_sev.get("Major", 0))

st.divider()

# ---------------------------------------------------------------------------
# Fetch incidents
# ---------------------------------------------------------------------------
incidents = api_get("/api/incidents", params={"limit": 500}) or []
df = pd.DataFrame(incidents)

map_col, chart_col = st.columns([2, 1])

with map_col:
    st.subheader("📍 Live Incident Map")
    if not df.empty and df["lat"].notna().any():
        map_df = df.dropna(subset=["lat", "lon"]).copy()
        map_df["color"] = map_df["severity_label"].map(SEVERITY_COLORS)
        fig = px.scatter_mapbox(
            map_df, lat="lat", lon="lon",
            color="severity_label",
            color_discrete_map=SEVERITY_COLORS,
            hover_data=["location_name", "district", "direction", "speed_kmph", "confidence"],
            zoom=11, height=450,
        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No incidents yet. Start the simulation from the sidebar, or POST to "
                "/api/predict, to see incidents appear here.")

with chart_col:
    st.subheader("📊 Severity Distribution")
    if by_sev:
        pie_df = pd.DataFrame({"severity": list(by_sev.keys()), "count": list(by_sev.values())})
        fig2 = px.pie(pie_df, names="severity", values="count",
                       color="severity", color_discrete_map=SEVERITY_COLORS, hole=0.4)
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=280)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("No data yet.")

st.divider()

# ---------------------------------------------------------------------------
# Incident timeline
# ---------------------------------------------------------------------------
if not df.empty:
    st.subheader("⏱️ Incident Timeline")
    tdf = df.copy()
    tdf["detected_at"] = pd.to_datetime(tdf["detected_at"], errors="coerce")
    fig3 = px.scatter(
        tdf.sort_values("detected_at"), x="detected_at", y="speed_kmph",
        color="severity_label", color_discrete_map=SEVERITY_COLORS,
        size="confidence", hover_data=["location_name", "direction"],
        labels={"speed_kmph": "Speed at detection (km/h)", "detected_at": "Detected at"},
    )
    fig3.update_layout(height=320)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Incident feed table + report viewer
# ---------------------------------------------------------------------------
st.subheader("🧾 Incident Feed")
if df.empty:
    st.write("No incidents recorded yet.")
else:
    show_cols = ["id", "detected_at", "severity_label", "confidence", "speed_kmph",
                 "occupancy", "flow", "location_name", "district", "state", "direction", "source"]
    st.dataframe(df[show_cols].sort_values("id", ascending=False), use_container_width=True, height=300)

    incident_ids = df["id"].tolist()
    selected_id = st.selectbox("View AI-generated report for incident #", incident_ids)
    if selected_id:
        detail = api_get(f"/api/incidents/{selected_id}")
        if detail:
            st.code(detail.get("report", ""), language=None)
            st.write("**Emergency notifications dispatched:**")
            for n in detail.get("notifications", []):
                st.write(f"- 🚨 {n['department']} — sent at {n['sent_at']}")

if auto_refresh:
    time.sleep(5)
    st.rerun()
