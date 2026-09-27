"""
preprocess.py
--------------
Data Collection + Data Preprocessing modules (as per the project's Module-Wise
Description slide).

Builds the traffic feature table used by the Accident Detection module
(speed, flow, occupancy and their short-window volatility -- the standard
predictors used in incident-detection literature such as the RF-RFE + LSTM
and FA + WRF papers referenced in the Literature Survey).

*** India localization note ***
The original project used the "Mobile Century" dataset -- real loop-detector
+ GPS probe-vehicle data recorded on I-880 in the San Francisco Bay Area,
California, USA. No public Indian sensor dataset with equivalent per-minute
speed/occupancy/flow granularity was available, so this module now
*synthesizes* a traffic dataset instead: realistic diurnal traffic patterns,
with randomly injected slow-down events (used as incident examples for
training), generated for each of the real, named locations in
`india_stations.py` along the NH16 highway corridor in Andhra Pradesh.
The synthetic generator is fully documented below and can be swapped for a
real Indian sensor feed (NHAI / state highway ITS data) with no changes
needed anywhere else in the project -- everything downstream only depends
on the column names in `FEATURE_COLUMNS` plus `station_pm`/`lat`/`lon`/
`location_name`/`district`/`state`/`road`/`direction`.

Run directly to build the processed feature table used for model training:
    python backend/ml/preprocess.py
"""
import os

import numpy as np
import pandas as pd

from india_stations import STATIONS

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
os.makedirs(PROCESSED_DIR, exist_ok=True)

ROLL_WINDOW = 5           # rolling window (in resampled steps) used for volatility features

# Free-flow conditions on this stretch of NH16 (a 4-lane national highway)
FREE_FLOW_SPEED_KMPH = 85.0
FREE_FLOW_FLOW = 42.0       # vehicles/minute/station baseline
FREE_FLOW_OCC = 0.12        # baseline lane occupancy fraction

RNG_SEED = 42


def _diurnal_speed_factor(hour: float) -> float:
    """Simple rush-hour dip: traffic is slower during the morning (8-11) and
    evening (17-21) peaks, and freest late at night."""
    morning_dip = np.exp(-0.5 * ((hour - 9.5) / 1.8) ** 2)
    evening_dip = np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
    dip = max(morning_dip, evening_dip)
    return 1.0 - 0.35 * dip


def _diurnal_flow_factor(hour: float) -> float:
    morning_peak = np.exp(-0.5 * ((hour - 9.5) / 1.8) ** 2)
    evening_peak = np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
    return 0.55 + 0.75 * max(morning_peak, evening_peak)


def _generate_station_series(station: dict, start: pd.Timestamp, n_minutes: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generates one station's per-minute speed/occupancy/flow series, with a
    handful of randomly-placed slow-down events (accidents, breakdowns,
    congestion build-ups) of varying severity so the labeling heuristic
    below has realistic incidents to detect."""
    timestamps = pd.date_range(start=start, periods=n_minutes, freq="min", tz="UTC")
    hours = timestamps.hour + timestamps.minute / 60.0

    speed_factor = np.array([_diurnal_speed_factor(h) for h in hours])
    flow_factor = np.array([_diurnal_flow_factor(h) for h in hours])

    speed = FREE_FLOW_SPEED_KMPH * speed_factor + rng.normal(0, 3.0, n_minutes)
    flow = FREE_FLOW_FLOW * flow_factor + rng.normal(0, 3.0, n_minutes)
    occupancy = FREE_FLOW_OCC / speed_factor + rng.normal(0, 0.01, n_minutes)

    # ---- Inject a few slow-down / incident-like events ----
    n_events = rng.integers(2, 6)
    for _ in range(n_events):
        event_start = rng.integers(0, max(n_minutes - 30, 1))
        duration = rng.integers(4, 25)
        severity_factor = rng.uniform(0.25, 0.9)  # how much speed drops (fraction)
        end = min(event_start + duration, n_minutes)
        seg_len = end - event_start
        half = max(seg_len // 2, 1)
        ramp_up = np.linspace(0, 1, half)
        ramp = np.concatenate([ramp_up, ramp_up[::-1]])
        if len(ramp) < seg_len:
            ramp = np.concatenate([ramp, np.zeros(seg_len - len(ramp))])
        else:
            ramp = ramp[:seg_len]
        speed[event_start:end] -= FREE_FLOW_SPEED_KMPH * severity_factor * ramp
        occupancy[event_start:end] += 0.35 * severity_factor * ramp
        flow[event_start:end] -= FREE_FLOW_FLOW * 0.3 * severity_factor * ramp

    speed = np.clip(speed, 2, 110)
    occupancy = np.clip(occupancy, 0.02, 0.98)
    flow = np.clip(flow, 0, 80)

    return pd.DataFrame({
        "timestamp": timestamps,
        "station_pm": station["station_pm"],
        "location_name": station["name"],
        "district": station["district"],
        "state": station["state"],
        "road": station["road"],
        "lat": station["lat"],
        "lon": station["lon"],
        "speed_kmph": speed,
        "occupancy": occupancy,
        "flow": flow,
    })


def build_feature_table(direction: str, n_days: int = 3, seed: int = RNG_SEED) -> pd.DataFrame:
    """Builds the per-minute feature table for one direction of travel
    (e.g. towards Vijayawada / towards Ongole) across every named station."""
    rng = np.random.default_rng(seed if direction == "NB" else seed + 1)
    start = pd.Timestamp.now("UTC").normalize() - pd.Timedelta(days=n_days)
    n_minutes = n_days * 24 * 60

    frames = [_generate_station_series(st, start, n_minutes, rng) for st in STATIONS]
    merged = pd.concat(frames, ignore_index=True)
    merged["direction"] = direction
    merged = merged.sort_values(["station_pm", "timestamp"])

    grp = merged.groupby("station_pm")
    merged["speed_roll_mean"] = grp["speed_kmph"].transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    merged["speed_roll_std"] = grp["speed_kmph"].transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).std().fillna(0))
    merged["speed_drop_pct"] = (merged["speed_roll_mean"] - merged["speed_kmph"]) / merged["speed_roll_mean"].replace(0, np.nan)
    merged["speed_drop_pct"] = merged["speed_drop_pct"].fillna(0).clip(lower=0)

    merged["occ_roll_mean"] = grp["occupancy"].transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    merged["occ_spike"] = (merged["occupancy"] - merged["occ_roll_mean"]).clip(lower=0)

    merged["flow_roll_mean"] = grp["flow"].transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
    merged["flow_drop_pct"] = (merged["flow_roll_mean"] - merged["flow"]) / merged["flow_roll_mean"].replace(0, np.nan)
    merged["flow_drop_pct"] = merged["flow_drop_pct"].fillna(0).clip(lower=0)

    merged["hour"] = merged["timestamp"].dt.hour
    return merged


# ---------------------------------------------------------------------------
# Incident labeling heuristic
# ---------------------------------------------------------------------------
# There's no ground-truth accident label for a synthetic dataset either, so
# -- exactly as the original project did with Mobile Century -- severity is
# derived with a transparent, documented anomaly-based heuristic (sudden
# speed drop + occupancy build-up relative to the recent rolling baseline):
#   0 = Normal    -> no notable anomaly
#   1 = Minor     -> moderate, short-lived speed drop
#   2 = Moderate  -> larger speed drop with occupancy build-up
#   3 = Major     -> severe speed drop (near-stationary traffic) + high occupancy
# Swap this out for real incident logs (state police / NHAI accident data)
# if you obtain them, everything downstream is unaffected.
def label_incidents(df):
    df = df.copy()
    conditions = [
        (df["speed_drop_pct"] >= 0.65) & (df["occ_spike"] >= 0.08),
        (df["speed_drop_pct"] >= 0.45) & (df["occ_spike"] >= 0.05),
        (df["speed_drop_pct"] >= 0.25) & (df["occ_spike"] >= 0.02),
    ]
    choices = [3, 2, 1]
    df["severity"] = np.select(conditions, choices, default=0)
    return df


FEATURE_COLUMNS = [
    "speed_kmph", "occupancy", "flow",
    "speed_roll_mean", "speed_roll_std", "speed_drop_pct",
    "occ_roll_mean", "occ_spike", "flow_roll_mean", "flow_drop_pct",
    "hour",
]


def build_dataset():
    all_rows = []
    for direction in ("NB", "SB"):
        feats = build_feature_table(direction)
        labeled = label_incidents(feats)
        all_rows.append(labeled)
        print(f"[{direction}] built {len(labeled)} feature rows across {len(STATIONS)} NH16 locations")

    full = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(PROCESSED_DIR, "traffic_features.csv")
    full.to_csv(out_path, index=False)
    print(f"Saved {len(full)} rows -> {out_path}")
    print(full["severity"].value_counts().sort_index())
    return full


if __name__ == "__main__":
    build_dataset()
