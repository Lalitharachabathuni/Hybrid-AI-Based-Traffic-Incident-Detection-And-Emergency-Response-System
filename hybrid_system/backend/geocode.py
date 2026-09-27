"""
geocode.py
-----------
Location module for citizen-submitted accident reports.

Rural reporters usually cannot type an accurate address, so this module
converts the raw GPS coordinates captured from their phone browser into a
readable Indian address (village/town, mandal, district, state, PIN code)
using the free OpenStreetMap Nominatim reverse-geocoding service. No API
key or paid account is required.

If the lookup fails (no internet at that moment, service unavailable,
coordinates missing, etc.) this never raises -- it just returns {} and the
caller falls back to the raw GPS pin and/or whatever the reporter typed in
manually, so a report can never be blocked by a geocoding failure.

Note for production deployments: Nominatim's public server is rate-limited
(fair-use, ~1 request/second) and is fine for a pilot/demo. For higher
volume in a real district/state rollout, swap this for an Indian
government geocoding service (e.g. Survey of India / Bharatmaps) or a paid
provider -- every other module only talks to this file through
`reverse_geocode()`, so that's the only function that needs to change.
"""
import json
import urllib.parse
import urllib.request

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "HybridAI-TrafficIncidentSystem-India/1.0 (rural-accident-reporting)"


def reverse_geocode(lat: float | None, lon: float | None, timeout: float = 5.0) -> dict:
    """Best-effort reverse geocoding. Returns a dict with any of:
    village, town, mandal, district, state, pincode, road, display_address.
    Missing/unresolvable fields are simply absent from the dict.
    """
    if lat is None or lon is None:
        return {}

    params = {
        "format": "jsonv2",
        "lat": lat,
        "lon": lon,
        "zoom": 16,
        "addressdetails": 1,
    }
    url = f"{NOMINATIM_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # network down, rate-limited, malformed response, etc.
        print(f"[geocode] reverse geocode failed, falling back to raw GPS: {exc}")
        return {}

    addr = data.get("address", {}) or {}
    result = {
        "village": addr.get("village") or addr.get("hamlet"),
        "town": addr.get("town") or addr.get("city") or addr.get("suburb"),
        "mandal": addr.get("county") or addr.get("state_district"),
        "district": addr.get("state_district") or addr.get("county"),
        "state": addr.get("state"),
        "pincode": addr.get("postcode"),
        "road": addr.get("road"),
        "display_address": data.get("display_name"),
    }
    return {k: v for k, v in result.items() if v}
