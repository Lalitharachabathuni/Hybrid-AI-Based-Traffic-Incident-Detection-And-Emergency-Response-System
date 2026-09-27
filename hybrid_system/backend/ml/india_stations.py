"""
india_stations.py
-------------------
Real, named locations for the sensor-based Accident Detection pipeline,
replacing the original Mobile Century project's California (I-880, Bay
Area) station list with a genuine Indian highway corridor.

Corridor used: NH16 (the old NH5 / Golden Quadrilateral highway) through
Andhra Pradesh, covering the Vijayawada - Guntur - Tenali - Ponnur -
Chinaganjam - Ongole stretch. `station_pm` is the approximate distance
(km) along the corridor, used the same way the original "postmile" was
used, but every station also carries its real place name, district, state
and road number so reports and the dashboard can show an actual location
instead of a bare number.

Coordinates are approximate (town-centre level), which is sufficient for
a demo/dashboard; swap in exact NHAI chainage/GPS survey data for a real
deployment.
"""

STATIONS = [
    {"station_pm": 0.0,   "name": "Vijayawada",      "district": "NTR District",  "state": "Andhra Pradesh", "road": "NH16", "lat": 16.5062, "lon": 80.6480},
    {"station_pm": 8.0,   "name": "Penamaluru",       "district": "NTR District",  "state": "Andhra Pradesh", "road": "NH16", "lat": 16.4650, "lon": 80.6270},
    {"station_pm": 16.0,  "name": "Mangalagiri",      "district": "Guntur",        "state": "Andhra Pradesh", "road": "NH16", "lat": 16.4300, "lon": 80.5500},
    {"station_pm": 27.0,  "name": "Guntur",           "district": "Guntur",        "state": "Andhra Pradesh", "road": "NH16", "lat": 16.3067, "lon": 80.4365},
    {"station_pm": 38.0,  "name": "Tenali",           "district": "Guntur",        "state": "Andhra Pradesh", "road": "NH16", "lat": 16.2428, "lon": 80.6400},
    {"station_pm": 52.0,  "name": "Ponnur",           "district": "Guntur",        "state": "Andhra Pradesh", "road": "NH16", "lat": 16.0667, "lon": 80.5500},
    {"station_pm": 64.0,  "name": "Chinaganjam",      "district": "Bapatla",       "state": "Andhra Pradesh", "road": "NH16", "lat": 15.9167, "lon": 80.3667},
    {"station_pm": 78.0,  "name": "Bapatla",          "district": "Bapatla",       "state": "Andhra Pradesh", "road": "NH16", "lat": 15.9050, "lon": 80.4680},
    {"station_pm": 95.0,  "name": "Chirala",          "district": "Bapatla",       "state": "Andhra Pradesh", "road": "NH16", "lat": 15.8250, "lon": 80.3550},
    {"station_pm": 112.0, "name": "Ongole",           "district": "Prakasam",      "state": "Andhra Pradesh", "road": "NH16", "lat": 15.5057, "lon": 80.0499},
    {"station_pm": 130.0, "name": "Kanigiri Junction","district": "Prakasam",      "state": "Andhra Pradesh", "road": "NH16", "lat": 15.4080, "lon": 79.8890},
    {"station_pm": 148.0, "name": "Kavali",           "district": "SPSR Nellore",  "state": "Andhra Pradesh", "road": "NH16", "lat": 14.9122, "lon": 79.9942},
]
