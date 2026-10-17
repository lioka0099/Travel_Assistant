from langchain_core.tools import tool

from .http import session
from .weather import geocode

OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{a_lon},{a_lat};{b_lon},{b_lat}"
# When no road connects two points, OSRM moves a point to the nearest reachable road
# (London → New York "snaps" New York to Portugal), so a big move means there's no real route.
MAX_SNAP_M = 10_000


def format_duration(seconds: float) -> str:
    """4740 -> '1 h 19 min', 900 -> '15 min'."""
    hours, minutes = divmod(round(seconds / 60), 60)
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"


def parse_route(data: dict) -> dict | None:
    """The first route from an OSRM response, or None if there's no real driving route."""
    if data.get("code") != "Ok" or not data.get("routes"):
        return None
    if any(w["distance"] > MAX_SNAP_M for w in data.get("waypoints", [])):
        return None
    return data["routes"][0]


@tool
def get_travel_time(origin: str, destination: str) -> dict:
    """Driving distance and time between two places, by car only (no flights, trains or ferries).

    For the user's own location, call get_user_location first and pass its city as origin.
    """
    a, b = geocode(origin), geocode(destination)
    if not a or not b:
        return {"summary": f"Couldn't find a place called {origin if not a else destination}"}
    # ponytail: public OSRM demo server (light use only); self-host or use a paid router if traffic grows
    r = session.get(
        OSRM_URL.format(a_lon=a["lon"], a_lat=a["lat"], b_lon=b["lon"], b_lat=b["lat"]),
        params={"overview": "false"},
        timeout=20,
    )
    route = parse_route(r.json())  # OSRM answers NoRoute with HTTP 400 + JSON, so don't raise_for_status
    if not route:
        return {"summary": f"No driving route between {a['name']} and {b['name']}"}
    duration = format_duration(route["duration"])
    return {
        "summary": f"Drive time {a['name']} → {b['name']}: {duration}",
        "origin": f"{a['name']}, {a['country']}",
        "destination": f"{b['name']}, {b['country']}",
        "distance_km": round(route["distance"] / 1000, 1),
        "duration_min": round(route["duration"] / 60),
        "duration": duration,
    }
