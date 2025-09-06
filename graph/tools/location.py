import requests
import math
from typing import Optional, Dict, Any

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
IP_GEOLOCATION_URL = "http://ip-api.com/json/"  # Free service, no API key needed

def get_client_ip_from_headers() -> Optional[str]:
    """Extract the real client IP from Streamlit's request headers.

    Tries X-Forwarded-For first (left-most IP), then falls back to the
    request's remote_ip attribute. Returns None if unavailable.
    """
    try:
        # Local imports to avoid hard dependency at module import time
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        from streamlit.web.server import Server  # type: ignore

        ctx = get_script_run_ctx()
        if not ctx:
            return None
        server = Server.get_current()
        if not server:
            return None
        # _get_session_info is internal; guard with try/except
        try:
            session_info = server._get_session_info(ctx.session_id)  # type: ignore[attr-defined]
        except Exception:
            session_info = None
        if not session_info or not getattr(session_info, "ws", None):
            return None
        req = session_info.ws.request
        # Prefer X-Forwarded-For when behind proxies/load balancers
        xff = req.headers.get("X-Forwarded-For") if getattr(req, "headers", None) else None
        if xff:
            return xff.split(",")[0].strip()
        # Fallback to remote_ip
        return getattr(req, "remote_ip", None)
    except Exception:
        return None

def get_location_from_ip() -> Optional[Dict[str, Any]]:
    """Get user's location from their IP address."""
    try:
        # Prefer the client IP from the incoming request headers if available
        client_ip = get_client_ip_from_headers()
        if client_ip:
            response = requests.get(f"{IP_GEOLOCATION_URL}{client_ip}", timeout=10)
        else:
            # Fallback: service will use the server IP
            response = requests.get(IP_GEOLOCATION_URL, timeout=10)
        response.raise_for_status()
        ip_data = response.json()
        
        if ip_data.get("status") == "success":
            # Extract location info
            city = ip_data.get("city", "")
            region = ip_data.get("regionName", "")
            country = ip_data.get("country", "")
            country_code = ip_data.get("countryCode", "")
            lat = ip_data.get("lat")
            lon = ip_data.get("lon")
            
            # Create a readable location string
            location_parts = [city, region, country]
            location_string = ", ".join([part for part in location_parts if part])
            
            return {
                "location_string": location_string,
                "city": city,
                "region": region,
                "country": country,
                "country_code": country_code,
                "latitude": lat,
                "longitude": lon,
                "ip": ip_data.get("query", client_ip or ""),
                "detected_via": "ip_geolocation"
            }
        else:
            print(f"IP geolocation failed: {ip_data.get('message', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"IP geolocation error: {e}")
        return None

def geocode_location(place: str) -> Optional[Dict[str, Any]]:
    """Geocode a location and return coordinates and details."""
    try:
        r = requests.get(GEOCODE_URL, params={"name": place, "count": 1}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data.get("results"):
            return None
        
        result = data["results"][0]
        return {
            "name": result["name"],
            "country": result.get("country"),
            "country_code": result.get("country_code"),
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "admin1": result.get("admin1"),  # State/Province
            "admin2": result.get("admin2"),  # County
        }
    except Exception as e:
        print(f"Geocoding failed for {place}: {e}")
        return None

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float, units: str = "metric") -> float:
    """Calculate distance between two points using Haversine formula."""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in kilometers
    r = 6371
    
    # Calculate distance
    distance_km = c * r
    
    if units == "imperial":
        # Convert to miles
        return distance_km * 0.621371
    else:
        return distance_km

def find_places_within_distance(user_location: str, max_distance: float, units: str = "metric") -> list:
    """Find places within a certain distance from user location."""
    # This would require a database of places with coordinates
    # For now, we'll return a placeholder that the LLM can work with
    return []

def get_travel_time_estimate(distance_km: float, transport_mode: str = "car") -> Dict[str, Any]:
    """Estimate travel time based on distance and transport mode."""
    # Rough estimates
    if transport_mode == "car":
        avg_speed_kmh = 80  # Average highway speed
    elif transport_mode == "train":
        avg_speed_kmh = 120  # Average train speed
    elif transport_mode == "plane":
        avg_speed_kmh = 800  # Average plane speed
    else:
        avg_speed_kmh = 60  # Default
    
    time_hours = distance_km / avg_speed_kmh
    
    return {
        "distance_km": distance_km,
        "time_hours": time_hours,
        "transport_mode": transport_mode,
        "estimated_time": f"{time_hours:.1f} hours"
    }
