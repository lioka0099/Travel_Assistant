import requests
import math
from typing import Optional, Dict, Any

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
BIGDATACLOUD_REVERSE_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"
IP_GEOLOCATION_URL = "http://ip-api.com/json/"  # Free service, no API key needed

def get_client_location_data(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
    """Get the client location data by latitude and longitude"""
    try:
        # 1) Reverse geocoding if coordinates
        if latitude is not None and longitude is not None:
            r = requests.get(
                BIGDATACLOUD_REVERSE_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "localityLanguage": "en",
                },
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            city = data.get("city") or data.get("locality") or ""
            region = data.get("principalSubdivision") or ""
            country = data.get("countryName") or ""
            country_code = data.get("countryCode") or ""
            lat = float(data.get("latitude", latitude))
            lon = float(data.get("longitude", longitude))
            location_parts = [city, region, country]
            location_string = ", ".join([p for p in location_parts if p])
            return {
                "location_string": location_string,
                "city": city,
                "region": region,
                "country": country,
                "country_code": country_code,
                "latitude": lat,
                "longitude": lon,
                "detected_via": "browser_geolocation",
            }
            
    except Exception as e:
        print(f"get client location data error: {e}")
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
