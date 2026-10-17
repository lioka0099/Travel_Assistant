import requests
from typing import Optional, Dict, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

BIGDATACLOUD_REVERSE_URL = "https://api.bigdatacloud.net/data/reverse-geocode-client"
"""Reverse geocoding endpoint to map coordinates to a human-readable location."""

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


@tool
def get_user_location(config: RunnableConfig) -> dict:
    """The user's current location (city, region, country, coordinates), detected by their browser.

    Use for "near me", "from here" or "X hours away" questions.
    """
    loc = config.get("configurable", {}).get("user_location")
    if not loc:
        return {"summary": "Location not available; ask the user where they are."}
    return {
        "summary": f"Found your location: {loc['location_string']}",
        **{k: loc[k] for k in ("city", "region", "country", "latitude", "longitude")},
    }
