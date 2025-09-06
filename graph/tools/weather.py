import requests
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def geocode(place: str):
    r = requests.get(GEOCODE_URL, params={"name": place, "count": 1}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"): return None
    top = data["results"][0]
    return {
        "lat": top["latitude"], 
        "lon": top["longitude"], 
        "name": top["name"], 
        "country": top.get("country"),
        "country_code": top.get("country_code")
    }

def forecast_daily(lat: float, lon: float, units: str = "metric"):
    params = {
        "latitude": lat, 
        "longitude": lon,
        "daily": ["temperature_2m_max","temperature_2m_min","precipitation_probability_max"],
        "timezone": "auto",
        "temperature_unit": "celsius" if units == "metric" else "fahrenheit",
    }
    r = requests.get(FORECAST_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()
