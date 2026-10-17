from .http import session
from langchain_core.tools import tool

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS = 16
DEFAULT_DAYS = 7


def geocode(place: str) -> dict | None:
    """Best match for a place name: name, country, lat, lon. None if not found."""
    r = session.get(GEOCODE_URL, params={"name": place, "count": 1}, timeout=20)
    r.raise_for_status()
    results = r.json().get("results")
    if not results:
        return None
    top = results[0]
    return {"name": top["name"], "country": top.get("country", ""), "lat": top["latitude"], "lon": top["longitude"]}


def pick_days(days: list[dict], start_date: str | None, end_date: str | None) -> list[dict]:
    """Days within [start_date, end_date] (ISO strings compare correctly); the first week if no dates."""
    if not start_date:
        return days[:DEFAULT_DAYS]
    end_date = end_date or start_date
    return [d for d in days if start_date <= d["date"] <= end_date]


@tool
def get_weather(place: str, start_date: str | None = None, end_date: str | None = None) -> dict:
    """Daily weather forecast for a place: max/min temperature in °C and chance of rain.

    Covers today up to 16 days ahead. Dates are ISO (YYYY-MM-DD); omit both for the next 7 days,
    or give only start_date for a single day. Use for weather, temperature, rain or what-to-wear questions.
    """
    g = geocode(place)
    if not g:
        return {"summary": f"Couldn't find a place called {place}"}
    r = session.get(
        FORECAST_URL,
        params={
            "latitude": g["lat"],
            "longitude": g["lon"],
            "timezone": "auto",
            "forecast_days": FORECAST_DAYS,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        },
        timeout=20,
    )
    r.raise_for_status()
    daily = r.json()["daily"]
    days = [
        {"date": d, "max_c": hi, "min_c": lo, "rain_chance_pct": rain}
        for d, hi, lo, rain in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_probability_max"],
        )
    ]
    where = f"{g['name']}, {g['country']}"
    picked = pick_days(days, start_date, end_date)
    if not picked:
        return {"summary": f"No forecast for those dates; forecasts cover {days[0]['date']} to {days[-1]['date']}", "place": where}
    return {"summary": f"Weather in {where} for {picked[0]['date']} to {picked[-1]['date']}", "place": where, "days": picked}
