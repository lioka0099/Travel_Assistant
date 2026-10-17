from functools import cache

from langchain_core.tools import tool

from .http import session

# REST Countries retired its free v3.1 API (v5 needs an API key), so we load the open
# mledoze/countries dataset it was built on, pinned to a release, once per process.
DATA_URL = "https://cdn.jsdelivr.net/gh/mledoze/countries@5.1.0/countries.json"


@cache
def _countries() -> list[dict]:
    r = session.get(DATA_URL, timeout=30)
    r.raise_for_status()
    return r.json()


def find_country(name: str, countries: list[dict]) -> dict | None:
    """Exact match on name, official name, ISO code or alternative spelling; else the first partial name match."""
    q = name.strip().lower()
    for c in countries:
        names = {c["name"]["common"], c["name"]["official"], c["cca2"], c["cca3"], *c["altSpellings"]}
        if q in {n.lower() for n in names}:
            return c
    return next((c for c in countries if q in c["name"]["common"].lower()), None)


@tool
def get_country_info(country: str) -> dict:
    """Facts about a country: capital, currencies, languages, region and dialing code."""
    c = find_country(country, _countries())
    if not c:
        return {"summary": f"Couldn't find a country called {country}"}
    idd = c.get("idd") or {}
    suffixes = idd.get("suffixes") or [""]
    return {
        "summary": f"Looked up {c['name']['common']}",
        "name": c["name"]["common"],
        "capital": ", ".join(c.get("capital") or []) or "none",
        "currencies": [f"{v['name']} ({code}, {v.get('symbol', '')})" for code, v in (c.get("currencies") or {}).items()],
        "languages": list((c.get("languages") or {}).values()),
        "region": c.get("subregion") or c.get("region", ""),
        "dialing_code": idd.get("root", "") + (suffixes[0] if len(suffixes) == 1 else ""),
    }
