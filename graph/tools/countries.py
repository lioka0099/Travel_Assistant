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


# Names the dataset doesn't list ("America" would otherwise partly match American Samoa), mapped to ISO codes.
ALIASES = {"america": "us", "england": "gb", "scotland": "gb", "wales": "gb", "northern ireland": "gb"}


def find_country(name: str, countries: list[dict]) -> dict | None:
    """Exact match on a name or ISO code, then on an alternative spelling; else the first partial match, same order.

    Dots are ignored ("U.S." is "US"), and only queries of 3+ characters partly match ("a" matched Aruba).
    """

    def norm(s):
        return s.lower().replace(".", "")

    q = norm(name).strip().removeprefix("the ").strip()
    q = ALIASES.get(q, q)
    if not q:
        return None

    def names(c):
        return [c["name"]["common"], c["name"]["official"]]

    tests = [
        lambda c: q in {norm(n) for n in [*names(c), c["cca2"], c["cca3"]]},
        lambda c: q in {norm(n) for n in c["altSpellings"]},
    ]
    if len(q) >= 3:
        tests += [
            lambda c: any(q in norm(n) for n in names(c)),
            lambda c: any(q in norm(n) for n in c["altSpellings"]),  # e.g. Turkey in "Republic of Turkey"
        ]
    return next((c for test in tests for c in countries if test(c)), None)


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
