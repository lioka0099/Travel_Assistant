import requests
BASE = "https://restcountries.com/v3.1/name/{name}"

def country_facts(name: str):
    url = BASE.format(name=name)
    params = {"fullText": "false", "fields": "name,currencies,languages,timezones,capital,idd"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    arr = r.json()
    if not arr: return None
    c = arr[0]
    return {
        "name": c["name"]["common"],
        "capital": (c.get("capital") or ["?"])[0],
        "currencies": list((c.get("currencies") or {}).keys()),
        "languages": list((c.get("languages") or {}).values()),
        "timezones": c.get("timezones", []),
        "dial": c.get("idd", {}).get("root", "") + (c.get("idd", {}).get("suffixes", [""])[0]),
    }
