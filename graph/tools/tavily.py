import os, requests
BASE = "https://api.tavily.com/search"
API_KEY = os.environ.get("TAVILY_API_KEY")

def web_search(query: str, max_results: int = 5):
    if not API_KEY:
        return {"error": "Missing TAVILY_API_KEY"}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    body = {"query": query, "max_results": max_results}
    r = requests.post(BASE, headers=headers, json=body, timeout=20)
    r.raise_for_status()
    return r.json()
