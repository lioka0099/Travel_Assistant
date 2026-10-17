import os

from .http import session
from langchain_core.tools import tool

URL = "https://api.tavily.com/search"


@tool
def web_search(query: str) -> dict:
    """Search the web for current travel info: attractions, opening hours, events, prices, transport and trip ideas."""
    api_key = os.getenv("TAVILY_API_KEY")  # read per call, so keys loaded after import still work
    if not api_key:
        return {"summary": "Web search is not configured"}
    r = session.post(URL, headers={"Authorization": f"Bearer {api_key}"}, json={"query": query, "max_results": 5}, timeout=20)
    r.raise_for_status()
    results = [
        {"title": it.get("title", ""), "url": it.get("url", ""), "snippet": (it.get("content") or "")[:500]}
        for it in r.json().get("results", [])
    ]
    return {"summary": f"Searched the web for “{query}”", "results": results}
