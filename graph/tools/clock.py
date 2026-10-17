from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


@tool
def get_local_time(timezone: str) -> dict:
    """Current local date and time in an IANA timezone, e.g. 'Asia/Tokyo' or 'Europe/Paris'."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except (ZoneInfoNotFoundError, ValueError):
        return {"summary": f"Unknown timezone {timezone}"}
    return {"summary": f"Time in {timezone}: {now:%H:%M, %A %d %B}", "local_time": now.isoformat(timespec="minutes")}
