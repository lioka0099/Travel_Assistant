from datetime import datetime
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TIMEZONE = "Asia/Jerusalem"

def now_iso(tz: str = DEFAULT_TIMEZONE) -> str:
    return datetime.now(ZoneInfo(tz)).isoformat(timespec="seconds")

def today(tz: str | None = None) -> str:
    tz = tz or DEFAULT_TIMEZONE
    try:
        return datetime.now(ZoneInfo(tz)).date().isoformat()
    except ZoneInfoNotFoundError:
        # fallback to default if an invalid tz was passed
        return datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).date().isoformat()