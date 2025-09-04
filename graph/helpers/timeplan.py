from datetime import date, timedelta
from typing import List, Tuple, Optional

# Weekend day indices: Monday=0 ... Sunday=6
DEFAULT_WEEKEND: Tuple[int, int] = (5, 6)  # Sat, Sun

# Minimal table for places where weekend are not Sat–Sun.
WEEKEND_BY_COUNTRY: dict[str, Tuple[int, int]] = {
    "IL": (4, 5),  # Israel: Fri, Sat
    "SA": (4, 5),  # Saudi Arabia: Fri, Sat
    "QA": (4, 5),  # Qatar
    "KW": (4, 5),  # Kuwait
    "BH": (4, 5),  # Bahrain
    "OM": (4, 5),  # Oman
    "JO": (4, 5),  # Jordan
    "EG": (4, 5),  # Egypt
}

def weekend_for_country(country_code: Optional[str]) -> Tuple[int, int]:
    if not country_code:
        return DEFAULT_WEEKEND
    return WEEKEND_BY_COUNTRY.get(country_code.upper(), DEFAULT_WEEKEND)

def next_weekend(base: date, weekend: Tuple[int, int] = DEFAULT_WEEKEND) -> List[date]:
    """Return the upcoming weekend days for the given weekend tuple (start_idx, end_idx)."""
    start_idx, end_idx = weekend
    # distance from base.weekday() to start_idx
    delta = (start_idx - base.weekday()) % 7
    start_day = base + timedelta(days=delta)
    if end_idx == start_idx:
        return [start_day]
    end_delta = (end_idx - start_idx)
    end_day = start_day + timedelta(days=end_delta)
    return [start_day, end_day]

def resolve_relative_dates(target_type: str, base_iso: str, weekend: Tuple[int, int] = DEFAULT_WEEKEND) -> List[str]:
    """Resolve 'today'/'tomorrow'/'weekend' to ISO dates using base date and localized weekend."""
    base = date.fromisoformat(base_iso)
    if target_type == "today":
        return [base.isoformat()]
    if target_type == "tomorrow":
        return [(base + timedelta(days=1)).isoformat()]
    if target_type == "weekend":
        days = next_weekend(base, weekend)
        return [d.isoformat() for d in days]
    return [base.isoformat()] 
