from graph.tools.travel_time import format_duration, parse_route
from graph.tools.weather import pick_days

DAYS = [{"date": f"2026-10-{d}", "max_c": 25, "min_c": 18, "rain_chance_pct": 0} for d in range(17, 33)]


def test_pick_days_defaults_to_one_week():
    assert [d["date"] for d in pick_days(DAYS, None, None)] == [f"2026-10-{d}" for d in range(17, 24)]


def test_pick_days_range_and_single_day():
    assert [d["date"] for d in pick_days(DAYS, "2026-10-23", "2026-10-24")] == ["2026-10-23", "2026-10-24"]
    assert [d["date"] for d in pick_days(DAYS, "2026-10-20", None)] == ["2026-10-20"]


def test_pick_days_out_of_range_is_empty():
    assert pick_days(DAYS, "2027-01-01", None) == []


def test_format_duration():
    assert format_duration(4740) == "1 h 19 min"
    assert format_duration(900) == "15 min"


def test_parse_route():
    route = {"distance": 94500, "duration": 4740}
    ok = {"code": "Ok", "routes": [route], "waypoints": [{"distance": 3}, {"distance": 25}]}
    assert parse_route(ok) == route
    assert parse_route({"code": "NoRoute", "message": "Impossible route"}) is None
    snapped_far = {"code": "Ok", "routes": [route], "waypoints": [{"distance": 3}, {"distance": 5_534_807}]}
    assert parse_route(snapped_far) is None  # London → New York
