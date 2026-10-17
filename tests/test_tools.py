from graph.tools.clock import get_local_time
from graph.tools.countries import find_country
from graph.tools.location import get_user_location
from graph.tools.travel_time import format_duration, parse_route
from graph.tools.weather import pick_days, place_label

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


def country(cca2, cca3, common, official, *alt):
    return {"name": {"common": common, "official": official}, "cca2": cca2, "cca3": cca3, "altSpellings": [cca2, *alt]}


def test_find_country():
    # Names and order copy traps in the real mledoze dataset; Equatorial Guinea's "Guinea" spelling is made up.
    countries = [
        country("AW", "ABW", "Aruba", "Aruba"),  # first in the dataset, so an empty query used to match it
        country("AS", "ASM", "American Samoa", "American Samoa"),  # comes before United States
        country("GQ", "GNQ", "Equatorial Guinea", "Republic of Equatorial Guinea", "Guinea"),
        country("GN", "GIN", "Guinea", "Republic of Guinea"),
        country("GB", "GBR", "United Kingdom", "United Kingdom of Great Britain and Northern Ireland", "UK", "Great Britain"),
        country("IE", "IRL", "Ireland", "Republic of Ireland"),
        country("KR", "KOR", "South Korea", "Republic of Korea"),
        country("TR", "TUR", "Türkiye", "Republic of Türkiye", "Turkiye", "Republic of Turkey"),
        country("US", "USA", "United States", "United States of America", "USA", "United States of America"),
    ]
    expected = {
        "uk": "GB", "the UK": "GB", "Britain": "GB", "England": "GB", "Scotland": "GB", "Wales": "GB",
        "Northern Ireland": "GB", "Ireland": "IE", "Korea": "KR", "Turkey": "TR", "America": "US", "the USA": "US",
        "Guinea": "GN", "Narnia": None, "": None, "  ": None,
    }
    assert {q: (find_country(q, countries) or {}).get("cca2") for q in expected} == expected


def test_local_time_for_a_region_name_is_unknown():
    # "Asia" is a folder in the timezone database, so ZoneInfo raises IsADirectoryError
    assert get_local_time.invoke({"timezone": "Asia"})["summary"] == "Unknown timezone Asia"


def test_user_location_from_config():
    loc = {"location_string": "Tel Aviv, Israel", "city": "Tel Aviv", "region": "TA", "country": "Israel",
           "latitude": 32.08, "longitude": 34.78}
    out = get_user_location.invoke({}, config={"configurable": {"user_location": loc}})
    assert out["city"] == "Tel Aviv"
    assert out["summary"] == "Found your location: Tel Aviv, Israel"
    assert get_user_location.invoke({})["summary"] == "Location not available; ask the user where they are."


def test_place_label_includes_the_region():
    assert place_label({"name": "Springfield", "admin1": "Illinois", "country": "United States"}) == (
        "Springfield, Illinois, United States"
    )
    assert place_label({"name": "Tokyo", "admin1": "Tokyo", "country": "Japan"}) == "Tokyo, Japan"
    assert place_label({"name": "Singapore", "country": "Singapore"}) == "Singapore"
