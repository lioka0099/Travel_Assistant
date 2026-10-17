from .clock import get_local_time
from .countries import get_country_info
from .location import get_user_location
from .tavily import web_search
from .travel_time import get_travel_time
from .weather import get_weather

TOOLS = [get_weather, get_country_info, web_search, get_local_time, get_user_location, get_travel_time]
