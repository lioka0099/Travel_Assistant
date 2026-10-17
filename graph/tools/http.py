import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Shared session that retries brief outages (Open-Meteo has returned 503 on back-to-back calls).
session = requests.Session()
session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)),
)
