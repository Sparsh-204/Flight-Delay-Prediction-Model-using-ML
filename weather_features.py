import os
from functools import lru_cache

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
DEFAULT_WEATHER = {
    "weather_wind": None,
    "weather_visibility": None,
    "weather_rain": 0.0,
    "weather_storm": 0.0,
}

# Extend this mapping with the airports you care about most for your demo.
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277),
    "ANC": (61.1744, -149.9964),
    "AUS": (30.1945, -97.6699),
    "BNA": (36.1245, -86.6782),
    "BOS": (42.3656, -71.0096),
    "BWI": (39.1754, -76.6684),
    "CLT": (35.2144, -80.9473),
    "DEN": (39.8561, -104.6737),
    "DFW": (32.8998, -97.0403),
    "DTW": (42.2162, -83.3554),
    "EWR": (40.6895, -74.1745),
    "HNL": (21.3245, -157.9251),
    "IAD": (38.9531, -77.4565),
    "IAH": (29.9902, -95.3368),
    "JFK": (40.6413, -73.7781),
    "LAS": (36.0840, -115.1537),
    "LAX": (33.9416, -118.4085),
    "MCO": (28.4312, -81.3081),
    "MIA": (25.7959, -80.2870),
    "MSP": (44.8848, -93.2223),
    "ORD": (41.9742, -87.9073),
    "PBI": (26.6832, -80.0956),
    "PDX": (45.5898, -122.5951),
    "PHX": (33.4342, -112.0116),
    "RDU": (35.8801, -78.7880),
    "SAN": (32.7338, -117.1933),
    "SEA": (47.4502, -122.3088),
    "SFO": (37.6213, -122.3790),
    "SLC": (40.7899, -111.9791),
    "STL": (38.7487, -90.3700),
    "TPA": (27.9755, -82.5332),
}


@lru_cache(maxsize=256)
def get_weather(airport_code: str) -> dict:
    normalized_airport = str(airport_code).upper().strip()
    coordinates = AIRPORT_COORDS.get(normalized_airport)
    if not API_KEY or coordinates is None:
        return DEFAULT_WEATHER.copy()

    lat, lon = coordinates
    url = "https://api.openweathermap.org/data/2.5/weather"

    try:
        response = requests.get(
            url,
            params={"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return DEFAULT_WEATHER.copy()

    weather_main = str(payload.get("weather", [{}])[0].get("main", "")).lower()
    return {
        "weather_wind": payload.get("wind", {}).get("speed"),
        "weather_visibility": payload.get("visibility"),
        "weather_rain": float("rain" in payload),
        "weather_storm": float(weather_main == "thunderstorm"),
    }


def enrich_weather_features(df: pd.DataFrame, airport_column: str = "ORIGIN_AIRPORT") -> pd.DataFrame:
    enriched = df.copy()
    weather_lookup = {
        airport: get_weather(airport)
        for airport in enriched[airport_column].dropna().astype(str).str.upper().unique()
    }
    weather_records = enriched[airport_column].apply(
        lambda airport: weather_lookup.get(str(airport).upper(), DEFAULT_WEATHER)
    )
    weather_df = pd.DataFrame(list(weather_records)).fillna(value=DEFAULT_WEATHER)
    return pd.concat([enriched.reset_index(drop=True), weather_df.reset_index(drop=True)], axis=1)