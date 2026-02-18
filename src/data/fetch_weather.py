"""
fetch_weather.py
----------------
Fetches daily weather forecasts AND historical data from Open-Meteo.
No API key required. Completely free.

Docs: https://open-meteo.com/en/docs
"""

import requests
from datetime import date, timedelta
from src.data.cities import CITY_COORDINATES

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# WMO weather codes we care about
# https://open-meteo.com/en/docs#weathervariables
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "weather_code",
]


def fetch_forecast(city: str, days: int = 7) -> list[dict]:
    """
    Fetch weather FORECAST for a city (up to 16 days ahead).

    Returns:
        list of dicts, one per day:
        {city, date, temp_max, temp_min, temp_mean,
         precipitation, wind_speed_max, wind_gusts_max, weather_code}
    """
    if city not in CITY_COORDINATES:
        raise ValueError(f"Unknown city: {city}")

    coords = CITY_COORDINATES[city]

    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/Paris",
        "forecast_days": days,
    }

    response = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    return _parse_daily_response(city, data)


def fetch_historical(city: str, start_date: date, end_date: date) -> list[dict]:
    """
    Fetch historical weather for a city between two dates.
    Uses the Open-Meteo Archive API (free, goes back to 1940).

    Returns same format as fetch_forecast.
    """
    if city not in CITY_COORDINATES:
        raise ValueError(f"Unknown city: {city}")

    coords = CITY_COORDINATES[city]

    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/Paris",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    return _parse_daily_response(city, data)


def _parse_daily_response(city: str, data: dict) -> list[dict]:
    """
    Parse the Open-Meteo daily response into a clean list of dicts.
    Handles both forecast and archive responses (same format).
    """
    daily = data.get("daily", {})
    dates = daily.get("time", [])

    if not dates:
        return []

    records = []
    for i, date_str in enumerate(dates):
        temp_max = _safe_float(daily.get("temperature_2m_max", [None])[i])
        temp_min = _safe_float(daily.get("temperature_2m_min", [None])[i])

        # Compute mean from max/min (Open-Meteo doesn't provide mean directly)
        if temp_max is not None and temp_min is not None:
            temp_mean = (temp_max + temp_min) / 2
        else:
            temp_mean = None

        records.append({
            "city":             city,
            "date":             date_str,          # string "YYYY-MM-DD"
            "temp_max":         temp_max,
            "temp_min":         temp_min,
            "temp_mean":        temp_mean,
            "precipitation":    _safe_float(daily.get("precipitation_sum", [None])[i]),
            "wind_speed_max":   _safe_float(daily.get("wind_speed_10m_max", [None])[i]),
            "wind_gusts_max":   _safe_float(daily.get("wind_gusts_10m_max", [None])[i]),
            "weather_code":     _safe_int(daily.get("weather_code", [None])[i]),
        })

    return records


def fetch_all_cities_forecast(days: int = 7) -> list[dict]:
    """
    Fetch forecast for ALL 20 cities. Called by the daily Airflow DAG.
    Returns flat list of records (city × days rows).
    """
    all_records = []
    for city in CITY_COORDINATES:
        try:
            records = fetch_forecast(city, days=days)
            all_records.extend(records)
            print(f"  ✓ {city}: {len(records)} days fetched")
        except Exception as e:
            print(f"  ✗ {city}: ERROR - {e}")
    return all_records


def fetch_all_cities_historical(start_date: date, end_date: date) -> list[dict]:
    """
    Fetch historical data for ALL 20 cities over a date range.
    Used for initial data backfill and model training.
    """
    all_records = []
    for city in CITY_COORDINATES:
        try:
            records = fetch_historical(city, start_date, end_date)
            all_records.extend(records)
            print(f"  ✓ {city}: {len(records)} days fetched")
        except Exception as e:
            print(f"  ✗ {city}: ERROR - {e}")
    return all_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    # Quick test: fetch forecast for Paris
    print("Fetching 7-day forecast for Paris...")
    records = fetch_forecast("Paris", days=7)
    for r in records:
        print(f"  {r['date']}: {r['temp_mean']:.1f}°C, "
              f"{r['precipitation']}mm rain, "
              f"{r['wind_speed_max']} km/h wind")
