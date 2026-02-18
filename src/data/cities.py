"""
cities.py
---------
Static city list + coordinate fetching via Nominatim (OpenStreetMap).
No API key required. Rate limit: 1 request/second (we respect this).
"""

import time
import requests

# ---------------------------------------------------------------------------
# City list
# ---------------------------------------------------------------------------

CITIES = [
    # North / Atlantic coast
    "Lille", "Rouen", "Brest", "Nantes", "La Rochelle",
    # Paris basin
    "Paris", "Reims", "Orléans",
    # East / Continental
    "Strasbourg", "Dijon", "Lyon",
    # Mountain influence
    "Grenoble", "Clermont-Ferrand",
    # South-West / Atlantic
    "Bordeaux", "Toulouse", "Bayonne",
    # Mediterranean / South-East
    "Marseille", "Montpellier", "Nice",
    # Island
    "Ajaccio",
]

# ---------------------------------------------------------------------------
# Coordinate fetching
# ---------------------------------------------------------------------------

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "france-weather-recommender/1.0 (learning project)"}
# Nominatim requires a User-Agent and max 1 req/sec


def fetch_coordinates(city: str, country: str = "France") -> dict | None:
    """
    Fetch lat/lon for a city using the Nominatim geocoding API.

    Returns:
        dict with keys: city, country, latitude, longitude
        None if the city was not found
    """
    params = {
        "q": f"{city}, {country}",
        "format": "json",
        "limit": 1,
    }
    try:
        response = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        results = response.json()

        if not results:
            print(f"  [WARNING] No results found for: {city}")
            return None

        best = results[0]
        return {
            "city": city,
            "country": country,
            "latitude": float(best["lat"]),
            "longitude": float(best["lon"]),
        }

    except requests.RequestException as e:
        print(f"  [ERROR] Failed to fetch coordinates for {city}: {e}")
        return None


def fetch_all_coordinates(cities: list[str] = CITIES, delay: float = 1.1) -> list[dict]:
    """
    Fetch coordinates for all cities, respecting Nominatim's rate limit.

    Args:
        cities: list of city names
        delay: seconds to wait between requests (Nominatim requires >= 1s)

    Returns:
        list of dicts with city, country, latitude, longitude
    """
    results = []
    print(f"Fetching coordinates for {len(cities)} cities...")

    for i, city in enumerate(cities):
        print(f"  [{i+1}/{len(cities)}] {city}...", end=" ")
        coord = fetch_coordinates(city)
        if coord:
            print(f"→ ({coord['latitude']:.4f}, {coord['longitude']:.4f})")
            results.append(coord)
        time.sleep(delay)  # respect rate limit

    print(f"\nDone. Got coordinates for {len(results)}/{len(cities)} cities.")
    return results


# ---------------------------------------------------------------------------
# Fallback: hardcoded coordinates (used if Nominatim is unavailable)
# These were fetched once and stored to avoid repeated API calls.
# ---------------------------------------------------------------------------

CITY_COORDINATES = {
    "Lille":              {"latitude": 50.6292,  "longitude":  3.0573},
    "Rouen":              {"latitude": 49.4432,  "longitude":  1.0993},
    "Brest":              {"latitude": 48.3905,  "longitude": -4.4860},
    "Nantes":             {"latitude": 47.2184,  "longitude": -1.5536},
    "La Rochelle":        {"latitude": 46.1591,  "longitude": -1.1520},
    "Paris":              {"latitude": 48.8566,  "longitude":  2.3522},
    "Reims":              {"latitude": 49.2583,  "longitude":  4.0317},
    "Orléans":            {"latitude": 47.9029,  "longitude":  1.9093},
    "Strasbourg":         {"latitude": 48.5734,  "longitude":  7.7521},
    "Dijon":              {"latitude": 47.3220,  "longitude":  5.0415},
    "Lyon":               {"latitude": 45.7640,  "longitude":  4.8357},
    "Grenoble":           {"latitude": 45.1885,  "longitude":  5.7245},
    "Clermont-Ferrand":   {"latitude": 45.7772,  "longitude":  3.0870},
    "Bordeaux":           {"latitude": 44.8378,  "longitude": -0.5792},
    "Toulouse":           {"latitude": 43.6047,  "longitude":  1.4442},
    "Bayonne":            {"latitude": 43.4929,  "longitude": -1.4748},
    "Marseille":          {"latitude": 43.2965,  "longitude":  5.3698},
    "Montpellier":        {"latitude": 43.6119,  "longitude":  3.8772},
    "Nice":               {"latitude": 43.7102,  "longitude":  7.2620},
    "Ajaccio":            {"latitude": 41.9192,  "longitude":  8.7386},
}


def get_cities_dataframe():
    """
    Return a pandas DataFrame with all cities and their coordinates.
    Uses hardcoded values (no API call) — safe to call from DAGs.
    """
    import pandas as pd
    rows = [
        {"city": city, "latitude": coords["latitude"], "longitude": coords["longitude"]}
        for city, coords in CITY_COORDINATES.items()
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Run this once to verify / refresh coordinates
    coords = fetch_all_coordinates()
    for c in coords:
        print(f'    "{c["city"]}": {{"latitude": {c["latitude"]:.4f}, "longitude": {c["longitude"]:.4f}}},')
