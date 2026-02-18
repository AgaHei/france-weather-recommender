"""
engineer.py
-----------
Feature engineering from raw weather data.

Two main responsibilities:
1. Compute rolling window features (7-day, 3-day) per city
2. Compute the comfort_score label used to train the regression model

The comfort score is our "ground truth" label — a human-designed function
that encodes what makes a good weekend destination:
  - Temperature: Gaussian peak at 20°C
  - Rain: exponential penalty
  - Wind: exponential penalty
  Max possible score: ~100
"""

import math
import pandas as pd
from datetime import date


# ---------------------------------------------------------------------------
# Comfort score formula
# ---------------------------------------------------------------------------

def comfort_score(
    temp_mean: float,
    precipitation: float,
    wind_speed_max: float,
    temp_ideal: float = 20.0,
    temp_sigma: float = 6.0,
    rain_decay: float = 5.0,
    wind_decay: float = 25.0,
) -> float:
    """
    Compute a comfort score between 0 and ~100.

    Components:
        - Temperature (0-50 pts): Gaussian centered on temp_ideal (20°C)
          Full score at 20°C, drops to ~30pts at 14°C or 26°C
        - Rain penalty (0-30 pts): Exponential decay with precipitation
          Full score at 0mm, ~18pts at 5mm, ~11pts at 10mm
        - Wind penalty (0-20 pts): Exponential decay with wind speed
          Full score at 0 km/h, ~12pts at 25 km/h, ~7pts at 50 km/h

    Args:
        temp_mean: mean temperature in °C
        precipitation: total rainfall in mm
        wind_speed_max: max wind speed in km/h
        temp_ideal: target temperature (default 20°C)
        temp_sigma: temperature tolerance (default 6°C)
        rain_decay: rain scale factor (lower = harsher penalty)
        wind_decay: wind scale factor (lower = harsher penalty)

    Returns:
        float score in range [0, ~100]
    """
    # Handle None/NaN values gracefully
    if temp_mean is None or math.isnan(temp_mean):
        temp_score = 25.0  # neutral
    else:
        temp_score = 50.0 * math.exp(-0.5 * ((temp_mean - temp_ideal) / temp_sigma) ** 2)

    if precipitation is None or math.isnan(precipitation):
        rain_score = 15.0  # neutral
    else:
        rain_score = 30.0 * math.exp(-precipitation / rain_decay)

    if wind_speed_max is None or math.isnan(wind_speed_max):
        wind_score = 10.0  # neutral
    else:
        wind_score = 20.0 * math.exp(-wind_speed_max / wind_decay)

    return round(temp_score + rain_score + wind_score, 2)


# ---------------------------------------------------------------------------
# Rolling feature computation
# ---------------------------------------------------------------------------

def compute_rolling_features(df: pd.DataFrame, as_of_date: date | None = None) -> pd.DataFrame:
    """
    Compute rolling window features from raw weather data.

    Input DataFrame (from raw_weather table) must have columns:
        city, date, temp_mean, precipitation, wind_speed_max

    Returns a DataFrame with one row per (city, feature_date) containing:
        city, feature_date,
        temp_mean_7d, temp_mean_3d,
        precip_sum_7d, precip_sum_3d,
        wind_max_7d, wind_max_3d,
        comfort_score
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"])

    if as_of_date is None:
        as_of_date = df["date"].max().date()

    feature_rows = []

    for city, city_df in df.groupby("city"):
        city_df = city_df.set_index("date").sort_index()

        # Rolling 7-day features (for K-Means: historical weather profile)
        temp_7d  = city_df["temp_mean"].rolling("7D").mean()
        precip_7d = city_df["precipitation"].rolling("7D").sum()
        wind_7d  = city_df["wind_speed_max"].rolling("7D").max()

        # Rolling 3-day features (for regression: near-future forecast)
        temp_3d  = city_df["temp_mean"].rolling("3D").mean()
        precip_3d = city_df["precipitation"].rolling("3D").sum()
        wind_3d  = city_df["wind_speed_max"].rolling("3D").max()

        # Get the latest values (as of as_of_date or last available date)
        latest_date = city_df.index.max()

        score = comfort_score(
            temp_mean=temp_3d.iloc[-1] if not pd.isna(temp_3d.iloc[-1]) else None,
            precipitation=precip_3d.iloc[-1] if not pd.isna(precip_3d.iloc[-1]) else None,
            wind_speed_max=wind_3d.iloc[-1] if not pd.isna(wind_3d.iloc[-1]) else None,
        )

        feature_rows.append({
            "city":             city,
            "feature_date":     as_of_date,
            "temp_mean_7d":     round(float(temp_7d.iloc[-1]), 2) if not pd.isna(temp_7d.iloc[-1]) else None,
            "temp_mean_3d":     round(float(temp_3d.iloc[-1]), 2) if not pd.isna(temp_3d.iloc[-1]) else None,
            "precip_sum_7d":    round(float(precip_7d.iloc[-1]), 2) if not pd.isna(precip_7d.iloc[-1]) else None,
            "precip_sum_3d":    round(float(precip_3d.iloc[-1]), 2) if not pd.isna(precip_3d.iloc[-1]) else None,
            "wind_max_7d":      round(float(wind_7d.iloc[-1]), 2) if not pd.isna(wind_7d.iloc[-1]) else None,
            "wind_max_3d":      round(float(wind_3d.iloc[-1]), 2) if not pd.isna(wind_3d.iloc[-1]) else None,
            "comfort_score":    score,
        })

    return pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# Feature matrix for ML models
# ---------------------------------------------------------------------------

# Features used by K-Means (historical profile — who is this city, climatically?)
KMEANS_FEATURES = ["temp_mean_7d", "precip_sum_7d", "wind_max_7d"]

# Features used by the regression model (near-future forecast — how good is this weekend?)
REGRESSION_FEATURES = ["temp_mean_3d", "precip_sum_3d", "wind_max_3d"]

# Target variable
REGRESSION_TARGET = "comfort_score"


def get_kmeans_matrix(features_df: pd.DataFrame) -> tuple:
    """
    Extract the feature matrix for K-Means clustering.

    Returns:
        (X: np.ndarray, city_names: list[str])
    """
    clean = features_df[["city"] + KMEANS_FEATURES].dropna()
    X = clean[KMEANS_FEATURES].values
    cities = clean["city"].tolist()
    return X, cities


def get_regression_matrix(features_df: pd.DataFrame) -> tuple:
    """
    Extract feature matrix and target for regression training.

    Returns:
        (X: np.ndarray, y: np.ndarray, city_names: list[str])
    """
    clean = features_df[["city"] + REGRESSION_FEATURES + [REGRESSION_TARGET]].dropna()
    X = clean[REGRESSION_FEATURES].values
    y = clean[REGRESSION_TARGET].values
    cities = clean["city"].tolist()
    return X, y, cities


if __name__ == "__main__":
    # Quick test of the comfort score formula
    print("Comfort score examples:")
    print(f"  Perfect weekend (20°C, 0mm, 0 km/h):  {comfort_score(20, 0, 0):.1f}/100")
    print(f"  Warm & windy    (25°C, 0mm, 50 km/h): {comfort_score(25, 0, 50):.1f}/100")
    print(f"  Rainy autumn    (14°C, 20mm, 30 km/h): {comfort_score(14, 20, 30):.1f}/100")
    print(f"  Cold & wet      (5°C, 30mm, 40 km/h): {comfort_score(5, 30, 40):.1f}/100")
    print(f"  Hot summer      (35°C, 0mm, 10 km/h):  {comfort_score(35, 0, 10):.1f}/100")
