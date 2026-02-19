"""
test_features.py
----------------
Unit tests for feature engineering functions.

Run with:
    pytest tests/test_features.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.engineer import (
    comfort_score,
    comfort_score_multi_profile,
    compute_all_profile_scores,
    compute_rolling_features,
    KMEANS_FEATURES,
    REGRESSION_FEATURES
)


# ============================================================================
# Test: comfort_score (leisure profile baseline)
# ============================================================================

def test_comfort_score_perfect_weather():
    """Perfect weather (20°C, no rain, no wind) should score ~100."""
    score = comfort_score(temp_mean=20, precipitation=0, wind_speed_max=0)
    assert 95 <= score <= 100, f"Expected ~100, got {score}"


def test_comfort_score_cold_rainy():
    """Cold rainy weather should score low."""
    score = comfort_score(temp_mean=5, precipitation=30, wind_speed_max=40)
    assert score < 30, f"Expected <30, got {score}"


def test_comfort_score_hot_dry():
    """Hot but dry weather should be moderate (temp penalty)."""
    score = comfort_score(temp_mean=35, precipitation=0, wind_speed_max=5)
    assert 40 <= score <= 60, f"Expected moderate score, got {score}"


def test_comfort_score_gaussian_temp():
    """Temperature score should peak at 20°C (Gaussian)."""
    score_ideal = comfort_score(20, 0, 0)
    score_lower = comfort_score(14, 0, 0)
    score_higher = comfort_score(26, 0, 0)
    
    assert score_ideal > score_lower
    assert score_ideal > score_higher
    assert abs(score_lower - score_higher) < 5  # Symmetric around 20°C


def test_comfort_score_handles_none():
    """Should handle None values gracefully."""
    score = comfort_score(temp_mean=None, precipitation=5, wind_speed_max=10)
    assert 0 <= score <= 100  # Should not crash


# ============================================================================
# Test: comfort_score_multi_profile (Phase 3)
# ============================================================================

def test_multi_profile_surfer_prefers_wind():
    """Surfer profile should score high with strong wind."""
    surfer_params = {
        'temp_weight': 30, 'temp_ideal': 18, 'temp_tolerance': 8,
        'rain_weight': 10, 'rain_decay': 10, 'rain_preference': 'neutral',
        'wind_weight': 60, 'wind_decay': 15, 'wind_preference': 'seek'
    }
    
    score_strong_wind = comfort_score_multi_profile(18, 5, 35, surfer_params)
    score_calm = comfort_score_multi_profile(18, 5, 5, surfer_params)
    
    assert score_strong_wind > score_calm + 20, "Surfer should prefer wind"


def test_multi_profile_cyclist_avoids_rain():
    """Cyclist profile should heavily penalize rain."""
    cyclist_params = {
        'temp_weight': 40, 'temp_ideal': 18, 'temp_tolerance': 5,
        'rain_weight': 40, 'rain_decay': 3, 'rain_preference': 'avoid',
        'wind_weight': 20, 'wind_decay': 20, 'wind_preference': 'avoid'
    }
    
    score_dry = comfort_score_multi_profile(18, 0, 10, cyclist_params)
    score_rainy = comfort_score_multi_profile(18, 20, 10, cyclist_params)
    
    assert score_dry > score_rainy + 30, "Cyclist should avoid rain strongly"


def test_multi_profile_skier_wants_cold():
    """Skier profile should prefer freezing temps."""
    skier_params = {
        'temp_weight': 40, 'temp_ideal': 0, 'temp_tolerance': 5,
        'rain_weight': 30, 'rain_decay': 10, 'rain_preference': 'seek',
        'wind_weight': 30, 'wind_decay': 30, 'wind_preference': 'neutral'
    }
    
    score_freezing = comfort_score_multi_profile(0, 10, 15, skier_params)
    score_warm = comfort_score_multi_profile(20, 10, 15, skier_params)
    
    assert score_freezing > score_warm + 20, "Skier should prefer freezing"


def test_compute_all_profile_scores():
    """Should compute scores for all profiles."""
    profiles_data = [
        {'profile_name': 'leisure', 'temp_weight': 50, 'temp_ideal': 20, 'temp_tolerance': 6,
         'rain_weight': 30, 'rain_decay': 5, 'rain_preference': 'avoid',
         'wind_weight': 20, 'wind_decay': 25, 'wind_preference': 'avoid'},
        {'profile_name': 'surfer', 'temp_weight': 30, 'temp_ideal': 18, 'temp_tolerance': 8,
         'rain_weight': 10, 'rain_decay': 10, 'rain_preference': 'neutral',
         'wind_weight': 60, 'wind_decay': 15, 'wind_preference': 'seek'},
    ]
    
    profiles_df = pd.DataFrame(profiles_data)
    
    scores = compute_all_profile_scores(
        temp_mean=18,
        precipitation=5,
        wind_speed_max=30,
        profiles_df=profiles_df
    )
    
    assert len(scores) == 2
    assert 'leisure' in scores
    assert 'surfer' in scores
    assert 0 <= scores['leisure'] <= 100
    assert 0 <= scores['surfer'] <= 100


# ============================================================================
# Test: compute_rolling_features
# ============================================================================

def test_rolling_features_basic():
    """Should compute 7-day and 3-day rolling windows."""
    # Create 10 days of weather data for 2 cities
    dates = [date.today() - timedelta(days=i) for i in range(10, 0, -1)]
    
    data = []
    for city in ['Paris', 'Nice']:
        for d in dates:
            data.append({
                'city': city,
                'date': d,
                'temp_mean': 15 + np.random.randn(),
                'precipitation': abs(np.random.randn() * 5),
                'wind_speed_max': abs(np.random.randn() * 10 + 15)
            })
    
    df = pd.DataFrame(data)
    
    features = compute_rolling_features(df, as_of_date=date.today())
    
    # Should have 2 rows (1 per city)
    assert len(features) == 2
    
    # Should have required columns
    assert all(col in features.columns for col in ['temp_mean_7d', 'temp_mean_3d'])
    assert all(col in features.columns for col in ['precip_sum_7d', 'precip_sum_3d'])
    assert all(col in features.columns for col in ['wind_max_7d', 'wind_max_3d'])
    assert 'comfort_score' in features.columns


def test_rolling_features_no_nulls():
    """Rolling features should not have nulls if enough data."""
    dates = [date.today() - timedelta(days=i) for i in range(14, 0, -1)]
    
    data = []
    for d in dates:
        data.append({
            'city': 'Paris',
            'date': d,
            'temp_mean': 15,
            'precipitation': 5,
            'wind_speed_max': 20
        })
    
    df = pd.DataFrame(data)
    features = compute_rolling_features(df, as_of_date=date.today())
    
    # With 14 days of data, 7-day window should have no nulls
    assert features['temp_mean_7d'].notna().all()
    assert features['precip_sum_7d'].notna().all()


def test_kmeans_features_list():
    """KMEANS_FEATURES should be the expected columns."""
    assert KMEANS_FEATURES == ['temp_mean_7d', 'precip_sum_7d', 'wind_max_7d']


def test_regression_features_list():
    """REGRESSION_FEATURES should be the expected columns."""
    assert REGRESSION_FEATURES == ['temp_mean_3d', 'precip_sum_3d', 'wind_max_3d']


# ============================================================================
# Test: Edge cases
# ============================================================================

def test_extreme_temperature():
    """Should handle extreme temperatures without crashing."""
    score_hot = comfort_score(temp_mean=50, precipitation=0, wind_speed_max=0)
    score_cold = comfort_score(temp_mean=-20, precipitation=0, wind_speed_max=0)
    
    assert 0 <= score_hot <= 100
    assert 0 <= score_cold <= 100
    # Extreme temps get ~0 temp score, but rain+wind can still contribute ~50 points
    assert score_hot < 60  # Max possible from rain+wind components
    assert score_cold < 60  # Max possible from rain+wind components


def test_extreme_precipitation():
    """Should handle extreme rain gracefully."""
    score = comfort_score(temp_mean=20, precipitation=200, wind_speed_max=10)
    
    assert 0 <= score <= 100
    # Extreme rain gets ~0 rain score, but temp(20°C) + wind(10km/h) still contribute ~65 points
    assert score < 70  # Should be lower due to rain, but temp+wind still contribute


def test_comfort_score_deterministic():
    """Same inputs should always give same outputs."""
    score1 = comfort_score(15, 10, 25)
    score2 = comfort_score(15, 10, 25)
    
    assert score1 == score2, "Comfort score should be deterministic"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
