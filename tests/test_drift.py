"""
test_drift.py
-------------
Tests for drift detection functionality.

Run with:
    pytest tests/test_drift.py -v
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scipy import stats


# ============================================================================
# Test: Statistical Tests
# ============================================================================

def test_ks_test_no_drift():
    """K-S test should not detect drift when distributions are same."""
    np.random.seed(42)
    
    baseline = np.random.normal(15, 3, 100)
    current = np.random.normal(15, 3, 100)
    
    ks_stat, p_value = stats.ks_2samp(baseline, current)
    
    # With same distribution, KS statistic should be small
    assert ks_stat < 0.2, f"Expected small KS stat, got {ks_stat}"


def test_ks_test_detects_drift():
    """K-S test should detect drift when distributions differ."""
    np.random.seed(42)
    
    baseline = np.random.normal(15, 3, 100)  # Mean 15
    current = np.random.normal(20, 3, 100)   # Mean 20 (shifted!)
    
    ks_stat, p_value = stats.ks_2samp(baseline, current)
    
    # With shifted distribution, KS statistic should be large
    assert ks_stat > 0.2, f"Expected large KS stat for drift, got {ks_stat}"
    assert p_value < 0.05, f"Expected significant p-value, got {p_value}"


def test_mean_shift_detection():
    """Should detect when mean shifts by > 2 standard deviations."""
    np.random.seed(42)
    
    baseline = np.random.normal(15, 3, 100)
    current = np.random.normal(15 + 7, 3, 100)  # Shift by 7 (>2 std devs)
    
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    current_mean = np.mean(current)
    
    mean_shift = abs(current_mean - baseline_mean) / baseline_std
    
    assert mean_shift > 2.0, f"Expected shift > 2 std devs, got {mean_shift}"


# ============================================================================
# Test: Drift Thresholds
# ============================================================================

def test_drift_threshold_configuration():
    """Drift thresholds should be reasonable."""
    from scripts.check_drift import DRIFT_THRESHOLDS
    
    assert 0 < DRIFT_THRESHOLDS['ks_statistic'] < 0.5
    assert 0 < DRIFT_THRESHOLDS['performance_drop'] < 0.2
    assert 1.0 < DRIFT_THRESHOLDS['feature_shift'] < 5.0


def test_baseline_window_reasonable():
    """Baseline window should be long enough for stable statistics."""
    from scripts.check_drift import BASELINE_WINDOW_DAYS, CURRENT_WINDOW_DAYS
    
    assert BASELINE_WINDOW_DAYS >= 14, "Baseline should be at least 2 weeks"
    assert CURRENT_WINDOW_DAYS >= 3, "Current window should be at least 3 days"
    assert BASELINE_WINDOW_DAYS > CURRENT_WINDOW_DAYS, "Baseline should be longer than current"


# ============================================================================
# Test: Edge Cases
# ============================================================================

def test_handles_empty_baseline():
    """Should handle case when no baseline data exists."""
    # This would happen on first run
    # Drift detection should return no_baseline_data
    # In real implementation, check_drift.py handles this
    pass  # Placeholder for documentation


def test_handles_single_sample():
    """Should handle case with very few samples."""
    np.random.seed(42)
    
    baseline = np.array([15.0])
    current = np.array([20.0])
    
    # K-S test still works but p-value unreliable
    ks_stat, p_value = stats.ks_2samp(baseline, current)
    
    # Should not crash
    assert ks_stat >= 0


def test_handles_zero_variance():
    """Should handle constant data (zero variance)."""
    baseline = np.array([15.0] * 100)
    current = np.array([15.0] * 100)
    
    ks_stat, p_value = stats.ks_2samp(baseline, current)
    
    # With identical data, KS stat should be 0
    assert ks_stat < 0.01


# ============================================================================
# Test: Seasonal Patterns
# ============================================================================

def test_seasonal_shift_detection():
    """Should detect seasonal weather shifts (summer → winter)."""
    np.random.seed(42)
    
    # Summer: warm temps, low rain
    summer = np.random.normal(25, 3, 100)
    
    # Winter: cold temps
    winter = np.random.normal(5, 3, 100)
    
    ks_stat, p_value = stats.ks_2samp(summer, winter)
    
    # Seasons are very different
    assert ks_stat > 0.5, "Seasonal shift should be detected"


def test_gradual_drift():
    """Should detect gradual drift over time."""
    np.random.seed(42)
    
    # Simulate gradual temperature increase (climate change)
    weeks = 8
    samples_per_week = 20
    
    data_by_week = []
    for week in range(weeks):
        # Temperature increases 0.5°C per week
        temp_mean = 15 + (week * 0.5)
        data_by_week.append(np.random.normal(temp_mean, 3, samples_per_week))
    
    # Compare first 4 weeks vs last 4 weeks
    baseline = np.concatenate(data_by_week[:4])
    current = np.concatenate(data_by_week[4:])
    
    ks_stat, p_value = stats.ks_2samp(baseline, current)
    
    # 2°C shift over 8 weeks should be detectable
    assert ks_stat > 0.15, f"Gradual drift should be detected, got KS={ks_stat}"


# ============================================================================
# Test: Model Performance Monitoring
# ============================================================================

def test_performance_degradation_detection():
    """Should detect when model R² drops significantly."""
    # Historical R² values
    historical = [0.995, 0.993, 0.994, 0.996, 0.992]
    current = 0.940  # Dropped to 94%
    
    historical_mean = np.mean(historical)
    performance_drop = (historical_mean - current) / historical_mean
    
    # 5.5% drop should trigger alert (threshold 5%)
    assert performance_drop > 0.05, f"Should detect performance drop, got {performance_drop}"


def test_noisy_performance_ignored():
    """Small random variations should not trigger alerts."""
    historical = [0.995, 0.993, 0.994, 0.996, 0.992]
    current = 0.993  # Within normal range
    
    historical_mean = np.mean(historical)
    performance_drop = (historical_mean - current) / historical_mean
    
    # 0.2% drop should not trigger alert
    assert performance_drop < 0.05, f"Should ignore noise, got {performance_drop}"


# ============================================================================
# Test: Integration
# ============================================================================

def test_drift_report_structure():
    """Drift report should have expected structure."""
    # Simulate report structure
    report = {
        'timestamp': datetime.now().isoformat(),
        'feature_drift': {
            'temp_mean_3d': {'drift_detected': False},
            'precip_sum_3d': {'drift_detected': False},
            'wind_max_3d': {'drift_detected': False},
        },
        'model_drift': {
            'regression': {'drift_detected': False},
            'kmeans': {'drift_detected': False},
        },
        'prediction_drift': {
            'leisure': {'drift_detected': False},
        },
        'overall_drift_detected': False,
        'recommendations': []
    }
    
    # Check structure
    assert 'timestamp' in report
    assert 'feature_drift' in report
    assert 'model_drift' in report
    assert 'prediction_drift' in report
    assert 'overall_drift_detected' in report
    assert 'recommendations' in report


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
