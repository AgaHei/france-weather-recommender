"""
test_models.py
--------------
Unit tests for ML models (K-Means and Regression).

Run with:
    pytest tests/test_models.py -v
"""

import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.clustering import WeatherClusterModel, find_optimal_k
from src.models.regression import ComfortScoreModel, compare_models


# Helper functions
def create_sample_kmeans_data(n_samples=100):
    """Create synthetic weather data with 3 clusters."""
    np.random.seed(42)
    
    # Ensure we get exactly n_samples by distributing evenly
    n_per_cluster = n_samples // 3
    remainder = n_samples % 3
    
    # Mediterranean: warm, dry, calm
    n_med = n_per_cluster + (1 if remainder > 0 else 0)
    med = np.random.randn(n_med, 3) * [2, 1, 3] + [20, 2, 10]
    
    # Atlantic: cool, rainy, windy  
    n_atl = n_per_cluster + (1 if remainder > 1 else 0)
    atlantic = np.random.randn(n_atl, 3) * [3, 2, 5] + [12, 15, 25]
    
    # Continental: variable temps, moderate rain
    n_cont = n_per_cluster
    continental = np.random.randn(n_cont, 3) * [5, 1.5, 4] + [15, 8, 18]
    
    X = np.vstack([med, atlantic, continental])
    cities = [f"City_{i}" for i in range(len(X))]  # Match actual length
    
    return X, cities


def test_kmeans_model_trains():
    """K-Means model should train without errors."""
    X, cities = create_sample_kmeans_data()
    
    model = WeatherClusterModel(n_clusters=3)
    metrics = model.train(X, cities)
    
    assert 'silhouette_score' in metrics
    assert 'inertia' in metrics
    assert 'n_clusters' in metrics
    assert metrics['n_clusters'] == 3


def test_kmeans_silhouette_positive():
    """Silhouette score should be positive for good clusters."""
    X, cities = create_sample_kmeans_data()
    
    model = WeatherClusterModel(n_clusters=3)
    metrics = model.train(X, cities)
    
    assert metrics['silhouette_score'] > 0, "Silhouette should be positive"
    assert metrics['silhouette_score'] < 1, "Silhouette should be < 1"


def test_kmeans_predicts_correct_shape():
    """Predictions should have correct shape."""
    X_train, cities = create_sample_kmeans_data(90)
    X_test, _ = create_sample_kmeans_data(10)
    
    model = WeatherClusterModel(n_clusters=3)
    model.train(X_train, cities)
    
    predictions = model.predict(X_test)
    
    assert predictions.shape == (10,)
    assert predictions.min() >= 0
    assert predictions.max() < 3


def test_kmeans_cluster_stats():
    """Cluster stats should be computed correctly."""
    X, cities = create_sample_kmeans_data()
    
    model = WeatherClusterModel(n_clusters=3)
    metrics = model.train(X, cities[:10])  # Only first 10 cities
    
    assert 'cluster_stats' in metrics
    assert len(metrics['cluster_stats']) == 3
    
    for cluster_name, stats in metrics['cluster_stats'].items():
        assert 'size' in stats
        assert 'avg_temp' in stats
        assert 'avg_precip' in stats
        assert 'avg_wind' in stats


def test_kmeans_save_load():
    """Model should save and load correctly."""
    X, cities = create_sample_kmeans_data()
    
    model = WeatherClusterModel(n_clusters=3)
    model.train(X, cities)
    
    # Save
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
        model_path = f.name
    
    model.save(model_path)
    
    # Load
    loaded_model = WeatherClusterModel.load(model_path)
    
    # Predictions should match
    pred_original = model.predict(X[:10])
    pred_loaded = loaded_model.predict(X[:10])
    
    np.testing.assert_array_equal(pred_original, pred_loaded)
    
    # Cleanup
    os.remove(model_path)


def test_find_optimal_k():
    """find_optimal_k should return a reasonable k value."""
    X, _ = create_sample_kmeans_data()
    
    optimal_k = find_optimal_k(X, k_range=range(2, 6))
    
    assert 2 <= optimal_k <= 5, f"Optimal k should be in range, got {optimal_k}"


# ============================================================================
# Test: ComfortScoreModel (Regression)
# ============================================================================

def create_sample_regression_data(n_samples=500):
    """Create synthetic regression data."""
    np.random.seed(42)
    
    # Features: temp, precip, wind
    X = np.random.randn(n_samples, 3) * [5, 3, 10] + [15, 5, 20]
    
    # Target: comfort score (non-linear relationship)
    y = (
        50 * np.exp(-0.5 * ((X[:, 0] - 20) / 6) ** 2) +  # temp
        30 * np.exp(-X[:, 1] / 5) +                      # precip
        20 * np.exp(-X[:, 2] / 25) +                     # wind
        np.random.randn(n_samples) * 2                   # noise
    )
    y = np.clip(y, 0, 100)
    
    return X, y


def test_regression_model_trains():
    """Regression model should train without errors."""
    X, y = create_sample_regression_data()
    
    model = ComfortScoreModel(model_type='gradient_boosting')
    metrics = model.train(X, y, test_size=0.2)
    
    assert 'test_r2' in metrics
    assert 'test_rmse' in metrics
    assert 'train_r2' in metrics


def test_regression_minimum_accuracy():
    """Regression model should achieve minimum R² threshold."""
    X, y = create_sample_regression_data()
    
    model = ComfortScoreModel(model_type='gradient_boosting')
    metrics = model.train(X, y, test_size=0.2)
    
    assert metrics['test_r2'] > 0.7, f"Test R² should be > 0.7, got {metrics['test_r2']}"


def test_regression_predictions_in_range():
    """Predictions should be in [0, 100] range."""
    X, y = create_sample_regression_data()
    
    model = ComfortScoreModel(model_type='gradient_boosting')
    model.train(X, y, test_size=0.2)
    
    X_test, _ = create_sample_regression_data(50)
    predictions = model.predict(X_test)
    
    assert predictions.min() >= 0, "Predictions should be >= 0"
    assert predictions.max() <= 100, "Predictions should be <= 100"


def test_regression_feature_importances():
    """Gradient boosting should report feature importances."""
    X, y = create_sample_regression_data()
    
    model = ComfortScoreModel(model_type='gradient_boosting')
    metrics = model.train(X, y)
    
    assert 'feature_importances' in metrics
    assert len(metrics['feature_importances']) == 3


def test_regression_cross_validation():
    """Cross-validation should be performed."""
    X, y = create_sample_regression_data()
    
    model = ComfortScoreModel(model_type='gradient_boosting')
    metrics = model.train(X, y)
    
    assert 'cv_r2_mean' in metrics
    assert 'cv_r2_std' in metrics
    assert metrics['cv_r2_mean'] > 0


def test_regression_save_load():
    """Model should save and load correctly."""
    X, y = create_sample_regression_data()
    
    model = ComfortScoreModel(model_type='gradient_boosting')
    model.train(X, y)
    
    # Save
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
        model_path = f.name
    
    model.save(model_path)
    
    # Load
    loaded_model = ComfortScoreModel.load(model_path)
    
    # Predictions should match
    X_test, _ = create_sample_regression_data(10)
    pred_original = model.predict(X_test)
    pred_loaded = loaded_model.predict(X_test)
    
    np.testing.assert_array_almost_equal(pred_original, pred_loaded, decimal=5)
    
    # Cleanup
    os.remove(model_path)


def test_compare_models():
    """compare_models should evaluate multiple model types."""
    X, y = create_sample_regression_data()
    
    comparison_df = compare_models(X, y, test_size=0.2)
    
    assert len(comparison_df) == 3  # ridge, random_forest, gradient_boosting
    assert 'model' in comparison_df.columns
    assert 'test_r2' in comparison_df.columns
    assert 'test_rmse' in comparison_df.columns


def test_ridge_vs_gradient_boosting():
    """Gradient boosting should outperform ridge on non-linear data."""
    X, y = create_sample_regression_data()
    
    ridge = ComfortScoreModel(model_type='ridge')
    gb = ComfortScoreModel(model_type='gradient_boosting')
    
    ridge_metrics = ridge.train(X, y, test_size=0.2)
    gb_metrics = gb.train(X, y, test_size=0.2)
    
    assert gb_metrics['test_r2'] > ridge_metrics['test_r2'], \
        "Gradient boosting should beat ridge on non-linear data"


# ============================================================================
# Integration test: Full pipeline
# ============================================================================

def test_full_ml_pipeline():
    """Test the full pipeline: clustering + regression."""
    # Create data
    X_cluster, cities = create_sample_kmeans_data(60)
    X_regression, y = create_sample_regression_data(300)
    
    # Train clustering
    kmeans = WeatherClusterModel(n_clusters=3)
    kmeans_metrics = kmeans.train(X_cluster, cities)
    
    assert kmeans_metrics['silhouette_score'] > 0
    
    # Predict clusters
    clusters = kmeans.predict(X_cluster[:20])
    
    assert len(clusters) == 20
    
    # Train regression
    regression = ComfortScoreModel(model_type='gradient_boosting')
    regression_metrics = regression.train(X_regression, y)
    
    assert regression_metrics['test_r2'] > 0.7
    
    # Predict scores
    scores = regression.predict(X_regression[:10])
    
    assert len(scores) == 10
    assert all(0 <= s <= 100 for s in scores)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
