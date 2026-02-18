"""
train_models.py
---------------
Train K-Means clustering and comfort score regression models
on real weather data from Neon.

This is a standalone script you can run before building DAG 3.
It helps you verify the models work and see their performance.

Usage:
    python scripts/train_models.py
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.db import execute_query
from src.features.engineer import get_kmeans_matrix, get_regression_matrix
from src.models.clustering import WeatherClusterModel, find_optimal_k
from src.models.regression import ComfortScoreModel, compare_models


def load_features_from_neon() -> pd.DataFrame:
    """Load all computed features from Neon."""
    query = """
        SELECT 
            city,
            feature_date,
            temp_mean_7d,
            temp_mean_3d,
            precip_sum_7d,
            precip_sum_3d,
            wind_max_7d,
            wind_max_3d,
            comfort_score
        FROM weather_features
        WHERE temp_mean_7d IS NOT NULL
          AND temp_mean_3d IS NOT NULL
        ORDER BY feature_date DESC, city
    """
    
    data = execute_query(query)
    df = pd.DataFrame(data)
    
    print(f"📊 Loaded {len(df)} feature records from Neon")
    print(f"   Date range: {df['feature_date'].min()} to {df['feature_date'].max()}")
    print(f"   Cities: {df['city'].nunique()}")
    
    return df


def train_clustering_model(features_df: pd.DataFrame, save_path: str = None):
    """
    Train K-Means clustering model.
    
    Uses the LATEST feature_date for each city (current weather profile).
    """
    print("\n" + "="*70)
    print("TRAINING K-MEANS CLUSTERING MODEL")
    print("="*70)
    
    # Use only the most recent data (current weather profile)
    latest_date = features_df['feature_date'].max()
    recent_df = features_df[features_df['feature_date'] == latest_date].copy()
    
    print(f"\n📅 Using data from: {latest_date}")
    print(f"   {len(recent_df)} cities")
    
    # Extract feature matrix
    X, city_names = get_kmeans_matrix(recent_df)
    
    print(f"\n🔍 Finding optimal number of clusters...")
    optimal_k = find_optimal_k(X, k_range=range(2, 6))
    
    print(f"\n🔧 Training K-Means with k={optimal_k}...")
    model = WeatherClusterModel(n_clusters=optimal_k)
    metrics = model.train(X, city_names)
    
    print(f"\n✅ Training complete!")
    print(f"   Silhouette score: {metrics['silhouette_score']}")
    print(f"   Inertia: {metrics['inertia']}")
    
    print(f"\n📊 Cluster breakdown:")
    for cluster_name, stats in metrics['cluster_stats'].items():
        print(f"\n   {cluster_name} ({stats['size']} cities):")
        print(f"      Avg temp (7d): {stats['avg_temp']}°C")
        print(f"      Avg precip (7d): {stats['avg_precip']}mm")
        print(f"      Avg wind (7d): {stats['avg_wind']} km/h")
        
        if 'cities' in stats:
            cities_str = ', '.join(stats['cities'][:5])
            if len(stats['cities']) > 5:
                cities_str += f", ... ({len(stats['cities'])} total)"
            print(f"      Cities: {cities_str}")
    
    # Rank clusters by comfort
    ranked = model.rank_clusters_by_comfort(metrics['cluster_stats'])
    print(f"\n🏆 Clusters ranked best → worst: {ranked}")
    print(f"   (Cluster {ranked[0]} has the best weather conditions)")
    
    # Save model
    if save_path:
        model.save(save_path)
    
    return model, metrics


def train_regression_model(features_df: pd.DataFrame, save_path: str = None):
    """
    Train comfort score regression model.
    
    Uses ALL historical data for training (not just latest).
    """
    print("\n" + "="*70)
    print("TRAINING COMFORT SCORE REGRESSION MODEL")
    print("="*70)
    
    # Extract feature matrix and target
    X, y, city_names = get_regression_matrix(features_df)
    
    print(f"\n📊 Training data:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target range: {y.min():.1f} - {y.max():.1f}")
    print(f"   Target mean: {y.mean():.1f} ± {y.std():.1f}")
    
    # Compare all model types
    print(f"\n🔧 Comparing model types...")
    comparison_df = compare_models(X, y, test_size=0.2)
    
    # Train the best model (gradient boosting)
    best_model_type = comparison_df.iloc[0]['model']
    
    print(f"\n🏆 Best model: {best_model_type}")
    print(f"   Training final model...")
    
    model = ComfortScoreModel(model_type=best_model_type)
    metrics = model.train(X, y, test_size=0.2)
    
    print(f"\n✅ Training complete!")
    print(f"   Test RMSE: {metrics['test_rmse']:.2f} points")
    print(f"   Test MAE: {metrics['test_mae']:.2f} points")
    print(f"   Test R²: {metrics['test_r2']:.3f}")
    print(f"   CV R²: {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
    
    if 'feature_importances' in metrics:
        print(f"\n📊 Feature importances:")
        for feat, imp in sorted(metrics['feature_importances'].items(), 
                                key=lambda x: x[1], reverse=True):
            print(f"   {feat}: {imp:.3f}")
    
    # Save model
    if save_path:
        model.save(save_path)
    
    return model, metrics


def test_models_on_today(clustering_model, regression_model, features_df: pd.DataFrame):
    """
    Run both models on today's data to see recommendations.
    """
    print("\n" + "="*70)
    print("TESTING MODELS ON TODAY'S DATA")
    print("="*70)
    
    # Get latest data
    latest_date = features_df['feature_date'].max()
    today_df = features_df[features_df['feature_date'] == latest_date].copy()
    
    print(f"\n📅 Date: {latest_date}")
    
    # K-Means clustering
    X_kmeans, cities_kmeans = get_kmeans_matrix(today_df)
    cluster_labels = clustering_model.predict(X_kmeans)
    
    today_df['cluster'] = cluster_labels
    
    # Regression predictions
    X_reg, y_true, cities_reg = get_regression_matrix(today_df)
    y_pred = regression_model.predict(X_reg)
    
    today_df['predicted_score'] = y_pred
    today_df['actual_score'] = y_true
    today_df['prediction_error'] = np.abs(y_pred - y_true)
    
    # Show results
    print(f"\n🏆 Top 5 recommended cities (by predicted score):")
    top_cities = today_df.nlargest(5, 'predicted_score')
    
    for i, row in enumerate(top_cities.itertuples(), 1):
        print(f"   {i}. {row.city}")
        print(f"      Cluster: {row.cluster}")
        print(f"      Predicted score: {row.predicted_score:.1f}/100")
        print(f"      Actual score: {row.actual_score:.1f}/100")
        print(f"      Error: {row.prediction_error:.1f} points")
        print()
    
    # Model accuracy on today
    mae = today_df['prediction_error'].mean()
    rmse = np.sqrt((today_df['prediction_error'] ** 2).mean())
    
    print(f"📊 Today's prediction accuracy:")
    print(f"   MAE: {mae:.2f} points")
    print(f"   RMSE: {rmse:.2f} points")


def main():
    """Main training pipeline."""
    print("="*70)
    print("WEATHER MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Load data
    features_df = load_features_from_neon()
    
    if len(features_df) < 100:
        print("\n⚠️  WARNING: Less than 100 samples found.")
        print("   Run backfill_historical_weather.py first to get more training data.")
        return
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mlflow', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Train clustering
    clustering_model, clustering_metrics = train_clustering_model(
        features_df,
        save_path=os.path.join(models_dir, 'kmeans_latest.joblib')
    )
    
    # Train regression
    regression_model, regression_metrics = train_regression_model(
        features_df,
        save_path=os.path.join(models_dir, 'regression_latest.joblib')
    )
    
    # Test on today's data
    test_models_on_today(clustering_model, regression_model, features_df)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nModels saved to: mlflow/models/")
    print("   - kmeans_latest.joblib")
    print("   - regression_latest.joblib")
    print("\nNext steps:")
    print("   1. Check the model performance above")
    print("   2. Build DAG 3 (weekly retraining)")
    print("   3. Build DAG 4 (daily recommendations)")


if __name__ == "__main__":
    main()
