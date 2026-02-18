#!/usr/bin/env python3
"""
run_recommendations.py
----------------------
Standalone script to generate daily recommendations without Airflow.
Loads trained models and generates weekend destination recommendations.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.db import execute_query, bulk_insert
from src.features.engineer import get_kmeans_matrix, get_regression_matrix
from src.models.clustering import WeatherClusterModel
from src.models.regression import ComfortScoreModel

# Local models directory 
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'mlflow', 'models')


def load_champion_models():
    """
    Load the trained models from disk.
    Uses 'latest' models since we don't have champions yet.
    """
    kmeans_path = os.path.join(MODELS_DIR, 'kmeans_latest.joblib')
    regression_path = os.path.join(MODELS_DIR, 'regression_latest.joblib')
    
    # Check if models exist
    if not os.path.exists(kmeans_path):
        raise FileNotFoundError(
            f"No K-Means model found at {kmeans_path}. "
            "Run 'python scripts/train_models.py' first."
        )
    
    if not os.path.exists(regression_path):
        raise FileNotFoundError(
            f"No regression model found at {regression_path}. "
            "Run 'python scripts/train_models.py' first."
        )
    
    print("📂 Loading trained models...")
    kmeans_model = WeatherClusterModel.load(kmeans_path)
    regression_model = ComfortScoreModel.load(regression_path)
    
    return kmeans_model, regression_model


def load_today_features():
    """
    Load today's weather features from Neon.
    """
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
        WHERE feature_date = CURRENT_DATE
          AND temp_mean_7d IS NOT NULL
          AND temp_mean_3d IS NOT NULL
        ORDER BY city
    """
    
    data = execute_query(query)
    
    if not data:
        raise ValueError(
            "No feature data for today! Run feature engineering first:\\n"
            "python scripts/run_feature_engineering.py"
        )
    
    df = pd.DataFrame(data)
    print(f"📊 Loaded features for {len(df)} cities on {df['feature_date'].iloc[0]}")
    
    return df


def generate_recommendations():
    """
    Main recommendation generation logic.
    
    Two-stage process:
    1. K-Means clustering (coarse filter)
    2. Regression scoring (fine ranking)
    """
    print("\n" + "="*70)
    print("GENERATING DAILY RECOMMENDATIONS")
    print("="*70)
    
    # Load models and data
    kmeans_model, regression_model = load_champion_models()
    today_df = load_today_features()
    
    # Stage 1: K-Means Clustering (coarse filter)
    print("\n🔍 Stage 1: Clustering cities by weather profile...")
    
    X_kmeans, city_names_kmeans = get_kmeans_matrix(today_df)
    cluster_labels = kmeans_model.predict(X_kmeans)
    
    today_df['cluster_id'] = cluster_labels
    
    # Compute cluster statistics
    cluster_stats = {}
    for cluster_id in range(kmeans_model.n_clusters):
        mask = cluster_labels == cluster_id
        cluster_X = X_kmeans[mask]
        
        cluster_stats[cluster_id] = {
            'size': int(mask.sum()),
            'avg_temp': float(cluster_X[:, 0].mean()),
            'avg_precip': float(cluster_X[:, 1].mean()),
            'avg_wind': float(cluster_X[:, 2].mean()),
            'cities': [city_names_kmeans[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        }
    
    print("\n📊 Cluster breakdown:")
    for cluster_id, stats in cluster_stats.items():
        cities_str = ', '.join(stats['cities'][:3])
        if len(stats['cities']) > 3:
            cities_str += f" (+{len(stats['cities']) - 3} more)"
        
        print(f"   Cluster {cluster_id} ({stats['size']} cities): {cities_str}")
        print(f"      Weather: {stats['avg_temp']:.1f}°C, "
              f"{stats['avg_precip']:.1f}mm rain, "
              f"{stats['avg_wind']:.1f} km/h wind")
    
    # Rank clusters by comfort (use cluster stats to determine best)
    cluster_comfort = {}
    for cluster_id, stats in cluster_stats.items():
        # Simple heuristic: lower rain + moderate temp + lower wind = better
        comfort = max(0, 30 - stats['avg_precip']) + max(0, 20 - abs(stats['avg_temp'] - 18)) + max(0, 15 - stats['avg_wind']/3)
        cluster_comfort[cluster_id] = comfort
    
    ranked_clusters = sorted(cluster_comfort.keys(), key=lambda x: cluster_comfort[x], reverse=True)
    
    print(f"\n🏆 Clusters ranked best → worst: {ranked_clusters}")
    
    # Filter to top 2 clusters (good weather)
    good_clusters = ranked_clusters[:2]
    good_cities_mask = today_df['cluster_id'].isin(good_clusters)
    
    print(f"   Keeping cities in clusters {good_clusters} for fine ranking")
    
    # Stage 2: Regression Scoring (fine ranking)
    print("\n🎯 Stage 2: Predicting comfort scores...")
    
    X_reg, y_actual, city_names_reg = get_regression_matrix(today_df)
    y_pred = regression_model.predict(X_reg)
    
    today_df['comfort_score_pred'] = y_pred
    today_df['comfort_score_actual'] = y_actual
    
    # Filter to good clusters and rank by predicted score
    recommendations_df = today_df[good_cities_mask].copy()
    recommendations_df = recommendations_df.sort_values('comfort_score_pred', ascending=False)
    
    # Add ranking
    recommendations_df['rank'] = range(1, len(recommendations_df) + 1)
    
    print(f"\\n🏆 Top 5 recommendations for this weekend:")
    top5 = recommendations_df.head(5)
    
    for i, row in enumerate(top5.itertuples(), 1):
        error = abs(row.comfort_score_pred - row.comfort_score_actual)
        print(f"   {i}. {row.city}")
        print(f"      Cluster: {row.cluster_id}")
        print(f"      Predicted score: {row.comfort_score_pred:.1f}/100")
        print(f"      Actual score: {row.comfort_score_actual:.1f}/100")
        print(f"      Prediction error: {error:.1f} points")
        print()
    
    # Prepare records for database
    recommendation_date = today_df['feature_date'].iloc[0]
    
    records = []
    for row in recommendations_df.itertuples():
        records.append({
            'recommendation_date': recommendation_date,
            'city': row.city,
            'cluster_id': int(row.cluster_id),
            'comfort_score_pred': round(float(row.comfort_score_pred), 2),
            'rank': int(row.rank),
        })
    
    # Write to database
    print(f"💾 Writing {len(records)} recommendations to database...")
    
    bulk_insert(
        table='recommendations',
        rows=records,
        conflict_action='(recommendation_date, city) DO UPDATE SET '
                       'cluster_id = EXCLUDED.cluster_id, '
                       'comfort_score_pred = EXCLUDED.comfort_score_pred, '
                       'rank = EXCLUDED.rank, '
                       'created_at = EXCLUDED.created_at'
    )
    
    print(f"✅ Recommendations saved!")
    
    # Summary
    summary = {
        'recommendation_date': str(recommendation_date),
        'n_recommendations': len(records),
        'top_city': top5.iloc[0]['city'],
        'top_score': float(top5.iloc[0]['comfort_score_pred']),
        'good_clusters': good_clusters,
    }
    
    print("\\n" + "="*70)
    print("✅ RECOMMENDATIONS GENERATED")  
    print("="*70)
    
    return summary


def log_recommendation_stats(summary):
    """
    Log statistics about today's recommendations.
    """
    print("\\n📊 Recommendation Statistics:")
    print(f"   Date: {summary['recommendation_date']}")
    print(f"   Total recommendations: {summary['n_recommendations']}")
    print(f"   Top destination: {summary['top_city']} ({summary['top_score']:.1f}/100)")
    print(f"   Good weather clusters: {summary['good_clusters']}")


if __name__ == "__main__":
    print(f"🚀 Starting recommendation generation at {datetime.now()}")
    
    try:
        # Generate recommendations
        summary = generate_recommendations()
        
        # Log statistics
        log_recommendation_stats(summary)
        
        print(f"\\n🎉 Successfully generated {summary['n_recommendations']} recommendations!")
        print(f"🏆 Best destination: {summary['top_city']} ({summary['top_score']:.1f}/100)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)