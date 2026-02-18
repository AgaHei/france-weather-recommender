"""
dag_generate_recommendations.py
--------------------------------
DAG 4: Generate daily weekend destination recommendations.

Schedule: Daily at 7:00 AM (after weather data is fetched and features computed)
Duration: ~5 seconds

What it does:
1. Loads champion K-Means and Regression models
2. Runs clustering on today's weather features
3. Identifies "good weather" clusters
4. Runs regression prediction on cities in good clusters
5. Ranks cities by predicted comfort score
6. Writes top recommendations to database
7. Optionally triggers email/notification

MLOps patterns demonstrated:
- Model serving (loading champion from registry)
- Batch inference (scoring all cities)
- Two-stage retrieval (coarse filter → fine ranking)
- Production logging (write results to database)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.db import execute_query, bulk_insert
from src.features.engineer import get_kmeans_matrix, get_regression_matrix
from src.models.clustering import WeatherClusterModel
from src.models.regression import ComfortScoreModel

# Model directory
MODELS_DIR = '/opt/airflow/mlflow/models'

# ---------------------------------------------------------------------------
# DAG configuration
# ---------------------------------------------------------------------------

default_args = {
    'owner': 'aga',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 18),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'generate_recommendations',
    default_args=default_args,
    description='Daily generation of weekend destination recommendations',
    schedule_interval='0 7 * * *',  # 7:00 AM every day
    catchup=False,
    tags=['recommendations', 'ml-serving'],
)

# ---------------------------------------------------------------------------
# Serving functions
# ---------------------------------------------------------------------------

def load_champion_models() -> tuple:
    """
    Load the current champion models from disk.
    
    Returns:
        (kmeans_model, regression_model)
    """
    kmeans_path = os.path.join(MODELS_DIR, 'kmeans_champion.joblib')
    regression_path = os.path.join(MODELS_DIR, 'regression_champion.joblib')
    
    # Check if champions exist
    if not os.path.exists(kmeans_path):
        raise FileNotFoundError(
            f"No champion K-Means model found at {kmeans_path}. "
            "Run dag_retrain_models first."
        )
    
    if not os.path.exists(regression_path):
        raise FileNotFoundError(
            f"No champion regression model found at {regression_path}. "
            "Run dag_retrain_models first."
        )
    
    print("📂 Loading champion models...")
    kmeans_model = WeatherClusterModel.load(kmeans_path)
    regression_model = ComfortScoreModel.load(regression_path)
    
    return kmeans_model, regression_model


def load_today_features() -> pd.DataFrame:
    """
    Load today's weather features from Neon.
    
    Returns:
        DataFrame with today's features for all cities
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
            "No features found for today. "
            "Run dag_compute_features first."
        )
    
    df = pd.DataFrame(data)
    print(f"📊 Loaded features for {len(df)} cities on {df['feature_date'].iloc[0]}")
    
    return df


def generate_recommendations(**context):
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
    
    # Rank clusters by comfort
    ranked_clusters = kmeans_model.rank_clusters_by_comfort(
        {f'cluster_{i}': stats for i, stats in cluster_stats.items()}
    )
    
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
    
    # Filter to good clusters and rank
    recommendations_df = today_df[good_cities_mask].copy()
    recommendations_df = recommendations_df.sort_values('comfort_score_pred', ascending=False)
    
    # Add ranking
    recommendations_df['rank'] = range(1, len(recommendations_df) + 1)
    
    print(f"\n🏆 Top 5 recommendations for this weekend:")
    top5 = recommendations_df.head(5)
    
    for i, row in enumerate(top5.itertuples(), 1):
        print(f"   {i}. {row.city}")
        print(f"      Cluster: {row.cluster_id}")
        print(f"      Predicted score: {row.comfort_score_pred:.1f}/100")
        print(f"      Actual score: {row.comfort_score_actual:.1f}/100")
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
    print(f"\n💾 Writing {len(records)} recommendations to database...")
    
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
    
    # Push summary to XCom
    summary = {
        'recommendation_date': str(recommendation_date),
        'n_recommendations': len(records),
        'top_city': top5.iloc[0]['city'],
        'top_score': float(top5.iloc[0]['comfort_score_pred']),
        'good_clusters': good_clusters,
    }
    
    context['task_instance'].xcom_push(key='recommendation_summary', value=summary)
    
    print("\n" + "="*70)
    print("✅ RECOMMENDATIONS GENERATED")
    print("="*70)
    
    return summary


def log_recommendation_stats(**context):
    """
    Optional: Log statistics about today's recommendations.
    """
    summary = context['task_instance'].xcom_pull(
        task_ids='generate_recommendations',
        key='recommendation_summary'
    )
    
    print("\n📊 Recommendation Statistics:")
    print(f"   Date: {summary['recommendation_date']}")
    print(f"   Total recommendations: {summary['n_recommendations']}")
    print(f"   Top destination: {summary['top_city']} ({summary['top_score']:.1f}/100)")
    print(f"   Good weather clusters: {summary['good_clusters']}")
    
    # Could add more analytics here:
    # - Compare to yesterday's recommendations
    # - Track if top city is changing week-to-week
    # - Alert if all scores are below threshold (no good destinations!)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

task_generate = PythonOperator(
    task_id='generate_recommendations',
    python_callable=generate_recommendations,
    dag=dag,
)

task_log_stats = PythonOperator(
    task_id='log_stats',
    python_callable=log_recommendation_stats,
    dag=dag,
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

task_generate >> task_log_stats
