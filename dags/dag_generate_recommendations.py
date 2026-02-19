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
    
    Phase 3 update: Generates recommendations for ALL profiles.
    
    Two-stage process (per profile):
    1. K-Means clustering (coarse filter)
    2. Regression scoring (fine ranking)
    3. Join with hotels data (enrichment)
    """
    print("\n" + "="*70)
    print("GENERATING DAILY RECOMMENDATIONS (ALL PROFILES)")
    print("="*70)
    
    # Load models and data
    kmeans_model, regression_model = load_champion_models()
    today_df = load_today_features()
    
    # Load all profiles
    profiles_query = "SELECT profile_name, icon FROM scoring_profiles ORDER BY profile_name"
    profiles = execute_query(profiles_query)
    
    print(f"\n🎯 Generating recommendations for {len(profiles)} profiles...")
    
    all_recommendations = []
    
    for profile in profiles:
        profile_name = profile['profile_name']
        profile_icon = profile['icon']
        
        print(f"\n{'='*70}")
        print(f"{profile_icon} Profile: {profile_name.upper()}")
        print(f"{'='*70}")
        
        # Load profile-specific scores
        profile_scores_query = """
            SELECT city, comfort_score
            FROM profile_scores
            WHERE feature_date = CURRENT_DATE
              AND profile_name = %s
        """
        
        profile_scores_data = execute_query(profile_scores_query, (profile_name,))
        
        if not profile_scores_data:
            print(f"⚠️  No scores found for {profile_name}, skipping...")
            continue
        
        # Merge with today's features
        profile_scores_df = pd.DataFrame(profile_scores_data)
        today_profile_df = today_df.merge(profile_scores_df, on='city', how='left', suffixes=('', '_profile'))
        today_profile_df['comfort_score_actual'] = today_profile_df['comfort_score_profile']
        
        # Stage 1: K-Means Clustering (coarse filter)
        print("\n🔍 Stage 1: Clustering cities...")
        
        X_kmeans, city_names_kmeans = get_kmeans_matrix(today_profile_df)
        cluster_labels = kmeans_model.predict(X_kmeans)
        
        today_profile_df['cluster_id'] = cluster_labels
        
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
            }
        
        # Rank clusters by comfort
        ranked_clusters = kmeans_model.rank_clusters_by_comfort(
            {f'cluster_{i}': stats for i, stats in cluster_stats.items()}
        )
        
        # Filter to top 2 clusters (good weather)
        good_clusters = ranked_clusters[:2]
        good_cities_mask = today_profile_df['cluster_id'].isin(good_clusters)
        
        print(f"   🏆 Best clusters: {good_clusters}")
        
        # Stage 2: Use profile-specific scores for ranking
        print("\n🎯 Stage 2: Ranking by profile scores...")
        
        # Filter to good clusters and rank by profile score
        recommendations_df = today_profile_df[good_cities_mask].copy()
        recommendations_df = recommendations_df.sort_values('comfort_score_actual', ascending=False)
        
        # Add ranking
        recommendations_df['rank'] = range(1, len(recommendations_df) + 1)
        recommendations_df['profile_name'] = profile_name
        
        print(f"\n🏆 Top 5 for {profile_name}:")
        top5 = recommendations_df.head(5)
        
        for i, row in enumerate(top5.itertuples(), 1):
            print(f"   {i}. {row.city}: {row.comfort_score_actual:.1f}/100")
        
        # Prepare records for database
        recommendation_date = today_profile_df['feature_date'].iloc[0]
        
        for row in recommendations_df.itertuples():
            all_recommendations.append({
                'recommendation_date': recommendation_date,
                'city': row.city,
                'profile_name': profile_name,
                'cluster_id': int(row.cluster_id),
                'comfort_score_pred': round(float(row.comfort_score_actual), 2),
                'rank': int(row.rank),
            })
    
    # Write all recommendations to database
    print(f"\n💾 Writing {len(all_recommendations)} recommendations to database...")
    print(f"   ({len(profiles)} profiles × ~{len(all_recommendations)//len(profiles)} cities per profile)")
    
    bulk_insert(
        table='recommendations',
        rows=all_recommendations,
        conflict_action='(recommendation_date, city, profile_name) DO UPDATE SET '
                       'cluster_id = EXCLUDED.cluster_id, '
                       'comfort_score_pred = EXCLUDED.comfort_score_pred, '
                       'rank = EXCLUDED.rank, '
                       'created_at = EXCLUDED.created_at'
    )
    
    print(f"✅ Recommendations saved!")
    
    # Show hotels for top city of leisure profile
    print("\n🏨 Sample: Hotels for top 'leisure' destination...")
    
    leisure_recs = [r for r in all_recommendations if r['profile_name'] == 'leisure']
    if leisure_recs:
        top_leisure_city = leisure_recs[0]['city']
        
        hotels_query = """
            SELECT hotel_name, stars
            FROM hotels
            WHERE city = %s
            ORDER BY stars DESC NULLS LAST
            LIMIT 3
        """
        
        hotels = execute_query(hotels_query, (top_leisure_city,))
        
        if hotels:
            print(f"   {top_leisure_city}:")
            for hotel in hotels:
                stars_display = "⭐" * (hotel['stars'] or 0) if hotel['stars'] else ""
                print(f"      • {hotel['hotel_name']} {stars_display}")
        else:
            print(f"   No hotels data yet for {top_leisure_city}")
    
    # Push summary to XCom
    summary = {
        'recommendation_date': str(recommendation_date),
        'n_recommendations': len(all_recommendations),
        'n_profiles': len(profiles),
        'profiles': [p['profile_name'] for p in profiles],
    }
    
    context['task_instance'].xcom_push(key='recommendation_summary', value=summary)
    
    print("\n" + "="*70)
    print("✅ MULTI-PROFILE RECOMMENDATIONS GENERATED")
    print("="*70)
    
    return summary



def log_recommendation_stats(**context):
    """
    Log statistics about today's multi-profile recommendations.
    (Phase 3 update: handles multiple profiles)
    """
    summary = context['task_instance'].xcom_pull(
        task_ids='generate_recommendations',
        key='recommendation_summary'
    )
    
    print("\n📊 Multi-Profile Recommendation Statistics:")
    print(f"   Date: {summary['recommendation_date']}")
    print(f"   Total recommendations: {summary['n_recommendations']}")
    print(f"   Profiles processed: {summary['n_profiles']}")
    print(f"   Profile types: {', '.join(summary['profiles'])}")
    
    # Get detailed stats per profile
    print("\n🎯 Top destinations by profile:")
    
    from src.data.db import execute_query
    
    stats_query = """
        SELECT profile_name, city, comfort_score_pred, rank
        FROM recommendations 
        WHERE recommendation_date = %s
          AND rank = 1
        ORDER BY profile_name
    """
    
    top_per_profile = execute_query(stats_query, (summary['recommendation_date'],))
    
    # Map profile names to emojis for display
    profile_icons = {
        'leisure': '🏖️', 'surfer': '🏄', 'cyclist': '🚴', 
        'stargazer': '⭐', 'skier': '⛷️'
    }
    
    for profile_rec in top_per_profile:
        profile = profile_rec['profile_name']
        city = profile_rec['city']
        score = profile_rec['comfort_score_pred']
        icon = profile_icons.get(profile, '🎯')
        print(f"      {icon} {profile:12s}: {city} ({score:.1f}/100)")
    
    # Overall statistics
    avg_score = summary['n_recommendations'] / summary['n_profiles'] if summary['n_profiles'] > 0 else 0
    print(f"\n📈 Avg recommendations per profile: {avg_score:.1f}")
    
    # Could add more analytics here:
    # - Compare to yesterday's recommendations per profile
    # - Track if top cities are changing week-to-week
    # - Alert if all scores are below threshold for any profile


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
