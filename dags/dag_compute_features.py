"""
dag_compute_features.py
-----------------------
DAG 2: Compute engineered features from raw weather data.

Triggered by: dag_fetch_weather (after new data arrives)
Duration: ~5 seconds

What it does:
1. Reads raw_weather table from Neon
2. Computes rolling windows (7-day, 3-day averages/sums)
3. Computes comfort_score labels
4. Writes to weather_features table

MLOps patterns demonstrated:
- Feature engineering as a separate DAG (modularity)
- Reproducible feature computation (same inputs → same outputs)
- Feature versioning (feature_date tracks "as of when")
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.db import execute_query, bulk_insert
from src.features.engineer import compute_rolling_features

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
    'compute_features',
    default_args=default_args,
    description='Compute rolling window features and comfort scores',
    schedule_interval=None,  # Triggered by dag_fetch_weather
    catchup=False,
    tags=['features', 'ml-pipeline'],
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def compute_and_store_features(**context):
    """
    Read raw weather, compute features, store in weather_features table.
    
    Phase 3 update: Now computes comfort scores for ALL profiles.
    """
    print("📊 Reading raw weather data from Neon...")
    
    # Get last 30 days of data (we need history for rolling windows)
    query = """
        SELECT city, date, temp_mean, precipitation, wind_speed_max
        FROM raw_weather
        WHERE date >= CURRENT_DATE - INTERVAL '30 days'
        ORDER BY city, date
    """
    
    raw_data = execute_query(query)
    
    if not raw_data:
        raise ValueError("No raw weather data found! Run dag_fetch_weather first.")
    
    print(f"✅ Loaded {len(raw_data)} raw weather records")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(raw_data)
    
    # Compute base features (rolling windows)
    print("🔧 Computing rolling window features...")
    features_df = compute_rolling_features(df, as_of_date=datetime.now().date())
    
    print(f"✅ Computed features for {len(features_df)} cities")
    
    # Phase 3: Load all profiles and compute scores for each
    print("\n🎯 Computing comfort scores for all profiles...")
    
    from src.features.engineer import compute_all_profile_scores
    
    profiles_query = "SELECT * FROM scoring_profiles ORDER BY profile_name"
    profiles_data = execute_query(profiles_query)
    profiles_df = pd.DataFrame(profiles_data)
    
    print(f"   Loaded {len(profiles_df)} profiles: {', '.join(profiles_df['profile_name'].tolist())}")
    
    # Compute scores for each city
    profile_scores_records = []
    
    for idx, row in features_df.iterrows():
        # Compute all profile scores at once
        scores = compute_all_profile_scores(
            temp_mean=row['temp_mean_3d'],
            precipitation=row['precip_sum_3d'],
            wind_speed_max=row['wind_max_3d'],
            profiles_df=profiles_df
        )
        
        # Store each profile score as a separate record
        for profile_name, score in scores.items():
            profile_scores_records.append({
                'city': row['city'],
                'feature_date': row['feature_date'],
                'profile_name': profile_name,
                'comfort_score': score
            })
    
    print(f"✅ Computed {len(profile_scores_records)} profile scores ({len(features_df)} cities × {len(profiles_df)} profiles)")
    
    # Insert into profile_scores table
    print("\n💾 Storing profile scores in database...")
    bulk_insert(
        table='profile_scores',
        rows=profile_scores_records,
        conflict_action='(city, feature_date, profile_name) DO UPDATE SET '
                       'comfort_score = EXCLUDED.comfort_score, '
                       'computed_at = EXCLUDED.computed_at'
    )
    
    print(f"✅ Profile scores saved!")
    
    # Also update weather_features with default 'leisure' score for backwards compatibility
    print("\n💾 Updating weather_features table...")
    
    leisure_scores = {
        row['city']: row['comfort_score'] 
        for row in profile_scores_records 
        if row['profile_name'] == 'leisure'
    }
    
    for idx, row in features_df.iterrows():
        features_df.at[idx, 'comfort_score'] = leisure_scores.get(row['city'], 0)
    
    feature_records = features_df.to_dict('records')
    
    bulk_insert(
        table='weather_features',
        rows=feature_records,
        conflict_action='(city, feature_date) DO UPDATE SET '
                       'temp_mean_7d = EXCLUDED.temp_mean_7d, '
                       'temp_mean_3d = EXCLUDED.temp_mean_3d, '
                       'precip_sum_7d = EXCLUDED.precip_sum_7d, '
                       'precip_sum_3d = EXCLUDED.precip_sum_3d, '
                       'wind_max_7d = EXCLUDED.wind_max_7d, '
                       'wind_max_3d = EXCLUDED.wind_max_3d, '
                       'comfort_score = EXCLUDED.comfort_score, '
                       'computed_at = EXCLUDED.computed_at'
    )
    
    print(f"✅ Weather features updated!")
    
    # Show sample scores
    print("\n📊 Sample profile scores for top cities:")
    sample_city = features_df.iloc[0]['city']
    sample_scores = {r['profile_name']: r['comfort_score'] 
                     for r in profile_scores_records 
                     if r['city'] == sample_city}
    
    print(f"\n   {sample_city}:")
    for profile, score in sorted(sample_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"      {profile:12s}: {score:5.1f}/100")
    
    # Push summary to XCom
    summary = {
        'feature_count': len(feature_records),
        'profile_scores_count': len(profile_scores_records),
        'profiles': profiles_df['profile_name'].tolist(),
        'sample_scores': sample_scores
    }
    
    context['task_instance'].xcom_push(key='feature_summary', value=summary)
    
    return summary


def log_feature_stats(**context):
    """
    Log feature statistics and profile score distributions.
    """
    summary = context['task_instance'].xcom_pull(
        task_ids='compute_features',
        key='feature_summary'
    )
    
    print("\n📊 Feature computation summary:")
    print(f"   Total features computed: {summary['feature_count']}")
    print(f"   Profile scores computed: {summary['profile_scores_count']}")
    print(f"   Profiles: {', '.join(summary['profiles'])}")
    
    print(f"\n📈 Sample profile scores:")
    for profile, score in sorted(summary['sample_scores'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {profile:12s}: {score:5.1f}/100")
    
    # Query aggregate stats per profile
    from src.data.db import execute_query
    
    query = """
        SELECT 
            profile_name,
            AVG(comfort_score) as avg_score,
            MAX(comfort_score) as max_score,
            MIN(comfort_score) as min_score
        FROM profile_scores
        WHERE feature_date = CURRENT_DATE
        GROUP BY profile_name
        ORDER BY avg_score DESC
    """
    
    stats = execute_query(query)
    
    if stats:
        print(f"\n📊 Profile score distributions (today):")
        for row in stats:
            print(f"   {row['profile_name']:12s}: avg={row['avg_score']:.1f}, "
                  f"min={row['min_score']:.1f}, max={row['max_score']:.1f}")
    
    print("\n✅ Feature statistics logged")


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

task_compute = PythonOperator(
    task_id='compute_features',
    python_callable=compute_and_store_features,
    dag=dag,
)

task_stats = PythonOperator(
    task_id='log_statistics',
    python_callable=log_feature_stats,
    dag=dag,
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

task_compute >> task_stats
