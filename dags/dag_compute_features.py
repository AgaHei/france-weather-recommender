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
    
    # Compute features
    print("🔧 Computing rolling window features...")
    features_df = compute_rolling_features(df, as_of_date=datetime.now().date())
    
    print(f"✅ Computed features for {len(features_df)} cities")
    print(f"\nSample features:")
    print(features_df.head(3).to_string(index=False))
    
    # Convert to list of dicts for database insertion
    feature_records = features_df.to_dict('records')
    
    # Insert into Neon
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
    
    print(f"✅ Inserted/updated {len(feature_records)} rows in weather_features table")
    
    # Push feature summary to XCom
    summary = {
        'feature_count': len(feature_records),
        'avg_comfort_score': float(features_df['comfort_score'].mean()),
        'max_comfort_score': float(features_df['comfort_score'].max()),
        'best_city': features_df.loc[features_df['comfort_score'].idxmax(), 'city'],
    }
    
    context['task_instance'].xcom_push(key='feature_summary', value=summary)
    
    print(f"\n📈 Feature summary:")
    print(f"   Average comfort score: {summary['avg_comfort_score']:.1f}/100")
    print(f"   Best city today: {summary['best_city']} ({summary['max_comfort_score']:.1f}/100)")
    
    return summary


def log_feature_stats(**context):
    """
    Optional: Log feature distribution for monitoring drift.
    In a real system, you'd compare this to historical distributions.
    """
    from src.data.db import execute_query
    
    query = """
        SELECT 
            AVG(temp_mean_3d) as avg_temp,
            AVG(precip_sum_3d) as avg_rain,
            AVG(wind_max_3d) as avg_wind,
            AVG(comfort_score) as avg_comfort,
            STDDEV(comfort_score) as std_comfort
        FROM weather_features
        WHERE feature_date = CURRENT_DATE
    """
    
    stats = execute_query(query)[0]
    
    print("\n📊 Feature statistics (today):")
    print(f"   Avg temperature (3d):   {stats['avg_temp']:.1f}°C")
    print(f"   Avg precipitation (3d): {stats['avg_rain']:.1f}mm")
    print(f"   Avg wind (3d):          {stats['avg_wind']:.1f} km/h")
    print(f"   Avg comfort score:      {stats['avg_comfort']:.1f} ± {stats['std_comfort']:.1f}")
    
    # In Phase 3, you'd add drift detection here:
    # - Compare today's avg_comfort to last week's
    # - If delta > threshold, trigger retraining


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
