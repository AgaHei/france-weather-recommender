"""
dag_fetch_weather.py
--------------------
DAG 1: Fetch daily weather forecasts for all 20 French cities.

Schedule: Daily at 6:00 AM Paris time
Duration: ~30 seconds (20 cities × API calls)

What it does:
1. Calls Open-Meteo API for each city (7-day forecast)
2. Inserts/updates raw_weather table in Neon
3. Triggers the feature engineering DAG

MLOps patterns demonstrated:
- Idempotent writes (ON CONFLICT DO UPDATE)
- Error handling per city (one failure doesn't kill the whole DAG)
- Data validation before insertion
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.fetch_weather import fetch_all_cities_forecast
from src.data.db import bulk_insert

# ---------------------------------------------------------------------------
# DAG configuration
# ---------------------------------------------------------------------------

default_args = {
    'owner': 'aga',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 18),  # Today
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fetch_weather',
    default_args=default_args,
    description='Fetch daily weather forecasts for 20 French cities',
    schedule_interval='0 6 * * *',  # 6:00 AM Paris time every day
    catchup=False,  # Don't backfill missed runs
    tags=['weather', 'data-ingestion'],
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def fetch_and_store_weather(**context):
    """
    Fetch weather for all cities and store in Neon.
    
    Returns the number of records inserted (for monitoring/logging).
    """
    print("🌤️  Fetching weather forecasts for all 20 cities...")
    
    # Fetch from Open-Meteo (returns list of dicts)
    records = fetch_all_cities_forecast(days=7)
    
    if not records:
        raise ValueError("No weather data fetched! API might be down.")
    
    print(f"✅ Fetched {len(records)} records ({len(records)//7} cities × 7 days)")
    
    # Insert into Neon (idempotent: ON CONFLICT on city+date)
    bulk_insert(
        table='raw_weather',
        rows=records,
        conflict_action='(city, date) DO UPDATE SET '
                       'temp_max = EXCLUDED.temp_max, '
                       'temp_min = EXCLUDED.temp_min, '
                       'temp_mean = EXCLUDED.temp_mean, '
                       'precipitation = EXCLUDED.precipitation, '
                       'wind_speed_max = EXCLUDED.wind_speed_max, '
                       'wind_gusts_max = EXCLUDED.wind_gusts_max, '
                       'weather_code = EXCLUDED.weather_code, '
                       'fetched_at = EXCLUDED.fetched_at'
    )
    
    print(f"✅ Inserted/updated {len(records)} rows in raw_weather table")
    
    # Push to XCom for downstream monitoring
    context['task_instance'].xcom_push(key='records_count', value=len(records))
    
    return len(records)


def validate_weather_data(**context):
    """
    Optional: Quick data quality check after insertion.
    
    Checks:
    - Do we have data for all 20 cities?
    - Are there any null temperature values?
    - Is precipitation in a reasonable range?
    """
    from src.data.db import execute_query
    
    # Check 1: City count
    result = execute_query("SELECT COUNT(DISTINCT city) as city_count FROM raw_weather")
    city_count = result[0]['city_count']
    
    if city_count < 20:
        print(f"⚠️  WARNING: Only {city_count}/20 cities have data")
    else:
        print(f"✅ All 20 cities present in database")
    
    # Check 2: Null temperatures
    result = execute_query(
        "SELECT COUNT(*) as null_count FROM raw_weather "
        "WHERE temp_mean IS NULL AND date >= CURRENT_DATE"
    )
    null_count = result[0]['null_count']
    
    if null_count > 0:
        print(f"⚠️  WARNING: {null_count} records with null temperatures")
    else:
        print(f"✅ No null temperatures in recent data")
    
    # Check 3: Crazy outliers
    result = execute_query(
        "SELECT city, date, temp_mean, precipitation FROM raw_weather "
        "WHERE date >= CURRENT_DATE "
        "AND (temp_mean < -20 OR temp_mean > 50 OR precipitation > 200)"
    )
    
    if result:
        print(f"⚠️  WARNING: Found {len(result)} outlier records:")
        for row in result:
            print(f"    {row['city']} on {row['date']}: "
                  f"{row['temp_mean']}°C, {row['precipitation']}mm")
    else:
        print(f"✅ No extreme outliers detected")
    
    print("\n📊 Data quality checks complete")


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

task_fetch = PythonOperator(
    task_id='fetch_weather_data',
    python_callable=fetch_and_store_weather,
    dag=dag,
)

task_validate = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_weather_data,
    dag=dag,
)

task_trigger_features = TriggerDagRunOperator(
    task_id='trigger_feature_engineering',
    trigger_dag_id='compute_features',  # DAG 2 (we'll build this next)
    wait_for_completion=False,  # Don't block, let it run async
    dag=dag,
)

# ---------------------------------------------------------------------------
# Task dependencies
# ---------------------------------------------------------------------------

task_fetch >> task_validate >> task_trigger_features
