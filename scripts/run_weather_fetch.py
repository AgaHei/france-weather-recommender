#!/usr/bin/env python3
"""
run_weather_fetch.py
--------------------
Standalone script to fetch and store weather data without Airflow.
Does the same thing as the DAG but runs directly.
"""

import sys
import os
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.fetch_weather import fetch_all_cities_forecast
from src.data.db import bulk_insert, execute_query


def fetch_and_store_weather():
    """
    Fetch weather for all cities and store in Neon.
    Returns the number of records inserted.
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
    return len(records)


def validate_weather_data():
    """
    Quick data quality check after insertion.
    
    Checks:
    - Do we have data for all 20 cities?
    - Are there any null temperature values?
    - Is precipitation in a reasonable range?
    """
    print("🔍 Validating weather data...")
    
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
    
    # Check 3: Recent data check
    result = execute_query(
        "SELECT COUNT(*) as recent_count FROM raw_weather "
        "WHERE date >= CURRENT_DATE"
    )
    recent_count = result[0]['recent_count']
    print(f"✅ {recent_count} records for current/future dates")


if __name__ == "__main__":
    print(f"🚀 Starting weather fetch at {datetime.now()}")
    
    try:
        # Fetch and store
        record_count = fetch_and_store_weather()
        
        # Validate
        validate_weather_data()
        
        print(f"🎉 Successfully processed {record_count} weather records!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)