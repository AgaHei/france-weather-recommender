#!/usr/bin/env python3
"""
run_feature_engineering.py
---------------------------
Standalone script to compute features from raw weather data without Airflow.
Does the same thing as dag_compute_features but runs directly.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.db import execute_query, bulk_insert
from src.features.engineer import compute_rolling_features


def compute_and_store_features():
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
        raise ValueError("No raw weather data found! Run fetch weather first.")
    
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
    
    # Feature summary
    summary = {
        'feature_count': len(feature_records),
        'avg_comfort_score': float(features_df['comfort_score'].mean()),
        'max_comfort_score': float(features_df['comfort_score'].max()),
        'best_city': features_df.loc[features_df['comfort_score'].idxmax(), 'city'],
    }
    
    print(f"\n📈 Feature summary:")
    print(f"   Average comfort score: {summary['avg_comfort_score']:.1f}/100")
    print(f"   Best city today: {summary['best_city']} ({summary['max_comfort_score']:.1f}/100)")
    
    return summary


def log_feature_stats():
    """
    Log feature distribution for monitoring drift.
    In a real system, you'd compare this to historical distributions.
    """
    print("\n🔍 Computing feature statistics...")
    
    query = """
        SELECT 
            AVG(temp_mean_3d) as avg_temp,
            AVG(precip_sum_3d) as avg_rain,
            AVG(wind_max_3d) as avg_wind,
            AVG(comfort_score) as avg_comfort,
            STDDEV(comfort_score) as std_comfort,
            COUNT(*) as city_count
        FROM weather_features
        WHERE feature_date = CURRENT_DATE
    """
    
    stats = execute_query(query)[0]
    
    print("\n📊 Feature statistics (today):")
    print(f"   Cities with features:   {stats['city_count']}")
    print(f"   Avg temperature (3d):   {stats['avg_temp']:.1f}°C")
    print(f"   Avg precipitation (3d): {stats['avg_rain']:.1f}mm")
    print(f"   Avg wind (3d):          {stats['avg_wind']:.1f} km/h")
    print(f"   Avg comfort score:      {stats['avg_comfort']:.1f} ± {stats['std_comfort']:.1f}")
    
    # Show top cities by comfort score
    top_cities_query = """
        SELECT city, comfort_score
        FROM weather_features
        WHERE feature_date = CURRENT_DATE
        ORDER BY comfort_score DESC
        LIMIT 5
    """
    
    top_cities = execute_query(top_cities_query)
    print("\n🏆 Top 5 cities today:")
    for i, city_data in enumerate(top_cities, 1):
        print(f"   {i}. {city_data['city']}: {city_data['comfort_score']:.1f}/100")


if __name__ == "__main__":
    print(f"🚀 Starting feature engineering at {datetime.now()}")
    
    try:
        # Compute and store features
        summary = compute_and_store_features()
        
        # Log statistics
        log_feature_stats()
        
        print(f"\n🎉 Successfully processed features for {summary['feature_count']} cities!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)