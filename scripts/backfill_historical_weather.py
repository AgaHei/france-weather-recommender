"""
backfill_historical_weather.py
-------------------------------
ONE-TIME SCRIPT: Backfill 90 days of historical weather data.

Run this ONCE before starting Airflow to populate your database
with training data for the ML models.

Usage:
    python scripts/backfill_historical_weather.py

This fetches historical weather from Open-Meteo Archive API (free)
for all 20 cities, then computes features for the entire period.
"""

import sys
import os
from datetime import date, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.fetch_weather import fetch_all_cities_historical
from src.data.db import bulk_insert, execute_query
from src.features.engineer import compute_rolling_features


def backfill_weather(days_back: int = 90):
    """
    Backfill historical weather data.
    
    Args:
        days_back: How many days of history to fetch (default 90)
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"🌍 Backfilling weather data from {start_date} to {end_date}")
    print(f"   ({days_back} days × 20 cities = {days_back * 20} records expected)\n")
    
    # Fetch historical data
    print("📡 Fetching from Open-Meteo Archive API...")
    records = fetch_all_cities_historical(start_date, end_date)
    
    if not records:
        print("❌ No data fetched. Check your internet connection or API status.")
        return
    
    print(f"\n✅ Fetched {len(records)} records")
    
    # Insert into Neon
    print("💾 Inserting into raw_weather table...")
    bulk_insert(
        table='raw_weather',
        rows=records,
        conflict_action='(city, date) DO NOTHING'  # Skip duplicates
    )
    
    print(f"✅ Inserted into database\n")
    
    # Now compute features for all dates
    print("🔧 Computing features for entire historical period...")
    
    # Read all data back
    query = "SELECT city, date, temp_mean, precipitation, wind_speed_max FROM raw_weather ORDER BY city, date"
    raw_data = execute_query(query)
    df = pd.DataFrame(raw_data)
    
    # Compute features for each date (we'll do this in batches to avoid memory issues)
    all_features = []
    
    dates = sorted(df['date'].unique())
    print(f"   Processing {len(dates)} dates...")
    
    for i, target_date in enumerate(dates):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(dates)}")
        
        # Get data up to this date
        df_subset = df[df['date'] <= target_date]
        
        # Compute features as of this date
        features = compute_rolling_features(df_subset, as_of_date=target_date)
        all_features.append(features)
    
    # Combine all features
    features_df = pd.concat(all_features, ignore_index=True)
    feature_records = features_df.to_dict('records')
    
    print(f"\n✅ Computed {len(feature_records)} feature records")
    
    # Insert features
    print("💾 Inserting into weather_features table...")
    bulk_insert(
        table='weather_features',
        rows=feature_records,
        conflict_action='(city, feature_date) DO NOTHING'
    )
    
    print(f"✅ Historical backfill complete!\n")
    
    # Summary statistics
    print("📊 Summary:")
    print(f"   Total raw weather records: {len(records)}")
    print(f"   Total feature records: {len(feature_records)}")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Cities: {len(df['city'].unique())}")
    
    # Show best/worst cities on average
    avg_scores = features_df.groupby('city')['comfort_score'].mean().sort_values(ascending=False)
    print(f"\n🏆 Top 3 cities (avg comfort score):")
    for city, score in avg_scores.head(3).items():
        print(f"   {city}: {score:.1f}/100")
    
    print(f"\n❄️  Bottom 3 cities:")
    for city, score in avg_scores.tail(3).items():
        print(f"   {city}: {score:.1f}/100")


if __name__ == "__main__":
    print("="*70)
    print("Historical Weather Data Backfill")
    print("="*70 + "\n")
    
    backfill_weather(days_back=90)
    
    print("\n" + "="*70)
    print("✅ Backfill complete! You can now start Airflow.")
    print("="*70)
