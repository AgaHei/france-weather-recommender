"""
dag_fetch_hotels.py
-------------------
DAG 5: Fetch hotel data for top recommended cities.

Schedule: Twice weekly (Monday & Thursday at 1:00 AM)
Duration: ~45 seconds (5 cities × API calls with delays)

What it does:
1. Gets top 5 cities from latest recommendations (expanded from 3)
2. Checks which cities need fresh hotel data (older than 2 weeks)
3. Fetches hotels from Overpass API (OpenStreetMap) for cities needing updates
4. Filters to top 10 hotels per city
5. Inserts into hotels table with conflict resolution

MLOps patterns demonstrated:
- Data enrichment (recommendations → hotels)
- External API integration with rate limiting & error handling
- Intelligent refresh strategy (only fetch when needed)
- Idempotent writes (ON CONFLICT DO UPDATE)
- Dependency on upstream DAG (needs recommendations first)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.fetch_hotels import (
    get_top_cities_from_recommendations,
    fetch_hotels_for_cities
)
from src.data.db import bulk_insert, execute_query

# ---------------------------------------------------------------------------
# DAG configuration
# ---------------------------------------------------------------------------

default_args = {
    'owner': 'aga',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 18),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fetch_hotels',
    default_args=default_args,
    description='Twice-weekly hotel data fetching for top recommended cities',
    schedule_interval='0 1 * * 1,4',  # Monday & Thursday at 1:00 AM
    catchup=False,
    tags=['hotels', 'data-enrichment'],
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def identify_target_cities(**context):
    """
    Identify which cities need hotel data (top 5 from latest recommendations + any missing).
    """
    print("\n🎯 Identifying target cities for hotel fetching...")
    
    # Get top 5 cities from latest recommendations (expanded from 3)
    top_cities = get_top_cities_from_recommendations(n=5)
    
    if not top_cities:
        raise ValueError(
            "No recommendations found! Run dag_generate_recommendations first."
        )
    
    # Check which top cities already have recent hotel data (within 2 weeks)
    from src.data.db import execute_query
    
    recent_hotels = execute_query("""
        SELECT DISTINCT city 
        FROM hotels 
        WHERE city = ANY(%s) 
          AND fetched_at >= NOW() - INTERVAL '14 days'
    """, (top_cities,))
    
    cities_with_recent_hotels = {h['city'] for h in recent_hotels}
    
    # Target cities that need fresh hotel data
    target_cities = [city for city in top_cities if city not in cities_with_recent_hotels]
    
    if target_cities:
        print(f"\n✅ Target cities (need hotel data): {', '.join(target_cities)}")
    else:
        print(f"\n✅ All top cities have recent hotel data")
        target_cities = top_cities[:2]  # Refresh top 2 anyway
        print(f"   Refreshing top 2 cities: {', '.join(target_cities)}")
    
    # Push to XCom for next task
    context['task_instance'].xcom_push(key='target_cities', value=target_cities)
    
    return target_cities


def fetch_and_store_hotels(**context):
    """
    Fetch hotels from Overpass API and store in database.
    """
    print("\n" + "="*70)
    print("FETCHING HOTEL DATA")
    print("="*70)
    
    # Get target cities from previous task
    target_cities = context['task_instance'].xcom_pull(
        task_ids='identify_cities',
        key='target_cities'
    )
    
    if not target_cities:
        raise ValueError("No target cities received from previous task")
    
    # Fetch hotels
    hotels = fetch_hotels_for_cities(
        cities=target_cities,
        radius_meters=8000,  # 8km radius (smaller for dense cities)
        max_results_per_city=10,  # Top 10 per city
        delay_seconds=3.0  # Extra respectful rate limiting
    )
    
    if not hotels:
        print("\n⚠️  No hotels found for target cities")
        return
    
    print(f"\n📊 Hotel breakdown:")
    for city in target_cities:
        city_hotels = [h for h in hotels if h['city'] == city]
        print(f"   {city}: {len(city_hotels)} hotels")
        
        # Show top 3 for each city
        for h in city_hotels[:3]:
            stars_display = f"({h['stars']}⭐)" if h['stars'] else "(unrated)"
            url_display = f" 🔗" if h.get('website_url') else ""
            print(f"      • {h['hotel_name']} {stars_display}{url_display}")
    
    # Deduplicate hotels before database insertion
    print(f"\n🔧 Deduplicating hotels...")
    seen = set()
    deduplicated_hotels = []
    for hotel in hotels:
        key = (hotel['city'], hotel['hotel_name'])
        if key not in seen:
            seen.add(key)
            deduplicated_hotels.append(hotel)
    
    removed = len(hotels) - len(deduplicated_hotels)
    if removed > 0:
        print(f"   Removed {removed} duplicate hotels")
    
    hotels = deduplicated_hotels
    
    # Insert into database
    print(f"\n💾 Inserting {len(hotels)} unique hotels into database...")
    
    bulk_insert(
        table='hotels',
        rows=hotels,
        conflict_action='(city, hotel_name) DO UPDATE SET '
                       'hotel_type = EXCLUDED.hotel_type, '
                       'stars = EXCLUDED.stars, '
                       'address = EXCLUDED.address, '
                       'latitude = EXCLUDED.latitude, '
                       'longitude = EXCLUDED.longitude, '
                       'website_url = EXCLUDED.website_url, '
                       'amenities = EXCLUDED.amenities, '
                       'data_source = EXCLUDED.data_source, '
                       'fetched_at = EXCLUDED.fetched_at'
    )
    
    print(f"✅ Hotels saved to database!")
    
    # Push summary to XCom
    summary = {
        'total_hotels': len(hotels),
        'cities': target_cities,
        'hotels_per_city': {
            city: len([h for h in hotels if h['city'] == city])
            for city in target_cities
        }
    }
    
    context['task_instance'].xcom_push(key='hotels_summary', value=summary)
    
    print("\n" + "="*70)
    print("✅ HOTEL FETCHING COMPLETE")
    print("="*70)
    
    return summary


def validate_hotel_data(**context):
    """
    Validate hotel data quality.
    """
    summary = context['task_instance'].xcom_pull(
        task_ids='fetch_hotels',
        key='hotels_summary'
    )
    
    print("\n📊 Hotel Data Quality Check:")
    
    # Check 1: Do we have hotels for all target cities?
    for city, count in summary['hotels_per_city'].items():
        if count == 0:
            print(f"   ⚠️  WARNING: No hotels found for {city}")
        if count < 5:
            print(f"   ⚠️  WARNING: Only {count} hotels for {city} (expected 8-10)")
        else:
            print(f"   ✅ {city}: {count} hotels")
    
    # Check 2: How many hotels have star ratings and websites?
    query = """
        SELECT 
            COUNT(*) as total,
            COUNT(stars) as with_stars,
            COUNT(website_url) as with_websites,
            ROUND(100.0 * COUNT(stars) / COUNT(*), 1) as stars_percentage,
            ROUND(100.0 * COUNT(website_url) / COUNT(*), 1) as website_percentage
        FROM hotels
        WHERE fetched_at >= NOW() - INTERVAL '1 day'
    """
    
    result = execute_query(query)[0]
    
    print(f"\n   Total hotels: {result['total']}")
    print(f"   With star ratings: {result['with_stars']} ({result['stars_percentage']}%)")
    print(f"   With websites: {result['with_websites']} ({result['website_percentage']}%)")
    
    if result['stars_percentage'] < 20:
        print(f"   ⚠️  Low star rating coverage (expected ~30-50%)")
        
    if result['website_percentage'] < 10:
        print(f"   ⚠️  Low website coverage (expected ~20-40%)")
    
    # Check 3: How many have amenities?
    query = """
        SELECT 
            COUNT(*) as total,
            COUNT(amenities) as with_amenities,
            ROUND(100.0 * COUNT(amenities) / COUNT(*), 1) as amenities_percentage
        FROM hotels
        WHERE fetched_at >= NOW() - INTERVAL '1 day'
    """
    
    result = execute_query(query)[0]
    
    print(f"\n   With amenities data: {result['with_amenities']} ({result['amenities_percentage']}%)")
    
    print("\n✅ Data quality check complete")


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

task_identify_cities = PythonOperator(
    task_id='identify_cities',
    python_callable=identify_target_cities,
    dag=dag,
)

task_fetch_hotels = PythonOperator(
    task_id='fetch_hotels',
    python_callable=fetch_and_store_hotels,
    dag=dag,
)

task_validate = PythonOperator(
    task_id='validate_data',
    python_callable=validate_hotel_data,
    dag=dag,
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

task_identify_cities >> task_fetch_hotels >> task_validate
