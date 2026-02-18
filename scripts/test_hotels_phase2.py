"""
test_hotels_phase2.py
---------------------
Test script for Phase 2 hotels layer.

Tests:
1. Database migration (hotels table creation)
2. Overpass API fetch for sample city
3. Hotel data insertion
4. Recommendation + hotel join

Run this BEFORE deploying to Airflow to verify everything works.

Usage:
    python scripts/test_hotels_phase2.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.fetch_hotels import fetch_hotels_for_city, fetch_hotels_for_cities
from src.data.db import bulk_insert, execute_query, get_connection
import pandas as pd


def test_database_schema():
    """Test 1: Verify hotels table exists."""
    print("\n" + "="*70)
    print("TEST 1: Database Schema")
    print("="*70)
    
    query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'hotels'
        ORDER BY ordinal_position
    """
    
    try:
        columns = execute_query(query)
        
        if not columns:
            print("❌ FAIL: hotels table does not exist")
            print("   Run: python scripts/add_hotels_table.py")
            return False
        
        print("✅ PASS: hotels table exists")
        print("\nColumns:")
        for col in columns:
            print(f"   • {col['column_name']}: {col['data_type']}")
        
        return True
    
    except Exception as e:
        print(f"❌ FAIL: Database error: {e}")
        return False


def test_overpass_api():
    """Test 2: Fetch hotels from Overpass API."""
    print("\n" + "="*70)
    print("TEST 2: Overpass API Fetch")
    print("="*70)
    
    print("\nFetching hotels for Paris (5km radius, max 5 results)...")
    
    try:
        hotels = fetch_hotels_for_city("Paris", radius_meters=5000, max_results=5)
        
        if not hotels:
            print("❌ FAIL: No hotels returned")
            print("   This might be a temporary Overpass API issue. Try again.")
            return False
        
        print(f"✅ PASS: Fetched {len(hotels)} hotels")
        
        print("\nSample hotel:")
        sample = hotels[0]
        for key, value in sample.items():
            print(f"   {key}: {value}")
        
        return True
    
    except Exception as e:
        print(f"❌ FAIL: API error: {e}")
        return False


def test_hotel_insertion():
    """Test 3: Insert hotels into database."""
    print("\n" + "="*70)
    print("TEST 3: Hotel Data Insertion")
    print("="*70)
    
    print("\nFetching hotels for Nice...")
    
    try:
        hotels = fetch_hotels_for_city("Nice", radius_meters=5000, max_results=3)
        
        if not hotels:
            print("⚠️  No hotels fetched, skipping insertion test")
            return True
        
        print(f"Inserting {len(hotels)} hotels...")
        
        bulk_insert(
            table='hotels',
            rows=hotels,
            conflict_action='(city, hotel_name) DO UPDATE SET '
                           'hotel_type = EXCLUDED.hotel_type, '
                           'stars = EXCLUDED.stars, '
                           'fetched_at = EXCLUDED.fetched_at'
        )
        
        # Verify insertion
        query = "SELECT COUNT(*) as count FROM hotels WHERE city = 'Nice'"
        result = execute_query(query)[0]
        
        print(f"✅ PASS: {result['count']} hotels in database for Nice")
        
        # Show sample
        query = "SELECT hotel_name, stars FROM hotels WHERE city = 'Nice' LIMIT 3"
        samples = execute_query(query)
        
        print("\nSample hotels in database:")
        for hotel in samples:
            stars = f"({hotel['stars']}⭐)" if hotel['stars'] else "(unrated)"
            print(f"   • {hotel['hotel_name']} {stars}")
        
        return True
    
    except Exception as e:
        print(f"❌ FAIL: Insertion error: {e}")
        return False


def test_recommendation_hotel_join():
    """Test 4: Join recommendations with hotels."""
    print("\n" + "="*70)
    print("TEST 4: Recommendation + Hotel Join")
    print("="*70)
    
    # Check if we have recommendations
    query = """
        SELECT city, comfort_score_pred, rank
        FROM recommendations
        WHERE recommendation_date = (SELECT MAX(recommendation_date) FROM recommendations)
        ORDER BY rank
        LIMIT 3
    """
    
    try:
        recs = execute_query(query)
        
        if not recs:
            print("⚠️  No recommendations found")
            print("   Run: python -c \"from dags.dag_generate_recommendations import generate_recommendations; generate_recommendations()\"")
            return True
        
        print(f"✅ Found {len(recs)} recommendations")
        
        # Join with hotels
        cities = [r['city'] for r in recs]
        
        hotels_query = """
            SELECT city, hotel_name, stars
            FROM hotels
            WHERE city = ANY(%s)
            ORDER BY city, stars DESC NULLS LAST
        """
        
        hotels = execute_query(hotels_query, (cities,))
        
        if not hotels:
            print("⚠️  No hotels found for recommended cities")
            print("   This is okay if you haven't run dag_fetch_hotels yet")
            return True
        
        # Group by city
        hotels_by_city = {}
        for hotel in hotels:
            city = hotel['city']
            if city not in hotels_by_city:
                hotels_by_city[city] = []
            hotels_by_city[city].append(hotel)
        
        print(f"\n✅ PASS: Successfully joined recommendations with hotels")
        print("\nRecommendations with hotels:")
        
        for rec in recs:
            print(f"\n   {rec['rank']}. {rec['city']} (score: {rec['comfort_score_pred']:.1f}/100)")
            
            if rec['city'] in hotels_by_city:
                city_hotels = hotels_by_city[rec['city']][:5]
                print(f"      Hotels ({len(city_hotels)}):")
                for hotel in city_hotels:
                    stars_display = "⭐" * (hotel['stars'] or 0) if hotel['stars'] else ""
                    print(f"         • {hotel['hotel_name']} {stars_display}")
            else:
                print(f"      No hotels data yet")
        
        return True
    
    except Exception as e:
        print(f"❌ FAIL: Join error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("PHASE 2 HOTELS LAYER — TESTING SUITE")
    print("="*70)
    
    results = []
    
    # Test 1: Database schema
    results.append(("Database Schema", test_database_schema()))
    
    # Test 2: Overpass API
    results.append(("Overpass API", test_overpass_api()))
    
    # Test 3: Hotel insertion
    results.append(("Hotel Insertion", test_hotel_insertion()))
    
    # Test 4: Recommendation join
    results.append(("Recommendation Join", test_recommendation_hotel_join()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nNext steps:")
        print("   1. Start Airflow: docker-compose up -d")
        print("   2. Enable dag_fetch_hotels in Airflow UI")
        print("   3. Trigger manually to fetch hotels for top cities")
        print("   4. Verify recommendations now include hotel data")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("   Fix the issues above before deploying to Airflow")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
