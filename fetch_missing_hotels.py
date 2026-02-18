#!/usr/bin/env python3
"""
Fetch hotels for top recommended cities that don't have hotel data yet
"""

import sys
import time
sys.path.append('src')

from data.db import get_connection
from data.fetch_hotels import fetch_hotels_for_city
from datetime import datetime

def fetch_hotels_for_top_cities():
    """Fetch hotels for top cities missing hotel data"""
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Get top cities without hotels
        cursor.execute('''
            SELECT r.city, r.rank, r.comfort_score_pred
            FROM recommendations r
            LEFT JOIN hotels h ON r.city = h.city
            WHERE h.city IS NULL
            ORDER BY r.rank
            LIMIT 5
        ''')
        
        cities_needing_hotels = cursor.fetchall()
        
        if not cities_needing_hotels:
            print("✅ All top cities already have hotels!")
            return
        
        print(f"🏨 Fetching hotels for {len(cities_needing_hotels)} cities...")
        
        for city, rank, score in cities_needing_hotels:
            print(f"\n{rank}. Fetching hotels for {city} (Score: {score:.2f})")
            
            try:
                # Use the existing hotel fetching function
                hotels = fetch_hotels_for_city(city)
                
                if hotels:
                    print(f"   ✅ Found {len(hotels)} hotels for {city}")
                    
                    # Insert into database
                    for hotel in hotels:
                        cursor.execute("""
                            INSERT INTO hotels (
                                city, hotel_name, hotel_type, stars, address,
                                latitude, longitude, amenities, data_source, 
                                fetched_at, website_url
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            city,
                            hotel.get('name', 'Unknown'),
                            hotel.get('type', 'hotel'), 
                            hotel.get('stars'),
                            hotel.get('address'),
                            hotel.get('lat'),
                            hotel.get('lon'),
                            ','.join(hotel.get('amenities', [])) if hotel.get('amenities') else None,
                            'openstreetmap',
                            datetime.now(),
                            hotel.get('website_url')
                        ))
                    
                    conn.commit()
                    print(f"   💾 Saved {len(hotels)} hotels to database")
                else:
                    print(f"   ❌ No hotels found for {city}")
                    
            except Exception as e:
                print(f"   ❌ Error fetching hotels for {city}: {e}")
            
            # Rate limiting (be nice to OpenStreetMap)
            time.sleep(2)
        
        print(f"\n✅ Hotel fetching complete!")

if __name__ == "__main__":
    print("🚀 Fetching hotels for top cities...")
    fetch_hotels_for_top_cities()
    
    # Show final summary
    print("\n📊 Final Summary:")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT r.city, r.rank, COUNT(h.id) as hotel_count
            FROM recommendations r
            LEFT JOIN hotels h ON r.city = h.city
            GROUP BY r.city, r.rank
            ORDER BY r.rank
            LIMIT 5
        ''')
        
        results = cursor.fetchall()
        for city, rank, hotel_count in results:
            status = '✅' if hotel_count > 0 else '❌'
            print(f"  {rank}. {city:15s} - {hotel_count:2d} hotels {status}")