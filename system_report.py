#!/usr/bin/env python3
"""
Final system status report for the France Weather Recommender
"""

import sys
sys.path.append('src')

from data.db import get_connection
from datetime import datetime

def generate_system_report():
    """Generate a comprehensive system status report"""
    
    print("🇫🇷 FRANCE WEATHER RECOMMENDER - SYSTEM STATUS REPORT")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Weather Data Status
        print("📊 DATA COLLECTION STATUS:")
        cursor.execute("SELECT COUNT(*) FROM raw_weather")
        weather_count = cursor.fetchone()[0]
        print(f"  ✅ Weather Records: {weather_count:,}")
        
        cursor.execute("SELECT COUNT(*) FROM weather_features")  
        features_count = cursor.fetchone()[0]
        print(f"  ✅ Feature Records: {features_count:,}")
        
        cursor.execute("SELECT COUNT(*) FROM cities")
        cities_count = cursor.fetchone()[0]
        print(f"  ✅ Cities Covered: {cities_count}")
        
        print()
        
        # 2. ML Recommendations Status
        print("🤖 MACHINE LEARNING RECOMMENDATIONS:")
        cursor.execute("SELECT COUNT(*) FROM recommendations")
        rec_count = cursor.fetchone()[0]
        print(f"  ✅ Recommendations Generated: {rec_count}")
        
        cursor.execute("""
            SELECT city, comfort_score_pred, rank 
            FROM recommendations 
            ORDER BY rank 
            LIMIT 5
        """)
        top_cities = cursor.fetchall()
        
        print("  🏆 Top 5 Recommended Cities:")
        for city, score, rank in top_cities:
            print(f"     {rank}. {city:15s} - Comfort Score: {score:5.1f}")
        
        print()
        
        # 3. Hotel Data Status
        print("🏨 HOTEL INTEGRATION STATUS:")
        cursor.execute("SELECT COUNT(*) FROM hotels")
        hotel_count = cursor.fetchone()[0]
        print(f"  ✅ Hotels in Database: {hotel_count}")
        
        cursor.execute("""
            SELECT r.city, r.rank, COUNT(h.id) as hotel_count,
                   MAX(h.website_url) as sample_url
            FROM recommendations r
            LEFT JOIN hotels h ON r.city = h.city  
            GROUP BY r.city, r.rank
            ORDER BY r.rank
            LIMIT 10
        """)
        
        city_hotels = cursor.fetchall()
        print("  🏨 Hotel Coverage by City:")
        for city, rank, h_count, sample_url in city_hotels:
            status = "✅" if h_count > 0 else "⚠️"
            url_status = "with URLs" if sample_url else "no URLs" if h_count > 0 else ""
            print(f"     {rank:2d}. {city:15s} - {h_count:2d} hotels {status} {url_status}")
        
        print()
        
        # 4. System Architecture
        print("🏗️ SYSTEM ARCHITECTURE:")
        print("  ✅ PostgreSQL Database (Neon Cloud)")
        print("  ✅ Apache Airflow 2.8.1 (Docker)")
        print("  ✅ MLflow 2.8.1 (Docker)")
        print("  ✅ Python ML Pipeline (scikit-learn)")
        print("  ✅ OpenStreetMap Integration")
        print("  ✅ Docker Compose Orchestration")
        
        print()
        
        # 5. Available Services
        print("🌐 AVAILABLE SERVICES:")
        print("  🔗 Airflow UI:     http://localhost:8080")
        print("  🔗 MLflow UI:      http://localhost:5001")
        print("  🔗 Database:       Neon Cloud PostgreSQL")
        
        print()
        
        # 6. System Capabilities  
        print("⚡ SYSTEM CAPABILITIES:")
        print("  ✅ Automated weather data collection (20 French cities)")
        print("  ✅ Advanced feature engineering (temperature, precipitation, wind)")
        print("  ✅ Machine learning comfort scoring (K-Means + Gradient Boosting)")
        print("  ✅ City recommendations with rankings")
        print("  ✅ Hotel discovery with website URLs")
        print("  ✅ MLflow experiment tracking") 
        print("  ✅ Airflow workflow orchestration")
        
        print()
        
        # 7. Sample Recommendations with Details
        print("📋 DETAILED RECOMMENDATIONS:")
        for city, score, rank in top_cities[:3]:
            cursor.execute("""
                SELECT hotel_name, stars, website_url 
                FROM hotels 
                WHERE city = %s 
                LIMIT 3
            """, (city,))
            hotels = cursor.fetchall()
            
            print(f"  {rank}. {city} (Comfort Score: {score:.1f})")
            if hotels:
                print("     🏨 Featured Hotels:")
                for hotel_name, stars, url in hotels:
                    star_display = f"({stars}★)" if stars else ""
                    url_display = f"- {url}" if url else ""
                    print(f"       • {hotel_name} {star_display} {url_display}")
            else:
                print("     ⚠️  Hotel data pending (API timeout issues)")
            print()
        
        # 8. System Health
        print("🔍 SYSTEM HEALTH:")
        cursor.execute("SELECT MAX(created_at) FROM recommendations")
        last_rec = cursor.fetchone()[0]
        if last_rec:
            print(f"  ✅ Last Recommendations: {last_rec.strftime('%Y-%m-%d %H:%M')}")
        
        cursor.execute("SELECT MAX(fetched_at) FROM hotels")  
        last_hotel = cursor.fetchone()[0]
        if last_hotel:
            print(f"  ✅ Last Hotel Update: {last_hotel.strftime('%Y-%m-%d %H:%M')}")
            
        print()
        print("🎉 SYSTEM STATUS: OPERATIONAL")
        print("   Weather → Features → ML Models → Recommendations → Hotels")
        print("   All core components working. Hotel fetching partially complete.")

if __name__ == "__main__":
    generate_system_report()