"""
fetch_hotels.py
---------------
Hotel data fetching from OpenStreetMap via Overpass API.

Fetches hotels for cities and extracts:
- Basic info (name, type, stars, address)
- Location (lat/lon for mapping)
- Website URLs (from OSM contact:website tag)
- Amenities (wifi, parking, restaurant, etc.)
"""

import requests
import time
from typing import List, Dict, Optional
from .db import execute_query


def get_top_cities_from_recommendations(n: int = 3) -> List[str]:
    """
    Get top N cities from latest recommendations.
    
    Args:
        n: Number of top cities to return
        
    Returns:
        List of city names sorted by recommendation score
    """
    query = """
        SELECT city 
        FROM recommendations 
        WHERE created_at = (SELECT MAX(created_at) FROM recommendations)
        ORDER BY comfort_score_pred DESC 
        LIMIT %s
    """
    
    results = execute_query(query, (n,))
    
    if not results:
        return []
    
    return [row['city'] for row in results]


def fetch_hotels_for_cities(
    cities: List[str], 
    radius_meters: int = 10000,
    max_results_per_city: int = 10,
    delay_seconds: float = 2.0
) -> List[Dict]:
    """
    Fetch hotels for multiple cities from OpenStreetMap.
    
    Args:
        cities: List of city names
        radius_meters: Search radius around city center
        max_results_per_city: Maximum hotels per city
        delay_seconds: Delay between API calls for rate limiting
        
    Returns:
        List of hotel dictionaries with all fields including website_url
    """
    all_hotels = []
    
    for city in cities:
        print(f"\n🏨 Fetching hotels for {city}...")
        
        try:
            city_hotels = fetch_hotels_for_city(
                city=city,
                radius_meters=radius_meters,
                max_results=max_results_per_city
            )
            
            all_hotels.extend(city_hotels)
            print(f"   ✅ Found {len(city_hotels)} hotels")
            
        except Exception as e:
            print(f"   ❌ Error fetching hotels for {city}: {e}")
            continue
        
        # Rate limiting
        if city != cities[-1]:  # Don't delay after last city
            time.sleep(delay_seconds)
    
    return all_hotels


def fetch_hotels_for_city(
    city: str, 
    radius_meters: int = 10000,
    max_results: int = 10
) -> List[Dict]:
    """
    Fetch hotels for a single city from OpenStreetMap.
    
    Args:
        city: City name
        radius_meters: Search radius around city center
        max_results: Maximum hotels to return
        
    Returns:
        List of hotel dictionaries
    """
    # First, get city coordinates
    city_coords = get_city_coordinates(city)
    if not city_coords:
        raise ValueError(f"Could not find coordinates for {city}")
    
    lat, lon = city_coords
    
    # Rate limiting between geocoding and Overpass API
    time.sleep(2.0)
    
    # Overpass API query for hotels
    overpass_query = f"""
    [out:json][timeout:30];
    (
      nwr["tourism"="hotel"](around:{radius_meters},{lat},{lon});
      nwr["tourism"="guest_house"](around:{radius_meters},{lat},{lon});
      nwr["tourism"="apartment"](around:{radius_meters},{lat},{lon});
      nwr["tourism"="hostel"](around:{radius_meters},{lat},{lon});
    );
    out center meta;
    """
    
    # Query Overpass API
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    headers = {
        'User-Agent': 'WeatherRecommender/1.0 (github.com/user/weather-recommender)'
    }
    
    try:
        response = requests.post(
            overpass_url,
            data=overpass_query,
            headers=headers,
            timeout=45  # Increased timeout for Overpass API
        )
        response.raise_for_status()
        
        data = response.json()
        elements = data.get('elements', [])
        
    except requests.exceptions.Timeout:
        print(f"   ⚠️  Overpass API timeout for {city}, trying with smaller radius...")
        # Retry with smaller radius
        smaller_query = overpass_query.replace(f"around:{radius_meters}", f"around:{radius_meters//2}")
        try:
            response = requests.post(overpass_url, data=smaller_query, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            elements = data.get('elements', [])
        except Exception as retry_e:
            raise Exception(f"Overpass API failed even with smaller radius: {retry_e}")
            
    except Exception as e:
        raise Exception(f"Overpass API error: {e}")
    
    # Parse hotels
    hotels = []
    for element in elements:
        try:
            hotel = parse_hotel_element(element, city)
            if hotel:
                hotels.append(hotel)
        except Exception as e:
            print(f"   Warning: Error parsing hotel element: {e}")
            continue
    
    # Sort by stars (highest first), then by name
    hotels.sort(key=lambda h: (-(h['stars'] or 0), h['hotel_name']))
    
    # Return top results
    return hotels[:max_results]


def parse_hotel_element(element: Dict, city: str) -> Optional[Dict]:
    """
    Parse a single hotel element from Overpass API response.
    
    Args:
        element: Raw element from Overpass API
        city: City name for context
        
    Returns:
        Hotel dictionary or None if invalid
    """
    tags = element.get('tags', {})
    
    # Skip elements without name
    name = tags.get('name')
    if not name:
        return None
    
    # Get coordinates (handle different element types)
    if element['type'] == 'node':
        lat, lon = element['lat'], element['lon']
    elif 'center' in element:
        lat, lon = element['center']['lat'], element['center']['lon']
    else:
        return None
    
    # Determine hotel type
    hotel_type = tags.get('tourism', 'hotel')
    
    # Extract star rating (multiple possible tags)
    stars = None
    for star_key in ['stars', 'star', 'rating']:
        if star_key in tags:
            try:
                stars = int(tags[star_key])
                if 1 <= stars <= 5:
                    break
            except (ValueError, TypeError):
                continue
    
    # Extract address components
    address_parts = []
    for addr_key in ['addr:housenumber', 'addr:street', 'addr:city', 'addr:postcode']:
        if tags.get(addr_key):
            address_parts.append(tags[addr_key])
    
    address = ', '.join(address_parts) if address_parts else None
    
    # Extract website URL (multiple possible tags)
    website_url = None
    for url_key in ['website', 'contact:website', 'url', 'homepage']:
        if tags.get(url_key):
            website_url = tags[url_key].strip()
            # Ensure URL has protocol
            if website_url and not website_url.startswith(('http://', 'https://')):
                website_url = 'https://' + website_url
            break
    
    # Extract amenities
    amenities = []
    amenity_mapping = {
        'internet_access': 'wifi',
        'wifi': 'wifi', 
        'parking': 'parking',
        'restaurant': 'restaurant',
        'bar': 'bar',
        'breakfast': 'breakfast',
        'air_conditioning': 'ac',
        'swimming_pool': 'pool',
        'spa': 'spa',
        'fitness_centre': 'gym',
        'wheelchair': 'accessible'
    }
    
    for osm_key, amenity_name in amenity_mapping.items():
        if tags.get(osm_key) in ['yes', 'true', '1']:
            amenities.append(amenity_name)
    
    amenities_str = ','.join(amenities) if amenities else None
    
    return {
        'city': city,
        'hotel_name': name,
        'hotel_type': hotel_type,
        'stars': stars,
        'address': address,
        'latitude': float(lat),
        'longitude': float(lon),
        'website_url': website_url,
        'amenities': amenities_str,
        'data_source': 'openstreetmap'
    }


def get_city_coordinates(city: str) -> Optional[tuple]:
    """
    Get coordinates for a city using Nominatim API.
    
    Args:
        city: City name
        
    Returns:
        (latitude, longitude) tuple or None
    """
    # Use Nominatim API for geocoding with proper rate limiting
    nominatim_url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': f"{city}, France",
        'format': 'json',
        'limit': 1,
        'addressdetails': 1
    }
    
    headers = {
        'User-Agent': 'WeatherRecommender/1.0 (github.com/user/weather-recommender)'
    }
    
    try:
        # Add delay to respect rate limits (max 1 request per second)
        time.sleep(1.1)
        
        response = requests.get(
            nominatim_url, 
            params=params, 
            headers=headers, 
            timeout=15  # Increased timeout
        )
        response.raise_for_status()
        
        results = response.json()
        if results:
            return float(results[0]['lat']), float(results[0]['lon'])
            
    except Exception as e:
        print(f"   Error geocoding {city}: {e}")
        
        # Fallback coordinates for major French cities
        city_fallbacks = {
            'paris': (48.8566, 2.3522),
            'lyon': (45.7640, 4.8357),
            'marseille': (43.2965, 5.3698),
            'nice': (43.7102, 7.2620),
            'toulouse': (43.6047, 1.4442),
            'strasbourg': (48.5734, 7.7521),
            'bordeaux': (44.8378, -0.5792),
            'lille': (50.6292, 3.0573),
            'rennes': (48.1173, -1.6778),
            'nantes': (47.2184, -1.5536)
        }
        
        city_lower = city.lower()
        if city_lower in city_fallbacks:
            print(f"   Using fallback coordinates for {city}")
            return city_fallbacks[city_lower]
    
    return None