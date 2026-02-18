"""
add_hotels_table.py
-------------------
Database migration: Add hotels table for Phase 2.

Run once to add the hotels table to your Neon database.

Usage:
    python scripts/add_hotels_table.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.db import get_connection

HOTELS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS hotels (
    id              SERIAL PRIMARY KEY,
    city            VARCHAR(100)    NOT NULL,
    hotel_name      VARCHAR(200)    NOT NULL,
    hotel_type      VARCHAR(50),        -- 'hotel', 'guest_house', 'apartment', 'hostel'
    stars           INTEGER,            -- 1-5 star rating (if available from OSM)
    address         TEXT,
    latitude        FLOAT               NOT NULL,
    longitude       FLOAT               NOT NULL,
    website_url     TEXT,               -- Hotel website or booking URL
    amenities       TEXT,               -- Comma-separated list: 'wifi,parking,restaurant'
    data_source     VARCHAR(50)         DEFAULT 'openstreetmap',
    fetched_at      TIMESTAMPTZ         DEFAULT NOW(),
    UNIQUE(city, hotel_name)            -- Prevent duplicate hotels
);

-- Index for faster city lookups
CREATE INDEX IF NOT EXISTS idx_hotels_city ON hotels(city);

-- Index for quality filtering
CREATE INDEX IF NOT EXISTS idx_hotels_stars ON hotels(stars DESC NULLS LAST);
"""


def add_hotels_table():
    """Add the hotels table to the database."""
    print("=" * 70)
    print("DATABASE MIGRATION: Adding hotels table")
    print("=" * 70)
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(HOTELS_TABLE_SQL)
    
    print("\n✅ Hotels table created successfully!")
    print("\nTable structure:")
    print("  - id (primary key)")
    print("  - city, hotel_name (unique constraint)")
    print("  - hotel_type, stars, address")
    print("  - latitude, longitude (for mapping)")
    print("  - website_url (hotel website or booking links)")
    print("  - amenities, data_source, fetched_at")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    add_hotels_table()
